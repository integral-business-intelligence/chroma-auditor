import json
import pytest
import chromadb
from conftest import auditor, COLLECTION


class TestGetFilesets:
    def test_returns_sorted_unique_filesets(self, db_path):
        result = auditor.get_filesets(db_path, COLLECTION)
        assert result == ["project_x", "project_y"]

    def test_pipe_delimited_chunk_contributes_to_both_filesets(self, db_path):
        # b1 has fileset="project_x|project_y" — both must appear
        result = auditor.get_filesets(db_path, COLLECTION)
        assert "project_x" in result
        assert "project_y" in result

    def test_collection_with_no_filesets_returns_empty_list(self, db_path):
        client = chromadb.PersistentClient(path=db_path)
        client.create_collection("no_filesets")
        assert auditor.get_filesets(db_path, "no_filesets") == []

    def test_nonexistent_collection_returns_empty_list(self, db_path):
        assert auditor.get_filesets(db_path, "nonexistent") == []


class TestGetUniqueFilenames:
    def test_returns_sorted_unique_basenames(self, db_path):
        result = auditor.get_unique_filenames(db_path, COLLECTION)
        assert result == ["data.txt", "notes.txt", "report.txt"]

    def test_nonexistent_collection_returns_empty_list(self, db_path):
        assert auditor.get_unique_filenames(db_path, "nonexistent") == []


class TestLoadCollection:
    def test_has_correct_columns(self, db_path):
        df = auditor.load_collection(COLLECTION, db_path)
        assert list(df.columns) == ["Selected", "Metadata", "File Chunk", "ID"]

    def test_returns_all_chunks(self, db_path):
        df = auditor.load_collection(COLLECTION, db_path)
        assert len(df) == 6

    def test_all_rows_default_to_not_selected(self, db_path):
        df = auditor.load_collection(COLLECTION, db_path)
        assert (df["Selected"] == "Not Selected").all()

    def test_sorted_by_source_file_then_chunk_index(self, db_path):
        df = auditor.load_collection(COLLECTION, db_path)
        source_files = [json.loads(m)["source_file"] for m in df["Metadata"]]
        # Alphabetical: data.txt (1 chunk), notes.txt (2 chunks), report.txt (3 chunks)
        assert source_files[0] == "data.txt"
        assert source_files[1:3] == ["notes.txt", "notes.txt"]
        assert source_files[3:] == ["report.txt", "report.txt", "report.txt"]

    def test_nonexistent_collection_returns_empty_df(self, db_path):
        df = auditor.load_collection("nonexistent", db_path)
        assert df.empty


class TestLoadFileChunks:
    def test_returns_correct_chunks_for_file(self, db_path):
        df, status = auditor.load_file_chunks(db_path, "report.txt", COLLECTION)
        assert len(df) == 3
        assert "3" in status

    def test_chunks_sorted_by_index(self, db_path):
        df, _ = auditor.load_file_chunks(db_path, "report.txt", COLLECTION)
        indices = [json.loads(m)["chunk_index"] for m in df["Metadata"]]
        assert indices == [1, 2, 3]

    def test_empty_filename_returns_empty_df(self, db_path):
        df, _ = auditor.load_file_chunks(db_path, "", COLLECTION)
        assert df.empty

    def test_nonexistent_file_returns_empty_df(self, db_path):
        df, _ = auditor.load_file_chunks(db_path, "ghost.txt", COLLECTION)
        assert df.empty


class TestLoadFilesetDocuments:
    def test_returns_all_chunks_in_fileset(self, db_path):
        df, _ = auditor.load_fileset_documents("project_x", COLLECTION, db_path)
        # a1, a2, a3 from report.txt + b1 from notes.txt (pipe-delimited)
        assert len(df) == 4
        assert set(df["ID"]) == {"a1", "a2", "a3", "b1"}

    def test_pipe_delimited_chunk_appears_in_both_filesets(self, db_path):
        df_x, _ = auditor.load_fileset_documents("project_x", COLLECTION, db_path)
        df_y, _ = auditor.load_fileset_documents("project_y", COLLECTION, db_path)
        assert "b1" in set(df_x["ID"])
        assert "b1" in set(df_y["ID"])

    def test_nonexistent_fileset_returns_empty_df(self, db_path):
        df, _ = auditor.load_fileset_documents("nonexistent", COLLECTION, db_path)
        assert df.empty


class TestCheckCollectionForFiles:
    def test_returns_true_when_source_file_metadata_present(self, db_path):
        has_files, msg = auditor.check_collection_for_files(db_path, COLLECTION)
        assert has_files is True
        assert msg is None

    def test_returns_false_and_warning_for_empty_collection(self, db_path):
        client = chromadb.PersistentClient(path=db_path)
        client.create_collection("empty_col")
        has_files, msg = auditor.check_collection_for_files(db_path, "empty_col")
        assert has_files is False
        assert msg is not None
