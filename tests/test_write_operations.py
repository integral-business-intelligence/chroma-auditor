import json
import pytest
import chromadb
from conftest import auditor, COLLECTION, row_pos


def _get_metadata(db_path, chunk_id):
    """Re-fetch a single chunk's metadata directly from ChromaDB."""
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(COLLECTION)
    return col.get(ids=[chunk_id])["metadatas"][0]


class TestAddMetadata:
    def test_adds_new_key_to_selected_chunk(self, db_path, loaded_df):
        pos = row_pos(loaded_df, "a1")
        auditor.add_metadata(
            db_path, [pos], "status", "reviewed",
            loaded_df, "collection", COLLECTION, COLLECTION,
        )
        assert _get_metadata(db_path, "a1")["status"] == "reviewed"

    def test_adds_to_pipe_delimited_fileset_without_overwriting(self, db_path, loaded_df):
        # b2 starts with fileset="project_y"; add it to "project_z"
        pos = row_pos(loaded_df, "b2")
        auditor.add_metadata(
            db_path, [pos], "fileset", "project_z",
            loaded_df, "collection", COLLECTION, COLLECTION,
        )
        meta = _get_metadata(db_path, "b2")
        filesets = {f.strip() for f in meta["fileset"].split("|")}
        assert "project_y" in filesets
        assert "project_z" in filesets

    def test_updates_multiple_chunks_at_once(self, db_path, loaded_df):
        positions = [row_pos(loaded_df, "a1"), row_pos(loaded_df, "a2")]
        auditor.add_metadata(
            db_path, positions, "reviewed_by", "alice",
            loaded_df, "collection", COLLECTION, COLLECTION,
        )
        assert _get_metadata(db_path, "a1")["reviewed_by"] == "alice"
        assert _get_metadata(db_path, "a2")["reviewed_by"] == "alice"

    def test_returns_updated_dataframe(self, db_path, loaded_df):
        pos = row_pos(loaded_df, "c1")
        result_df = auditor.add_metadata(
            db_path, [pos], "tag", "standalone",
            loaded_df, "collection", COLLECTION, COLLECTION,
        )
        assert not result_df.empty


class TestDeleteMetadata:
    def test_removes_single_value_key_via_sqlite(self, db_path, loaded_df):
        # c1 has source_file="data.txt" — single value, triggers the SQLite path
        pos = row_pos(loaded_df, "c1")
        auditor.delete_metadata(
            db_path, [pos], "source_file", "data.txt",
            loaded_df, "collection", COLLECTION, COLLECTION,
        )
        assert "source_file" not in _get_metadata(db_path, "c1")

    def test_removes_one_value_from_pipe_delimited_fileset(self, db_path, loaded_df):
        # b1 has fileset="project_x|project_y"; remove project_x
        pos = row_pos(loaded_df, "b1")
        auditor.delete_metadata(
            db_path, [pos], "fileset", "project_x",
            loaded_df, "collection", COLLECTION, COLLECTION,
        )
        meta = _get_metadata(db_path, "b1")
        assert "project_x" not in meta["fileset"]
        assert "project_y" in meta["fileset"]

    def test_removes_key_entirely_when_last_pipe_value_deleted(self, db_path, loaded_df):
        # b2 has fileset="project_y" (single value, no pipe); deleting it removes the key
        pos = row_pos(loaded_df, "b2")
        auditor.delete_metadata(
            db_path, [pos], "fileset", "project_y",
            loaded_df, "collection", COLLECTION, COLLECTION,
        )
        assert "fileset" not in _get_metadata(db_path, "b2")

    def test_returns_df_unchanged_when_category_missing(self, db_path, loaded_df):
        pos = row_pos(loaded_df, "a1")
        result_df = auditor.delete_metadata(
            db_path, [pos], "nonexistent_key", "any_value",
            loaded_df, "collection", COLLECTION, COLLECTION,
        )
        assert not result_df.empty


class TestDeleteEntries:
    def test_removes_chunk_from_collection(self, db_path, loaded_df):
        pos = row_pos(loaded_df, "c1")
        auditor.delete_entries(
            db_path, COLLECTION, [pos],
            loaded_df, "collection", COLLECTION,
        )
        client = chromadb.PersistentClient(path=db_path)
        col = client.get_collection(COLLECTION)
        assert col.count() == 5

    def test_returns_dataframe_without_deleted_chunk(self, db_path, loaded_df):
        pos = row_pos(loaded_df, "c1")
        updated_df, _ = auditor.delete_entries(
            db_path, COLLECTION, [pos],
            loaded_df, "collection", COLLECTION,
        )
        assert len(updated_df) == 5
        assert "c1" not in set(updated_df["ID"])

    def test_deletes_multiple_chunks(self, db_path, loaded_df):
        positions = [row_pos(loaded_df, "a1"), row_pos(loaded_df, "a2")]
        auditor.delete_entries(
            db_path, COLLECTION, positions,
            loaded_df, "collection", COLLECTION,
        )
        client = chromadb.PersistentClient(path=db_path)
        col = client.get_collection(COLLECTION)
        assert col.count() == 4
