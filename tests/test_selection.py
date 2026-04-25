import os
import pytest
import pandas as pd
from conftest import auditor


@pytest.fixture()
def simple_df():
    """Minimal four-row DataFrame for testing selection helpers."""
    return pd.DataFrame({
        "Selected": ["Not Selected"] * 4,
        "Metadata": ["{}"] * 4,
        "File Chunk": ["chunk a", "chunk b", "chunk c", "chunk d"],
        "ID": ["id1", "id2", "id3", "id4"],
    })


class TestUpdateSelectionState:
    def test_marks_specified_rows_as_selected(self, simple_df):
        result = auditor.update_selection_state([0, 2], simple_df)
        assert result.iloc[0]["Selected"] == "Selected"
        assert result.iloc[2]["Selected"] == "Selected"

    def test_leaves_other_rows_as_not_selected(self, simple_df):
        result = auditor.update_selection_state([0], simple_df)
        assert result.iloc[1]["Selected"] == "Not Selected"
        assert result.iloc[3]["Selected"] == "Not Selected"

    def test_empty_indices_marks_nothing_selected(self, simple_df):
        result = auditor.update_selection_state([], simple_df)
        assert (result["Selected"] == "Not Selected").all()

    def test_does_not_mutate_original_dataframe(self, simple_df):
        auditor.update_selection_state([0, 1], simple_df)
        assert (simple_df["Selected"] == "Not Selected").all()


class TestHandleSelectAll:
    def test_returns_all_positional_indices(self, simple_df):
        indices, _ = auditor.handle_select_all(simple_df)
        assert indices == [0, 1, 2, 3]

    def test_all_rows_marked_selected(self, simple_df):
        _, result = auditor.handle_select_all(simple_df)
        assert (result["Selected"] == "Selected").all()


class TestHandleClearSelection:
    def test_returns_empty_index_list(self, simple_df):
        indices, _ = auditor.handle_clear_selection(simple_df)
        assert indices == []

    def test_all_rows_marked_not_selected(self, simple_df):
        simple_df["Selected"] = "Selected"
        _, result = auditor.handle_clear_selection(simple_df)
        assert (result["Selected"] == "Not Selected").all()


class TestExportSelectedChunks:
    def test_creates_csv_file_on_disk(self, simple_df, tmp_path, monkeypatch):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        path = auditor.export_selected_chunks([0, 2], simple_df)
        assert path is not None
        assert os.path.isfile(path)

    def test_exported_file_contains_selected_rows(self, simple_df, tmp_path, monkeypatch):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        path = auditor.export_selected_chunks([1, 3], simple_df)
        import pandas as pd
        exported = pd.read_csv(path)
        assert set(exported["ID"]) == {"id2", "id4"}

    def test_returns_none_when_no_rows_selected(self, simple_df):
        result = auditor.export_selected_chunks([], simple_df)
        assert result is None
