#!/usr/bin/env python3
"""
Chroma Auditor — browse and edit ChromaDB chunk metadata in your terminal.

Point at any ChromaDB PersistentClient directory to inspect collections,
view chunk content and metadata, add or remove metadata keys, delete chunks,
and export selections to CSV.  No knowledge of how the data was ingested is
required or assumed.

Usage:
    python chroma-auditor.py
    python chroma-auditor.py /path/to/chroma/persistent-storage
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback
from datetime import datetime

import chromadb
import pandas as pd
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)

# ─── Configuration ────────────────────────────────────────────────────────────

# Set a default path here if you always work with the same database,
# or pass it as a command-line argument when launching the script.
DEFAULT_CHROMA_PATH = ""

# ─── Database helpers ─────────────────────────────────────────────────────────

def list_collections(db_path: str) -> tuple[list[str], str]:
    """Return (collection_names, status_message) for a ChromaDB directory."""
    try:
        if not db_path:
            return [], "Enter a database path and press Load."
        if not os.path.exists(db_path):
            return [], f"Path not found: {db_path}"
        if not os.path.isfile(os.path.join(db_path, "chroma.sqlite3")):
            return [], f"No ChromaDB database found at: {db_path}"
        client = chromadb.PersistentClient(path=db_path)
        names = client.list_collections()
        if not names:
            return [], "Connected — database has no collections yet."
        return names, f"Connected — {len(names)} collection(s) found."
    except Exception as e:
        return [], f"Connection error: {e}"

# ─── Fileset management ───────────────────────────────────────────────────────
# A "fileset" is a user-defined grouping of chunks stored as pipe-delimited
# metadata: fileset="project_x|project_y".  This lets one chunk belong to
# multiple filesets without Chroma's native array support.

def get_filesets(db_path: str, collection_name: str) -> list[str]:
    """Return sorted unique fileset names from a collection."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        filesets: set[str] = set()
        for metadata in results["metadatas"]:
            if metadata and "fileset" in metadata:
                filesets.update(
                    fs.strip()
                    for fs in metadata["fileset"].split("|")
                    if fs.strip()
                )
        return sorted(filesets)
    except Exception as e:
        print(f"Error getting filesets: {e}")
        return []


def load_fileset_documents(
    fileset_name: str, collection_name: str, db_path: str = DEFAULT_CHROMA_PATH
) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, status) for all chunks belonging to a fileset."""
    empty = pd.DataFrame(columns=["Selected", "Metadata", "File Chunk", "ID"])
    try:
        if not fileset_name or not collection_name:
            return empty, "Missing fileset name or collection name."
        client = chromadb.PersistentClient(path=db_path)
        if collection_name not in client.list_collections():
            return empty, f"Collection '{collection_name}' does not exist."
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        matches = [
            (doc_id, document, metadata)
            for doc_id, document, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
            if metadata
            and "fileset" in metadata
            and fileset_name
            in [fs.strip() for fs in metadata["fileset"].split("|")]
        ]
        if not matches:
            return empty, f"No chunks found in fileset '{fileset_name}'."
        ids, docs, metadatas = zip(*matches)
        df = pd.DataFrame({
            "Selected": ["Not Selected"] * len(ids),
            "Metadata": [json.dumps(m, indent=2) for m in metadatas],
            "File Chunk": docs,
            "ID": ids,
        })
        try:
            df["_src"] = df["Metadata"].apply(
                lambda x: json.loads(x).get("source_file", "")
            )
            df["_idx"] = df["Metadata"].apply(
                lambda x: json.loads(x).get("chunk_index", 0)
            )
            df = df.sort_values(["_src", "_idx"]).drop(columns=["_src", "_idx"])
        except Exception:
            pass
        return df, f"Loaded {len(df)} chunks from '{fileset_name}'."
    except Exception as e:
        traceback.print_exc()
        return empty, f"Fileset load error: {e}"

# ─── Collection management ────────────────────────────────────────────────────

def load_collection(
    collection_name: str, db_path: str = DEFAULT_CHROMA_PATH
) -> pd.DataFrame:
    """Return a DataFrame of all chunks in a collection."""
    empty = pd.DataFrame(columns=["Selected", "Metadata", "File Chunk", "ID"])
    if not collection_name:
        return empty
    try:
        client = chromadb.PersistentClient(path=db_path)
        if collection_name not in client.list_collections():
            return empty
        collection = client.get_collection(name=collection_name)
        items = collection.get(include=["metadatas", "documents"])
        if not items["ids"]:
            return empty
        df = pd.DataFrame({
            "Selected": ["Not Selected"] * len(items["ids"]),
            "Metadata": [
                json.dumps(m, indent=2) if m else "{}"
                for m in items["metadatas"]
            ],
            "File Chunk": items["documents"],
            "ID": items["ids"],
        })
        if len(df) > 0:
            df["_src"] = df["Metadata"].apply(
                lambda x: json.loads(x).get("source_file", "")
                if isinstance(x, str)
                else ""
            )
            df["_idx"] = df["Metadata"].apply(
                lambda x: int(json.loads(x).get("chunk_index", 0))
                if isinstance(x, str)
                else 0
            )
            df = df.sort_values(["_src", "_idx"]).drop(
                columns=["_src", "_idx"]
            )
        return df.copy()
    except Exception as e:
        traceback.print_exc()
        return empty

# ─── File management ──────────────────────────────────────────────────────────
# Chunks tagged with source_file metadata can be grouped back into virtual
# "files", letting users navigate by the original document rather than by
# individual chunks.

def get_unique_filenames(db_path: str, collection_name: str) -> list[str]:
    """Return sorted unique source_file basenames from a collection."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        return sorted({
            os.path.basename(m["source_file"])
            for m in results["metadatas"]
            if m and "source_file" in m
        })
    except Exception as e:
        print(f"Error getting filenames: {e}")
        return []


def load_file_chunks(
    db_path: str,
    filename: str,
    collection_name: str,
    key: str = "source_file",
) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, status) for all chunks belonging to a file."""
    empty = pd.DataFrame(columns=["Selected", "Metadata", "File Chunk", "ID"])
    if not filename:
        return empty, "No file selected."
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get(where={key: filename})
        if not results["ids"]:
            return empty, "No matching chunks found."
        df = pd.DataFrame({
            "Selected": ["Not Selected"] * len(results["ids"]),
            "Metadata": [
                json.dumps(m, indent=2) if m else "{}"
                for m in results["metadatas"]
            ],
            "File Chunk": results["documents"],
            "ID": results["ids"],
        })
        if key == "source_file":
            df["_idx"] = df["Metadata"].apply(
                lambda x: int(json.loads(x).get("chunk_index", 0))
            )
            df = df.sort_values("_idx").drop(columns=["_idx"])
        else:
            df["_src"] = df["Metadata"].apply(
                lambda x: json.loads(x).get("source_file", "")
            )
            df["_idx"] = df["Metadata"].apply(
                lambda x: int(json.loads(x).get("chunk_index", 0))
            )
            df = df.sort_values(["_src", "_idx"]).drop(
                columns=["_src", "_idx"]
            )
        return df.reset_index(drop=True), f"Successfully loaded {len(df)} chunks."
    except Exception as e:
        traceback.print_exc()
        return empty, f"Error: {e}"


def export_selected_chunks(
    selected_indices: list[int], df: pd.DataFrame
) -> str | None:
    """Write selected rows to a timestamped CSV; return the path or None."""
    try:
        if not selected_indices or df.empty:
            return None
        export_df = df.iloc[selected_indices].drop(columns=["Selected"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            tempfile.gettempdir(), f"chroma_export_{timestamp}.csv"
        )
        export_df.to_csv(path, index=False)
        return path
    except Exception as e:
        print(f"Export error: {e}")
        return None

# ─── Metadata management ──────────────────────────────────────────────────────

def add_metadata(
    db_path, selected_indices, category, value, df, view_type, view_value, collection_name
):
    """Add a metadata key/value to selected chunks; reload and return the view."""
    if not category or not value or not selected_indices:
        return df
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        selected_ids = [df.iloc[idx]["ID"] for idx in selected_indices]
        results = collection.get(ids=selected_ids)
        for i, doc_id in enumerate(selected_ids):
            metadata = results["metadatas"][i] or {}
            if category == "fileset":
                existing = {
                    fs.strip()
                    for fs in metadata.get("fileset", "").split("|")
                    if fs.strip()
                }
                existing.add(value)
                metadata["fileset"] = "|".join(sorted(existing))
            else:
                metadata[category] = value
            collection.update(ids=[doc_id], metadatas=[metadata])
        updated_df = _reload_view(db_path, view_type, view_value, collection_name)
        if not updated_df.empty:
            updated_df["Selected"] = "Not Selected"
            valid = [i for i in selected_indices if i < len(updated_df)]
            if valid:
                updated_df.iloc[
                    valid, updated_df.columns.get_loc("Selected")
                ] = "Selected"
        return updated_df
    except Exception as e:
        traceback.print_exc()
        return df


def delete_metadata(
    db_path, selected_indices, category, value, df, view_type, view_value, collection_name
):
    """Remove a metadata key/value from selected chunks; reload and return the view."""
    if not category or not selected_indices or not collection_name:
        return df
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        selected_ids = [df.iloc[idx]["ID"] for idx in selected_indices]
        results = collection.get(ids=selected_ids)
        conn = sqlite3.connect(os.path.join(db_path, "chroma.sqlite3"))
        cursor = conn.cursor()
        for i, doc_id in enumerate(selected_ids):
            metadata = results["metadatas"][i] or {}
            if category not in metadata:
                continue
            current_value = metadata[category]
            if isinstance(current_value, str) and "|" in current_value:
                values = [v.strip() for v in current_value.split("|")]
                if value in values:
                    values.remove(value)
                    if values:
                        metadata[category] = "|".join(values)
                    else:
                        del metadata[category]
                    collection.update(ids=[doc_id], metadatas=[metadata])
            else:
                if str(current_value) == str(value):
                    cursor.execute(
                        "SELECT id FROM embeddings WHERE embedding_id = ?",
                        (doc_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        cursor.execute(
                            "DELETE FROM embedding_metadata "
                            "WHERE id = ? AND key = ? AND string_value = ?",
                            (row[0], category, value),
                        )
        conn.commit()
        conn.close()
        updated_df = _reload_view(db_path, view_type, view_value, collection_name)
        if not updated_df.empty:
            updated_df["Selected"] = "Not Selected"
            valid = [i for i in selected_indices if i < len(updated_df)]
            if valid:
                updated_df.iloc[
                    valid, updated_df.columns.get_loc("Selected")
                ] = "Selected"
        return updated_df
    except Exception as e:
        traceback.print_exc()
        return df

# ─── Selection helpers ────────────────────────────────────────────────────────

def update_selection_state(indices: list[int], df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the Selected column reflecting the given indices."""
    if df.empty:
        return df
    result = df.copy()
    result["Selected"] = [
        "Selected" if i in indices else "Not Selected" for i in range(len(df))
    ]
    return result


def handle_select_all(df: pd.DataFrame) -> tuple[list[int], pd.DataFrame]:
    """Select all rows; return (indices, updated_df)."""
    indices = list(range(len(df)))
    return indices, update_selection_state(indices, df)


def handle_clear_selection(df: pd.DataFrame) -> tuple[list[int], pd.DataFrame]:
    """Clear all selections; return ([], updated_df)."""
    return [], update_selection_state([], df)

# ─── Chunk deletion ───────────────────────────────────────────────────────────

def delete_entries(
    db_path, collection_name, selected_indices, df, view_type, view_value
) -> tuple[pd.DataFrame, str]:
    """Delete selected chunks from Chroma; return (updated_df, status_message)."""
    if not selected_indices:
        return df, "No chunks selected."
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(collection_name)
        total_count = collection.count()
        selected_ids = [
            df.iloc[idx]["ID"] for idx in selected_indices if idx < len(df)
        ]
        if len(selected_ids) >= total_count:
            # Deleting everything — remove the UUID directory to reclaim disk
            # space, then recreate the empty collection.
            conn = sqlite3.connect(os.path.join(db_path, "chroma.sqlite3"))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT s.id FROM segments s "
                "JOIN collections c ON s.collection = c.id "
                "WHERE c.name = ? AND s.scope = 'VECTOR'",
                (collection_name,),
            )
            uuid_row = cursor.fetchone()
            conn.close()
            client.delete_collection(name=collection_name)
            if uuid_row:
                uuid_path = os.path.join(db_path, uuid_row[0])
                if os.path.exists(uuid_path):
                    shutil.rmtree(uuid_path)
            client.create_collection(name=collection_name)
            status = "Collection cleared and recreated."
        else:
            collection.delete(ids=selected_ids)
            status = f"Deleted {len(selected_ids)} chunk(s)."
        return _reload_view(db_path, view_type, view_value, collection_name), status
    except Exception as e:
        traceback.print_exc()
        return df, f"Error: {e}"

# ─── Collection inspection ────────────────────────────────────────────────────

def check_collection_for_files(
    db_path: str, collection_name: str
) -> tuple[bool, str | None]:
    """Return (has_source_file_metadata, warning_or_None)."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        if not results["ids"]:
            return False, f"Collection '{collection_name}' is empty."
        for m in results["metadatas"]:
            if m and "source_file" in m:
                return True, None
        return False, f"No 'source_file' metadata found in '{collection_name}'."
    except Exception as e:
        return False, f"Error: {e}"

# ─── Internal view-reload helper ─────────────────────────────────────────────

def _reload_view(
    db_path: str, view_type: str, view_value: str, collection_name: str
) -> pd.DataFrame:
    """Re-fetch the current view from Chroma after a mutation."""
    if view_type == "file":
        df, _ = load_file_chunks(db_path, view_value, collection_name)
    elif view_type == "fileset":
        df, _ = load_fileset_documents(view_value, collection_name, db_path)
    else:
        df = load_collection(collection_name, db_path)
    return df

# ─── Textual UI ───────────────────────────────────────────────────────────────

APP_CSS = """
Screen {
    background: $surface;
}

#connection-bar {
    height: 3;
    padding: 0 1;
    background: $panel;
    border-bottom: solid $primary-darken-1;
    align: left middle;
}

#db-path-input {
    width: 1fr;
    margin-right: 1;
}

#nav-bar {
    height: 3;
    padding: 0 1;
    background: $panel;
    border-bottom: solid $primary-darken-2;
    align: left middle;
}

#collection-select {
    width: 28;
    margin-right: 1;
}

#file-select, #fileset-select {
    width: 1fr;
    margin-right: 1;
}

.nav-sep {
    width: 1;
    color: $primary-darken-1;
    content-align: center middle;
    margin: 0 1;
}

#chunk-table {
    height: 1fr;
    border: solid $primary-darken-2;
}

#table-actions {
    height: 3;
    padding: 0 1;
    align: left middle;
    background: $panel;
    border-top: solid $primary-darken-2;
}

#metadata-add-bar, #metadata-del-bar {
    height: 3;
    padding: 0 1;
    align: left middle;
}

.meta-label {
    width: 18;
    color: $text-muted;
    content-align: right middle;
    padding-right: 1;
}

.meta-input {
    width: 24;
    margin-right: 1;
}

Button {
    margin-right: 1;
}

.danger {
    background: $error;
    color: $text;
}

#status-bar {
    height: 1;
    padding: 0 1;
    background: $panel;
    color: $text-muted;
    border-top: solid $primary-darken-2;
}
"""


class ChromaAuditorApp(App):
    """Terminal UI for inspecting and editing ChromaDB chunk metadata."""

    CSS = APP_CSS

    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select All", show=True),
        Binding("ctrl+d", "deselect_all", "Deselect All", show=True),
        Binding("ctrl+e", "export_csv", "Export CSV", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    # ── Internal state ────────────────────────────────────────────────────────

    current_df: pd.DataFrame | None = None
    selected_ids: set[str]
    view_type: str = "collection"
    view_value: str = ""
    _db_path: str = DEFAULT_CHROMA_PATH

    def __init__(self, db_path: str = DEFAULT_CHROMA_PATH):
        super().__init__()
        self._db_path = db_path
        self.selected_ids = set()

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="connection-bar"):
            yield Input(
                value=self._db_path,
                placeholder="/path/to/chroma/persistent-storage",
                id="db-path-input",
            )
            yield Button("Load Database", id="load-db-btn", variant="primary")

        with Horizontal(id="nav-bar"):
            yield Select([], prompt="Collection", id="collection-select", allow_blank=True)
            yield Button("Load All", id="load-all-btn")
            yield Static("│", classes="nav-sep")
            yield Select([], prompt="File", id="file-select", allow_blank=True)
            yield Button("Load File", id="load-file-btn")
            yield Static("│", classes="nav-sep")
            yield Select([], prompt="Fileset", id="fileset-select", allow_blank=True)
            yield Button("Load Fileset", id="load-fileset-btn")

        yield DataTable(id="chunk-table", cursor_type="row", zebra_stripes=True)

        with Horizontal(id="table-actions"):
            yield Button("Select All", id="select-all-btn")
            yield Button("Deselect All", id="deselect-btn")
            yield Button("⚠ Delete Selected", id="delete-chunks-btn", classes="danger")
            yield Button("Export CSV", id="export-btn")

        with Horizontal(id="metadata-add-bar"):
            yield Label("Add metadata:", classes="meta-label")
            yield Input(placeholder="category", id="add-category", classes="meta-input")
            yield Input(placeholder="value", id="add-value", classes="meta-input")
            yield Button("Add", id="add-meta-btn", variant="success")

        with Horizontal(id="metadata-del-bar"):
            yield Label("Delete metadata:", classes="meta-label")
            yield Input(placeholder="category", id="del-category", classes="meta-input")
            yield Input(placeholder="value", id="del-value", classes="meta-input")
            yield Button("Delete", id="del-meta-btn", variant="error")

        yield Label("Ready — load a database to begin.", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#chunk-table", DataTable)
        table.add_column("", key="sel", width=1)
        table.add_column("File", key="file", width=22)
        table.add_column("Metadata", key="meta", width=46)
        table.add_column("Preview", key="preview", width=50)
        table.add_column("ID", key="id", width=18)

        if self._db_path:
            self._load_db(self._db_path)

    # ── Button handlers ───────────────────────────────────────────────────────

    @on(Button.Pressed, "#load-db-btn")
    def handle_load_db(self) -> None:
        path = self.query_one("#db-path-input", Input).value.strip()
        self._load_db(path)

    def _load_db(self, path: str) -> None:
        self._db_path = path
        names, status = list_collections(path)
        self._set_status(status)
        col_select = self.query_one("#collection-select", Select)
        col_select.set_options([(n, n) for n in names])
        self.query_one("#file-select", Select).set_options([])
        self.query_one("#fileset-select", Select).set_options([])
        if len(names) == 1:
            col_select.value = names[0]

    @on(Select.Changed, "#collection-select")
    def handle_collection_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        collection = str(event.value)
        files = get_unique_filenames(self._db_path, collection)
        filesets = get_filesets(self._db_path, collection)
        self.query_one("#file-select", Select).set_options(
            [(f, f) for f in files]
        )
        self.query_one("#fileset-select", Select).set_options(
            [(fs, fs) for fs in filesets]
        )
        self._set_status(
            f"'{collection}' — {len(files)} file(s), {len(filesets)} fileset(s).  "
            "Choose a view below."
        )

    @on(Button.Pressed, "#load-all-btn")
    def handle_load_all(self) -> None:
        collection = self._current_collection()
        if not collection:
            return
        df = load_collection(collection, self._db_path)
        self.view_type = "collection"
        self.view_value = collection
        self._display(df, f"Loaded {len(df)} chunk(s) from '{collection}'.")

    @on(Button.Pressed, "#load-file-btn")
    def handle_load_file(self) -> None:
        collection = self._current_collection()
        filename = self._select_value("#file-select")
        if not collection or not filename:
            self._set_status("Select a collection and a file first.")
            return
        df, status = load_file_chunks(self._db_path, filename, collection)
        self.view_type = "file"
        self.view_value = filename
        self._display(df, status)

    @on(Button.Pressed, "#load-fileset-btn")
    def handle_load_fileset(self) -> None:
        collection = self._current_collection()
        fileset = self._select_value("#fileset-select")
        if not collection or not fileset:
            self._set_status("Select a collection and a fileset first.")
            return
        df, status = load_fileset_documents(fileset, collection, self._db_path)
        self.view_type = "fileset"
        self.view_value = fileset
        self._display(df, status)

    @on(Button.Pressed, "#select-all-btn")
    def handle_select_all_btn(self) -> None:
        self.action_select_all()

    @on(Button.Pressed, "#deselect-btn")
    def handle_deselect_btn(self) -> None:
        self.action_deselect_all()

    @on(Button.Pressed, "#delete-chunks-btn")
    def handle_delete_chunks(self) -> None:
        if not self.selected_ids or self.current_df is None:
            self._set_status("No chunks selected.")
            return
        collection = self._current_collection()
        if not collection:
            return
        indices = self._selected_indices()
        updated_df, status = delete_entries(
            self._db_path, collection, indices,
            self.current_df, self.view_type, self.view_value,
        )
        self.selected_ids.clear()
        self._display(updated_df, status)
        self._refresh_nav()

    @on(Button.Pressed, "#export-btn")
    def handle_export_btn(self) -> None:
        self.action_export_csv()

    @on(Button.Pressed, "#add-meta-btn")
    def handle_add_metadata(self) -> None:
        if not self.selected_ids or self.current_df is None:
            self._set_status("Select chunks first.")
            return
        collection = self._current_collection()
        if not collection:
            return
        category = self.query_one("#add-category", Input).value.strip()
        value = self.query_one("#add-value", Input).value.strip()
        if not category or not value:
            self._set_status("Both category and value are required.")
            return
        indices = self._selected_indices()
        updated_df = add_metadata(
            self._db_path, indices, category, value,
            self.current_df, self.view_type, self.view_value, collection,
        )
        self._display(
            updated_df,
            f"Added '{category}={value}' to {len(indices)} chunk(s).",
            keep_selection=True,
        )
        self._refresh_nav()

    @on(Button.Pressed, "#del-meta-btn")
    def handle_delete_metadata(self) -> None:
        if not self.selected_ids or self.current_df is None:
            self._set_status("Select chunks first.")
            return
        collection = self._current_collection()
        if not collection:
            return
        category = self.query_one("#del-category", Input).value.strip()
        value = self.query_one("#del-value", Input).value.strip()
        if not category or not value:
            self._set_status("Both category and value are required.")
            return
        indices = self._selected_indices()
        updated_df = delete_metadata(
            self._db_path, indices, category, value,
            self.current_df, self.view_type, self.view_value, collection,
        )
        self._display(
            updated_df,
            f"Deleted '{category}={value}' from {len(indices)} chunk(s).",
            keep_selection=True,
        )
        self._refresh_nav()

    # ── Row selection ─────────────────────────────────────────────────────────

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        chunk_id = str(event.row_key.value)
        table = self.query_one("#chunk-table", DataTable)
        if chunk_id in self.selected_ids:
            self.selected_ids.discard(chunk_id)
            table.update_cell(event.row_key, "sel", "·", update_width=False)
        else:
            self.selected_ids.add(chunk_id)
            table.update_cell(event.row_key, "sel", "●", update_width=False)
        n = len(self.selected_ids)
        self._set_status(f"{n} chunk{'s' if n != 1 else ''} selected.")

    # ── Actions (keyboard shortcuts) ──────────────────────────────────────────

    def action_select_all(self) -> None:
        if self.current_df is None or self.current_df.empty:
            return
        table = self.query_one("#chunk-table", DataTable)
        self.selected_ids = set(self.current_df["ID"])
        for chunk_id in self.selected_ids:
            try:
                table.update_cell(chunk_id, "sel", "●", update_width=False)
            except Exception:
                pass
        self._set_status(f"{len(self.selected_ids)} chunk(s) selected.")

    def action_deselect_all(self) -> None:
        table = self.query_one("#chunk-table", DataTable)
        for chunk_id in list(self.selected_ids):
            try:
                table.update_cell(chunk_id, "sel", "·", update_width=False)
            except Exception:
                pass
        self.selected_ids.clear()
        self._set_status("Selection cleared.")

    def action_export_csv(self) -> None:
        if not self.selected_ids or self.current_df is None:
            self._set_status("Select chunks to export.")
            return
        indices = self._selected_indices()
        path = export_selected_chunks(indices, self.current_df)
        if path:
            self._set_status(f"Exported {len(indices)} chunk(s) → {path}")
        else:
            self._set_status("Export failed.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _current_collection(self) -> str:
        value = self.query_one("#collection-select", Select).value
        if value is Select.BLANK:
            self._set_status("Select a collection first.")
            return ""
        return str(value)

    def _select_value(self, selector: str) -> str:
        value = self.query_one(selector, Select).value
        return "" if value is Select.BLANK else str(value)

    def _selected_indices(self) -> list[int]:
        if self.current_df is None or self.current_df.empty:
            return []
        return [
            i
            for i, id_val in enumerate(self.current_df["ID"])
            if id_val in self.selected_ids
        ]

    def _set_status(self, message: str) -> None:
        self.query_one("#status-bar", Label).update(message)

    def _display(
        self,
        df: pd.DataFrame,
        status: str,
        keep_selection: bool = False,
    ) -> None:
        """Store df as the current view and repopulate the DataTable."""
        self.current_df = df
        if not keep_selection:
            self.selected_ids.clear()
        elif not df.empty:
            self.selected_ids &= set(df["ID"])

        table = self.query_one("#chunk-table", DataTable)
        table.clear()

        if df.empty:
            self._set_status(status)
            return

        for _, row in df.iterrows():
            chunk_id = str(row["ID"])
            doc_text = str(row["File Chunk"] or "")
            try:
                meta = (
                    json.loads(row["Metadata"])
                    if isinstance(row["Metadata"], str)
                    else {}
                )
            except Exception:
                meta = {}

            sel = "●" if chunk_id in self.selected_ids else "·"
            src = os.path.basename(meta.get("source_file", "–"))[:22]
            pairs = ", ".join(f"{k}={v}" for k, v in meta.items())
            meta_summary = (pairs[:45] + "…") if len(pairs) > 46 else pairs
            preview_text = doc_text.replace("\n", " ")
            preview = (preview_text[:49] + "…") if len(preview_text) > 50 else preview_text
            id_disp = (chunk_id[:17] + "…") if len(chunk_id) > 18 else chunk_id

            table.add_row(sel, src, meta_summary, preview, id_disp, key=chunk_id)

        self._set_status(status)

    def _refresh_nav(self) -> None:
        """Repopulate file and fileset dropdowns after a metadata mutation."""
        collection = self._select_value("#collection-select")
        if not collection:
            return
        files = get_unique_filenames(self._db_path, collection)
        filesets = get_filesets(self._db_path, collection)
        self.query_one("#file-select", Select).set_options(
            [(f, f) for f in files]
        )
        self.query_one("#fileset-select", Select).set_options(
            [(fs, fs) for fs in filesets]
        )

# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHROMA_PATH
    ChromaAuditorApp(db_path=path).run()
