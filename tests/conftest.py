import sys
import os
from unittest.mock import MagicMock
import pytest
import chromadb

# Mock UI frameworks before the auditor module is imported so that the
# Gradio/FastAPI top-level code in the script runs without a live server.
for _name in [
    "gradio", "gradio.themes",
    "fastapi", "fastapi.middleware", "fastapi.middleware.cors",
]:
    sys.modules.setdefault(_name, MagicMock())

import importlib.util

_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "chroma-auditor.py")
)
_spec = importlib.util.spec_from_file_location("chroma_auditor", _SCRIPT)
auditor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(auditor)

# ── Shared constants ─────────────────────────────────────────────────────────

COLLECTION = "test_collection"

# Deterministic sample data.  Embeddings are dummy values — the auditor only
# uses collection.get() / collection.update() / collection.delete(), never
# collection.query(), so embedding values are irrelevant.
SAMPLE_DOCS = [
    {
        "id": "a1",
        "text": "First chunk of the report.",
        "meta": {"source_file": "report.txt", "chunk_index": 1, "fileset": "project_x"},
    },
    {
        "id": "a2",
        "text": "Second chunk of the report.",
        "meta": {"source_file": "report.txt", "chunk_index": 2, "fileset": "project_x"},
    },
    {
        "id": "a3",
        "text": "Third chunk of the report.",
        "meta": {"source_file": "report.txt", "chunk_index": 3, "fileset": "project_x"},
    },
    {
        "id": "b1",
        "text": "First chunk of notes.",
        "meta": {"source_file": "notes.txt", "chunk_index": 1, "fileset": "project_x|project_y"},
    },
    {
        "id": "b2",
        "text": "Second chunk of notes.",
        "meta": {"source_file": "notes.txt", "chunk_index": 2, "fileset": "project_y"},
    },
    {
        "id": "c1",
        "text": "Standalone chunk with no fileset.",
        "meta": {"source_file": "data.txt", "chunk_index": 1},
    },
]

_DUMMY_EMBEDDINGS = [[float(i) * 0.1, 0.0, 0.0] for i in range(len(SAMPLE_DOCS))]


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path):
    """Temporary ChromaDB populated with deterministic sample data."""
    path = str(tmp_path / "chroma")
    client = chromadb.PersistentClient(path=path)
    col = client.create_collection(COLLECTION)
    col.add(
        ids=[d["id"] for d in SAMPLE_DOCS],
        documents=[d["text"] for d in SAMPLE_DOCS],
        metadatas=[d["meta"] for d in SAMPLE_DOCS],
        embeddings=_DUMMY_EMBEDDINGS,
    )
    return path


@pytest.fixture()
def loaded_df(db_path):
    """Full-collection DataFrame — basis for write-operation tests."""
    return auditor.load_collection(COLLECTION, db_path)


def row_pos(df, chunk_id):
    """Return the positional index (0-based) of a chunk in a DataFrame.

    Uses list iteration rather than label lookup because the auditor passes
    selected_indices as positional (matching Gradio's row-click events) and
    looks them up with iloc.
    """
    return list(df["ID"]).index(chunk_id)
