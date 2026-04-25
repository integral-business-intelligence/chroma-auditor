#!/usr/bin/env python3
"""
Chroma Ingest Helper — TUI for loading .txt/.md files into ChromaDB via OpenAI embeddings.

Dev-testing utility for populating a ChromaDB instance so you can exercise
Chroma Auditor's chunk management and metadata inspection features.

Requires:
    pip install openai

Usage:
    python bonus/ingest.py
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

# ── Constants ─────────────────────────────────────────────────────────────────

# Change OPENAI_API_BASE to point at a proxy or compatible API.
OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL   = "text-embedding-3-small"
SUPPORTED       = {".txt", ".md"}

# ── Pure functions ────────────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                pos = text.rfind(sep, start + 1, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end - overlap)
    return chunks


def embed_texts(texts: list[str], api_key: str, model: str) -> list[list[float]]:
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package not found — install with:  pip install openai")
    client = openai.OpenAI(api_key=api_key, base_url=OPENAI_API_BASE)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def ingest_file(
    path: Path,
    collection: chromadb.Collection,
    chunk_size: int,
    overlap: int,
    fileset: str | None,
    api_key: str,
    model: str,
) -> int:
    text = read_file(path)
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks, api_key, model)
    timestamp = datetime.now().isoformat()
    total = len(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = []
    for i in range(total):
        meta: dict = {
            "source_file": path.name,
            "chunk_index": i + 1,
            "total_chunks": total,
            "upload_timestamp": timestamp,
        }
        if fileset:
            meta["fileset"] = fileset
        metadatas.append(meta)
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    return total

# ── TUI ───────────────────────────────────────────────────────────────────────

APP_CSS = """
Screen {
    background: $surface;
}

#notices {
    height: auto;
    padding: 1 2;
    background: $surface-darken-1;
    border-bottom: hline $primary-darken-2;
}

#notices Static {
    margin-bottom: 0;
}

#note-files, #note-chunking {
    color: $warning-darken-1;
}

#disclaimer {
    color: $text-disabled;
}

#form {
    height: auto;
    padding: 1 2;
    border-bottom: hline $primary-darken-2;
}

.field-row {
    height: auto;
    margin-bottom: 1;
    align: left middle;
}

.field-label {
    width: 20;
    padding-right: 1;
    text-align: right;
    color: $text-muted;
}

#chunk-size, #overlap {
    width: 10;
}

#overlap-label {
    width: auto;
    padding: 0 1;
    color: $text-muted;
}

#ingest-btn {
    margin-left: 21;
    margin-top: 1;
}

#log {
    height: 1fr;
    padding: 0 1;
    border-top: hline $primary-darken-2;
}
"""


class IngestApp(App):
    CSS = APP_CSS
    TITLE = "Chroma Ingest Helper"
    SUB_TITLE = f"{OPENAI_API_BASE}/embeddings  ·  {DEFAULT_MODEL}"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="notices"):
            yield Static(
                "[yellow]Note:[/yellow]  Only [bold].txt[/bold] and [bold].md[/bold] "
                "files are accepted — no extra parsing dependencies.",
                id="note-files",
            )
            yield Static(
                "[yellow]Note:[/yellow]  Character-based chunker: "
                "breaks at \\n\\n → \\n → '. ' → ' ' boundaries.",
                id="note-chunking",
            )
            yield Static(
                "[dim]Dev-testing utility for Chroma Auditor. "
                "Not intended for production ingestion.[/dim]",
                id="disclaimer",
            )
        with Vertical(id="form"):
            with Horizontal(classes="field-row"):
                yield Label("API Key", classes="field-label")
                yield Input(placeholder="sk-…  paste or type", password=True, id="api-key")
            with Horizontal(classes="field-row"):
                yield Label("File / Directory", classes="field-label")
                yield Input(placeholder="/path/to/file.txt  or  ./docs/", id="file-path")
            with Horizontal(classes="field-row"):
                yield Label("DB Path", classes="field-label")
                yield Input(value="./chroma", id="db-path")
            with Horizontal(classes="field-row"):
                yield Label("Collection", classes="field-label")
                yield Input(value="documents", id="collection")
            with Horizontal(classes="field-row"):
                yield Label("Fileset (optional)", classes="field-label")
                yield Input(placeholder="project-alpha", id="fileset")
            with Horizontal(classes="field-row"):
                yield Label("Chunk / Overlap", classes="field-label")
                yield Input(value="1000", id="chunk-size")
                yield Label(" / ", id="overlap-label")
                yield Input(value="200", id="overlap")
            with Horizontal(classes="field-row"):
                yield Label("Model", classes="field-label")
                yield Input(value=DEFAULT_MODEL, id="model")
            yield Button("Ingest", variant="primary", id="ingest-btn")
        yield RichLog(id="log", markup=True, highlight=True, wrap=True)
        yield Footer()

    @on(Button.Pressed, "#ingest-btn")
    def handle_ingest(self) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()

        api_key    = self.query_one("#api-key",   Input).value.strip()
        file_raw   = self.query_one("#file-path", Input).value.strip()
        db_path    = self.query_one("#db-path",   Input).value.strip() or "./chroma"
        col_name   = self.query_one("#collection",Input).value.strip() or "documents"
        fileset    = self.query_one("#fileset",   Input).value.strip() or None
        model      = self.query_one("#model",     Input).value.strip() or DEFAULT_MODEL

        try:
            chunk_size = int(self.query_one("#chunk-size", Input).value.strip() or "1000")
            overlap    = int(self.query_one("#overlap",    Input).value.strip() or "200")
        except ValueError:
            log.write("[red]Chunk size and overlap must be integers.[/red]")
            return

        if not api_key:
            log.write("[red]API key is required.[/red]")
            self.query_one("#api-key", Input).focus()
            return
        if not file_raw:
            log.write("[red]File or directory path is required.[/red]")
            self.query_one("#file-path", Input).focus()
            return

        btn = self.query_one("#ingest-btn", Button)
        btn.disabled = True
        btn.label = "Ingesting…"

        self.run_worker(
            lambda: self._worker(
                log, btn,
                api_key, file_raw, db_path, col_name,
                fileset, chunk_size, overlap, model,
            ),
            thread=True,
            exclusive=True,
        )

    def _worker(
        self, log, btn,
        api_key, file_raw, db_path, col_name,
        fileset, chunk_size, overlap, model,
    ) -> None:
        def emit(msg: str) -> None:
            self.call_from_thread(log.write, msg)

        def reset(label: str = "Ingest") -> None:
            def _do():
                btn.disabled = False
                btn.label = label
            self.call_from_thread(_do)

        # ── Resolve files ──────────────────────────────────────────────────
        p = Path(file_raw)
        files: list[Path] = []
        if p.is_dir():
            for ext in sorted(SUPPORTED):
                files.extend(sorted(p.rglob(f"*{ext}")))
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED:
                files.append(p)
            else:
                emit(
                    f"[red]Unsupported type '[bold]{p.suffix}[/bold]'. "
                    f"Only .txt and .md are accepted.[/red]"
                )
                reset()
                return
        else:
            emit(f"[red]Path not found:[/red] {file_raw}")
            reset()
            return

        if not files:
            emit("[red]No .txt or .md files found at that path.[/red]")
            reset()
            return

        emit(f"Found [bold]{len(files)}[/bold] file(s).\n")
        emit(f"  Endpoint:   [cyan]{OPENAI_API_BASE}/embeddings[/cyan]")
        emit(f"  Model:      [cyan]{model}[/cyan]")
        emit(f"  Database:   [cyan]{os.path.abspath(db_path)}[/cyan]")
        emit(f"  Collection: [cyan]{col_name}[/cyan]")
        emit(f"  Chunk size: [cyan]{chunk_size}[/cyan]  Overlap: [cyan]{overlap}[/cyan]")
        if fileset:
            emit(f"  Fileset:    [cyan]{fileset}[/cyan]")
        emit("")

        # ── Connect to ChromaDB ────────────────────────────────────────────
        try:
            chroma_client = chromadb.PersistentClient(path=db_path)
            collection = chroma_client.get_or_create_collection(name=col_name)
        except Exception as exc:
            emit(f"[red]ChromaDB connection failed:[/red] {exc}")
            reset()
            return

        # ── Ingest ─────────────────────────────────────────────────────────
        total_chunks = 0
        failed = 0
        for path in files:
            try:
                n = ingest_file(
                    path, collection,
                    chunk_size, overlap, fileset,
                    api_key, model,
                )
                total_chunks += n
                emit(f"  [green]✓[/green] {path.name:<40} {n} chunk(s)")
            except Exception as exc:
                failed += 1
                emit(f"  [red]✗[/red] {path.name:<40} {exc}")

        # ── Summary ────────────────────────────────────────────────────────
        emit("")
        emit("─" * 52)
        success = len(files) - failed
        summary = f"  [bold]Done.[/bold]  {total_chunks} chunk(s) from {success} file(s)."
        if failed:
            summary += f"  [red]{failed} failed.[/red]"
        emit(summary)
        emit(
            f"  Collection '[cyan]{col_name}[/cyan]' "
            f"now has [bold]{collection.count()}[/bold] chunk(s) total."
        )
        reset("Ingest Again")


if __name__ == "__main__":
    IngestApp().run()
