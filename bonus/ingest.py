#!/usr/bin/env python3
"""
Chroma Ingest Helper — load plain-text documents into ChromaDB using OpenAI embeddings.

Dev-testing utility for populating a ChromaDB instance so you can exercise
Chroma Auditor's chunk management and metadata inspection features.

Only .txt and .md files are supported (no extra parsing dependencies).

Requires:
    pip install openai
    OPENAI_API_KEY environment variable, or pass --api-key.

Usage examples:
    python bonus/ingest.py                              # interactive prompt
    python bonus/ingest.py notes.md
    python bonus/ingest.py docs/ --db ./my-chroma --collection research
    python bonus/ingest.py notes.md --fileset "project-alpha"
    python bonus/ingest.py *.txt --chunk-size 500 --overlap 100
"""

import argparse
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from rich.console import Console
from rich.prompt import Prompt

console = Console()

SUPPORTED = {".txt", ".md"}

DISCLAIMER = (
    "[dim]Disclaimer: This is a dev-testing utility for populating a ChromaDB instance "
    "to exercise Chroma Auditor's chunk management and metadata inspection features. "
    "It is not intended for production ingestion workflows.[/dim]"
)

NOTE_FILES = (
    "[yellow]Note:[/yellow] Only [bold].txt[/bold] and [bold].md[/bold] files are supported. "
    "No additional parsing libraries are required."
)

NOTE_CHUNKING = (
    "[yellow]Note:[/yellow] Chunking uses a simple character-based splitter "
    "(default 1000 chars, 200-char overlap) that breaks preferentially at "
    "paragraph boundaries, then line breaks, then sentence ends, then word boundaries."
)

# ─── Document reading ─────────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

# ─── Text splitting ───────────────────────────────────────────────────────────

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

# ─── Embedding setup ──────────────────────────────────────────────────────────

def get_embedding_function(api_key: str, model: str):
    try:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    except ImportError:
        console.print("[red]Could not import OpenAIEmbeddingFunction from chromadb.[/red]")
        console.print("Ensure chromadb is installed:  pip install chromadb")
        sys.exit(1)
    try:
        import openai  # noqa: F401
    except ImportError:
        console.print("[red]openai package not found.[/red]")
        console.print("Install it with:  pip install openai")
        sys.exit(1)
    return OpenAIEmbeddingFunction(api_key=api_key, model_name=model)

# ─── Ingestion ────────────────────────────────────────────────────────────────

def ingest_file(
    path: Path,
    collection: chromadb.Collection,
    chunk_size: int,
    overlap: int,
    fileset: str | None,
) -> int:
    """Chunk and embed one file; return the number of chunks stored."""
    text = read_file(path)
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return 0

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

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return total

# ─── File discovery ───────────────────────────────────────────────────────────

def resolve_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for ext in sorted(SUPPORTED):
                paths.extend(sorted(p.rglob(f"*{ext}")))
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED:
                paths.append(p)
            else:
                console.print(f"[yellow]Skipping unsupported file type: {p.name}[/yellow]")
                console.print(f"  Only .txt and .md files are accepted.")
        else:
            console.print(f"[yellow]Not found, skipping: {raw}[/yellow]")
    return paths

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest .txt/.md documents into ChromaDB using OpenAI embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs", nargs="*", metavar="FILE_OR_DIR",
        help="Files or directories to ingest (.txt and .md only). "
             "Omit to be prompted interactively.",
    )
    parser.add_argument(
        "--db", default="./chroma", metavar="PATH",
        help="ChromaDB storage directory (default: ./chroma).",
    )
    parser.add_argument(
        "--collection", default="documents", metavar="NAME",
        help="Collection to ingest into (default: documents).",
    )
    parser.add_argument(
        "--fileset", default=None, metavar="NAME",
        help="Optional fileset tag applied to every chunk.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, metavar="N",
        help="Target chunk size in characters (default: 1000).",
    )
    parser.add_argument(
        "--overlap", type=int, default=200, metavar="N",
        help="Overlap between consecutive chunks in characters (default: 200).",
    )
    parser.add_argument(
        "--model", default="text-embedding-3-small", metavar="MODEL",
        help="OpenAI embedding model (default: text-embedding-3-small).",
    )
    parser.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print("[bold]Chroma Ingest Helper[/bold]")
    console.print("─" * 58)
    console.print(NOTE_FILES)
    console.print(NOTE_CHUNKING)
    console.print(DISCLAIMER)
    console.print("─" * 58)
    console.print()

    # ── Interactive file prompt if no inputs given ─────────────────────────────
    if not args.inputs:
        raw = Prompt.ask(
            "[cyan]Enter path to a .txt or .md file (or a directory)[/cyan]"
        ).strip()
        if not raw:
            console.print("[red]No path entered. Exiting.[/red]")
            sys.exit(1)
        args.inputs = [raw]

    # ── API key ────────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]No OpenAI API key found.[/red]")
        console.print(
            "Set the [bold]OPENAI_API_KEY[/bold] environment variable "
            "or pass [bold]--api-key[/bold]."
        )
        sys.exit(1)

    # ── Config summary ─────────────────────────────────────────────────────────
    console.print(f"  Database:   [cyan]{os.path.abspath(args.db)}[/cyan]")
    console.print(f"  Collection: [cyan]{args.collection}[/cyan]")
    console.print(f"  Model:      [cyan]{args.model}[/cyan]")
    console.print(f"  Chunk size: [cyan]{args.chunk_size}[/cyan]  "
                  f"Overlap: [cyan]{args.overlap}[/cyan]")
    if args.fileset:
        console.print(f"  Fileset:    [cyan]{args.fileset}[/cyan]")
    console.print()

    # ── Resolve files ──────────────────────────────────────────────────────────
    files = resolve_inputs(args.inputs)
    if not files:
        console.print("[red]No supported files found. Only .txt and .md are accepted.[/red]")
        sys.exit(1)
    console.print(f"Found [bold]{len(files)}[/bold] file(s) to process.\n")

    # ── Connect to ChromaDB ────────────────────────────────────────────────────
    ef = get_embedding_function(api_key, args.model)
    try:
        client = chromadb.PersistentClient(path=args.db)
        collection = client.get_or_create_collection(
            name=args.collection, embedding_function=ef
        )
    except Exception as e:
        console.print(f"[red]Failed to connect to ChromaDB: {e}[/red]")
        sys.exit(1)

    # ── Ingest ─────────────────────────────────────────────────────────────────
    total_chunks = 0
    failed = 0
    for path in files:
        try:
            n = ingest_file(path, collection, args.chunk_size, args.overlap, args.fileset)
            total_chunks += n
            console.print(f"  [green]✓[/green] {path.name:<42} {n} chunk(s)")
        except Exception as e:
            failed += 1
            console.print(f"  [red]✗[/red] {path.name:<42} {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    console.print()
    console.print("─" * 58)
    success = len(files) - failed
    summary = f"  [bold]Done.[/bold]  {total_chunks} chunk(s) ingested from {success} file(s)."
    if failed:
        summary += f"  [red]{failed} failed.[/red]"
    console.print(summary)
    console.print(
        f"  Collection '[cyan]{args.collection}[/cyan]' "
        f"now has [bold]{collection.count()}[/bold] chunk(s) total."
    )
    console.print()


if __name__ == "__main__":
    main()
