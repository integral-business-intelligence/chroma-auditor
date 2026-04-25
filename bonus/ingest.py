#!/usr/bin/env python3
"""
Chroma Ingest Helper — load documents into ChromaDB using OpenAI embeddings.

Reads .txt, .md, .pdf, and .docx files, splits them into overlapping chunks,
embeds them with OpenAI, and stores them in a ChromaDB collection with
metadata compatible with Chroma Auditor (source_file, chunk_index,
total_chunks, upload_timestamp, and an optional fileset tag).

Requires:
    pip install openai pypdf python-docx

    OPENAI_API_KEY environment variable, or pass --api-key.

Usage examples:
    python bonus/ingest.py report.pdf
    python bonus/ingest.py docs/ --db ./my-chroma --collection research
    python bonus/ingest.py notes.pdf --fileset "project-alpha"
    python bonus/ingest.py *.txt --chunk-size 500 --overlap 100
    python bonus/ingest.py report.pdf --model text-embedding-3-large
"""

import argparse
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from rich.console import Console

console = Console()

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}

# ─── Document reading ─────────────────────────────────────────────────────────

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError:
        raise RuntimeError("pypdf is required for PDF files:  pip install pypdf")
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path: Path) -> str:
    try:
        import docx
    except ImportError:
        raise RuntimeError(
            "python-docx is required for DOCX files:  pip install python-docx"
        )
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return read_txt(path)
    elif ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")

# ─── Text splitting ───────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, breaking at natural boundaries."""
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
    """Return a ChromaDB OpenAI embedding function."""
    try:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    except ImportError:
        console.print("[red]Could not import OpenAIEmbeddingFunction from chromadb.[/red]")
        console.print("Ensure chromadb is installed:  pip install chromadb")
        sys.exit(1)
    try:
        import openai  # noqa: F401 — confirm the package is present
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
    """Expand files and directories into a sorted list of supported paths."""
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
        else:
            console.print(f"[yellow]Not found, skipping: {raw}[/yellow]")
    return paths

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into ChromaDB using OpenAI embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="FILE_OR_DIR",
        help="Files or directories to ingest (.txt .md .pdf .docx).",
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

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]No OpenAI API key found.[/red]")
        console.print(
            "Set the [bold]OPENAI_API_KEY[/bold] environment variable "
            "or pass [bold]--api-key[/bold]."
        )
        sys.exit(1)

    # Print config summary
    console.print()
    console.print("[bold]Chroma Ingest Helper[/bold]")
    console.print("─" * 42)
    console.print(f"  Database:   [cyan]{os.path.abspath(args.db)}[/cyan]")
    console.print(f"  Collection: [cyan]{args.collection}[/cyan]")
    console.print(f"  Model:      [cyan]{args.model}[/cyan]")
    console.print(f"  Chunk size: [cyan]{args.chunk_size}[/cyan]  "
                  f"Overlap: [cyan]{args.overlap}[/cyan]")
    if args.fileset:
        console.print(f"  Fileset:    [cyan]{args.fileset}[/cyan]")
    console.print()

    # Resolve input files
    files = resolve_inputs(args.inputs)
    if not files:
        console.print("[red]No supported files found.[/red]")
        sys.exit(1)
    console.print(f"Found [bold]{len(files)}[/bold] file(s) to process.\n")

    # Set up embedding function and connect to ChromaDB
    ef = get_embedding_function(api_key, args.model)
    try:
        client = chromadb.PersistentClient(path=args.db)
        collection = client.get_or_create_collection(
            name=args.collection, embedding_function=ef
        )
    except Exception as e:
        console.print(f"[red]Failed to connect to ChromaDB: {e}[/red]")
        sys.exit(1)

    # Ingest
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

    # Summary
    console.print()
    console.print("─" * 42)
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
