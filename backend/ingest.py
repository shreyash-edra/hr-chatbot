"""Standalone ingestion script for the `documents` table in Supabase.

Usage
-----
    python -m backend.ingest path/to/policy.pdf
    python -m backend.ingest path/to/notes.txt

What it does
------------
  1. Reads a .pdf (via pypdf, falling back to pdfplumber) or .txt file.
  2. Splits the text into 1000-char chunks with 100-char overlap.
  3. Embeds each chunk with text-embedding-3-small (matches retrieval).
  4. DELETES all existing rows in the documents table before inserting
     so stale chunks don't mix with the new document.
  5. Inserts each chunk as {content, metadata: {source: filename}, embedding}.

Reuses .env values: SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY,
SUPABASE_TABLE (defaults to "documents").

⚠️  This script is DESTRUCTIVE — it wipes the existing rows before
    inserting the new ones. Re-run ingestion any time you switch
    source documents.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.services.supabase_client import _get_supabase, embed_query

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(path: Path) -> str:
    """Extract plain text from a .pdf or .txt file."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise ValueError(
        f"Unsupported file type: {suffix!r}. Supported: .pdf, .txt"
    )


def _extract_pdf(path: Path) -> str:
    """Try pypdf first; if it returns no text, fall back to pdfplumber.

    pypdf is fast and handles most well-formed PDFs. pdfplumber is slower
    but does a better job on PDFs with complex layouts or unusual encodings.
    """
    # --- attempt pypdf ---
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        parts: list[str] = []
        for i, page in enumerate(reader.pages, 1):
            try:
                parts.append(page.extract_text() or "")
            except Exception as e:
                print(f"[ingest] pypdf failed on page {i}: {e}")
        text = "\n".join(parts).strip()
        if text:
            return text
        print("[ingest] pypdf returned no text — falling back to pdfplumber")
    except ImportError:
        print("[ingest] pypdf not installed — using pdfplumber")

    # --- fallback: pdfplumber ---
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Sliding-window character chunks. Each chunk is up to `size` chars long;
    consecutive chunks share `overlap` characters."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be > 0 and overlap must be in [0, size)")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    step = size - overlap
    n = len(text)

    for i in range(0, n, step):
        chunk = text[i : i + size]
        if chunk.strip():
            chunks.append(chunk)
        if i + size >= n:
            break

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a document into the Supabase `documents` table.",
    )
    parser.add_argument("file", help="Path to a .pdf or .txt file")
    args = parser.parse_args()

    load_dotenv()

    path = Path(args.file).expanduser().resolve()
    if not path.is_file():
        print(f"[ingest] File not found: {path}")
        return 1
    if path.suffix.lower() not in {".pdf", ".txt"}:
        print(f"[ingest] Unsupported file type: {path.suffix!r} "
              "(only .pdf and .txt are supported)")
        return 1

    print(f"[ingest] Reading {path.name}...")
    text = extract_text(path)
    if not text.strip():
        print("[ingest] No text content extracted. Nothing to ingest.")
        return 1
    print(f"[ingest] Extracted {len(text):,} characters")

    chunks = chunk_text(text)
    print(
        f"[ingest] Split into {len(chunks)} chunk(s) "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    table = os.environ.get("SUPABASE_TABLE", "documents")
    sb = _get_supabase()

    # ---- 1. Wipe existing rows ---------------------------------------------
    print(f"[ingest] Deleting existing rows from {table!r}...")
    sb.table(table).delete().neq("id", 0).execute()
    print("[ingest] Cleared.")

    # ---- 2. Embed + insert each chunk --------------------------------------
    filename = path.name
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        print(f"[ingest] Embedding chunk {i}/{total}...", flush=True)
        embedding = embed_query(chunk)
        row = {
            "content": chunk,
            "metadata": {"source": filename},
            "embedding": embedding,
        }
        sb.table(table).insert(row).execute()

    print(f"[ingest] Done. Inserted {total} chunk(s) for {filename!r}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
