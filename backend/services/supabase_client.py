"""Supabase pgvector retrieval for the policy RAG agent.

The user's existing Supabase project already has:
  - A `documents` table with columns: content, metadata, embedding (1536-dim)
  - A `match_documents` RPC function that takes
      (query_embedding, match_count, filter) and returns rows ordered
      by cosine similarity.

This module only does the QUERY side. No ingestion.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI
from supabase import create_client, Client

EMBED_MODEL = "text-embedding-3-small"  # 1536 dims — matches the loaded data


def _get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def _get_openai() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_query(text: str) -> list[float]:
    """Embed a single query string with text-embedding-3-small."""
    client = _get_openai()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def match_documents(
    query: str,
    match_count: int = 5,
    filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top `match_count` most relevant document chunks for `query`.

    Returns a list of rows from the `match_documents` RPC. Each row typically
    has at least `content` and `metadata` (and often a `similarity` score),
    but we don't depend on the exact extra fields.
    """
    rpc_name = os.environ.get("SUPABASE_QUERY", "match_documents")
    sb = _get_supabase()
    embedding = embed_query(query)

    params = {
        "query_embedding": embedding,
        "match_count": match_count,
        "filter": filter or {},
    }
    resp = sb.rpc(rpc_name, params).execute()
    return resp.data or []


# ---------------------------------------------------------------------------
# Smoke test — `python -m backend.services.supabase_client`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("[supabase] Embedding a test query...")
    q = "What is the company policy on vacation?"
    rows = match_documents(q, match_count=3)
    print(f"[supabase] Got {len(rows)} matches for: {q!r}")
    for i, row in enumerate(rows, 1):
        content = (row.get("content") or "").strip().replace("\n", " ")
        preview = content[:140] + ("..." if len(content) > 140 else "")
        sim = row.get("similarity")
        sim_str = f" (sim={sim:.3f})" if isinstance(sim, (int, float)) else ""
        print(f"  {i}.{sim_str} {preview}")
    if not rows:
        print("[supabase] WARNING: no rows returned. Check that:")
        print("  - the `documents` table has rows with non-null `embedding`")
        print("  - the `match_documents` RPC exists with the expected signature")
        print("  - SUPABASE_SERVICE_KEY is the service-role key (not anon)")
