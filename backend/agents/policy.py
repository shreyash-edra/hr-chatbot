"""Policy questions agent — RAG over Supabase pgvector.

Flow:
  1. Embed the user's question with text-embedding-3-small.
  2. Retrieve top-k chunks via the `match_documents` RPC.
  3. Stuff those chunks into a system prompt as the ONLY allowed context.
  4. Ask the model to answer strictly from the context; if it can't, it
     should say so and suggest checking with HR.
"""

from __future__ import annotations

import os
from datetime import datetime

from openai import OpenAI

from backend.services.supabase_client import match_documents

TOP_K = 5

SYSTEM_PROMPT = """You are the policy expert inside an HR assistant chatbot.

You will be given a USER QUESTION and a CONTEXT block containing excerpts \
from the company's policy documents. Answer the user's question using ONLY \
the information in the CONTEXT. Do not invent policies, numbers, or dates \
that are not in the context.

If the context does not contain enough information to answer, reply:
"I couldn't find that in the policy documents. Please check with HR for \
the most accurate information."

Keep the answer concise, friendly, and well-formatted (use short paragraphs \
or bullet points when listing things)."""


def _format_context(rows: list[dict]) -> str:
    """Turn retrieved rows into a readable context block."""
    parts = []
    for i, row in enumerate(rows, 1):
        content = (row.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"[Source {i}]\n{content}")
    return "\n\n".join(parts)


def answer_policy_question(question: str) -> str:
    """Answer an HR policy question using RAG. Returns the assistant text."""
    print(f"[policy] question: {question!r}")
    rows = match_documents(question, match_count=TOP_K)
    print(f"[policy] retrieved {len(rows)} chunk(s)")

    if not rows:
        return (
            "I couldn't find that in the policy documents. "
            "Please check with HR for the most accurate information."
        )

    context = _format_context(rows)
    now = datetime.now().astimezone()
    today_str = now.strftime("%A, %B %d, %Y")
    user_msg = (
        f"TODAY: {today_str}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}"
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()
    print(f"[policy] answer length: {len(answer)} chars")
    return answer


# ---------------------------------------------------------------------------
# Smoke test — `python -m backend.agents.policy`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    for q in [
        "How many vacation days do I get per year?",
        "What is the leave carry-forward policy?",
        "Are there any holidays in March?",
    ]:
        print("=" * 70)
        print(f"Q: {q}")
        print("-" * 70)
        print(answer_policy_question(q))
        print()
