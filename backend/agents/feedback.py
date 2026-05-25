"""Feedback collection agent — anonymous Sheets append.

Feedback Tracker columns (Sheet1):
  Feedback | Sentiment | Action Items

Rules:
  - NEVER record or ask for a name or employee ID.
  - The actual feedback content (an opinion / complaint / suggestion /
    compliment / observation) is REQUIRED. Intent-only messages like
    "I have feedback" do NOT get logged — the agent asks the user to
    share the actual content first.
  - Classify sentiment as Positive / Neutral / Negative.
  - Generate one short, actionable next step (Action Item).
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from backend.services.sheets_client import append_row

FEEDBACK_HEADERS = ["Feedback", "Sentiment", "Action Items"]


# ---------------------------------------------------------------------------
# Intent-vs-content gate
# ---------------------------------------------------------------------------

INTENT_SYSTEM = """Decide whether the text below contains ACTUAL feedback \
content — an opinion, complaint, compliment, suggestion, or observation \
about something in the workplace — or whether it only signals INTENT to \
share feedback without saying what it is.

Return ONLY JSON: {"has_content": true|false}

Examples:
- "I have feedback to share"                       → {"has_content": false}
- "I want to share some feedback"                  → {"has_content": false}
- "I have a suggestion"                            → {"has_content": false}
- "feedback"                                       → {"has_content": false}
- "anonymous feedback"                             → {"has_content": false}
- "the office coffee is terrible"                  → {"has_content": true}
- "I love the new flexible WFH policy"             → {"has_content": true}
- "meetings are too long, can we shorten them?"    → {"has_content": true}
- "the AC is freezing"                             → {"has_content": true}
- "yes"                                            → {"has_content": false}"""


def _has_content(text: str) -> bool:
    """True if the message contains actual feedback content, not just intent."""
    if len(text.strip()) < 6:
        return False
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        messages=[
            {"role": "system", "content": INTENT_SYSTEM},
            {"role": "user",   "content": text},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return bool(json.loads(raw).get("has_content"))
    except json.JSONDecodeError:
        # Fail closed — if we can't tell, don't log a row we might regret.
        return False


# ---------------------------------------------------------------------------
# Sentiment + action-item classifier
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """You process anonymous employee feedback for an HR \
system.

Given the raw feedback text, return ONLY a JSON object with:
  sentiment    (string — exactly one of "Positive", "Neutral", "Negative")
  action_item  (string — ONE short, concrete next step HR or management \
could take, max ~15 words)

Do not include any other fields. Do not wrap in markdown."""


def _classify(feedback_text: str) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": feedback_text},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    sentiment = (data.get("sentiment") or "").strip().title()
    if sentiment not in {"Positive", "Neutral", "Negative"}:
        sentiment = "Neutral"
    action_item = (data.get("action_item") or "Review with team.").strip()
    return {"sentiment": sentiment, "action_item": action_item}


# ---------------------------------------------------------------------------
# Public action
# ---------------------------------------------------------------------------

def submit_feedback(feedback_text: str) -> str:
    """Classify and append the row — but ONLY if the text contains actual
    feedback content. Intent-only messages get a polite prompt back."""
    print(f"[feedback] received: {feedback_text!r}")
    text = feedback_text.strip()

    if not text:
        return "I'd love to capture your feedback — what would you like to share?"

    if not _has_content(text):
        print("[feedback] intent-only — refusing to log")
        return (
            "Sure — what would you like to share? It can be an opinion, "
            "complaint, suggestion, or compliment about anything in the "
            "workplace (food, equipment, processes, culture, etc.). "
            "Whatever you send will be recorded **anonymously**."
        )

    result = _classify(text)
    print(f"[feedback] classified: {result}")

    row = [text, result["sentiment"], result["action_item"]]
    assert len(row) == len(FEEDBACK_HEADERS), "row length must match header count"

    sheet_id = os.environ["FEEDBACK_SHEET_ID"]
    append_row(sheet_id, row)
    print("[feedback] appended row (anonymous)")

    return (
        "Thanks — your feedback has been recorded **anonymously**.\n\n"
        f"- **Sentiment:** {result['sentiment']}\n"
        f"- **Suggested action:** {result['action_item']}"
    )


# ---------------------------------------------------------------------------
# Smoke test — `python -m backend.agents.feedback`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    samples = [
        # Intent-only — should be refused
        "I have feedback to share",
        "I want to give some feedback",
        # Actual content — should be logged
        "The office coffee is terrible",
        "I really love the new flexible WFH policy, great for productivity.",
        "The new project management tool is okay, but the UI could be more intuitive.",
    ]
    for s in samples:
        print("=" * 70)
        print(f"Feedback input: {s}")
        print("-" * 70)
        print(submit_feedback(s))
        print()
