"""Leave management agent — reads from and writes to the Leave Tracker sheet.

Leave Tracker column order (Sheet1):
  Employee ID | Employee Name | Leave Type | Start Date | End Date |
  Number of Days | Leave Status | Requested On | Approval Date |
  Comments/Reason

A new row is appended ONLY when ALL of these are supplied by the user via
the chat: Employee ID, Employee Name, Leave Type, Start Date, End Date,
Reason. If any are missing, the agent asks for the missing ones instead
of inventing defaults. The auto-filled columns are:
  - Leave Status   = "Pending"
  - Requested On   = today's date (YYYY-MM-DD)
  - Approval Date  = ""  (blank — filled in when HR approves)
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime

from openai import OpenAI

from backend.services.sheets_client import append_row, read_all_records

LEAVE_HEADERS = [
    "Employee ID",
    "Employee Name",
    "Leave Type",
    "Start Date",
    "End Date",
    "Number of Days",
    "Leave Status",
    "Requested On",
    "Approval Date",
    "Comments/Reason",
]

# Fields the USER must supply (in the order we ask for them in the chat).
# Each tuple is (json_key, human-friendly label used in the "still missing"
# message back to the user).
REQUIRED_USER_FIELDS: list[tuple[str, str]] = [
    ("employee_id",   "your **Employee ID**"),
    ("employee_name", "your **full name**"),
    ("leave_type",    "the **leave type** (e.g. Vacation, Sick, Personal, Casual)"),
    ("start_date",    "the **start date**"),
    ("end_date",      "the **end date**"),
    ("reason",        "a brief **reason** for the leave"),
]


# ---------------------------------------------------------------------------
# Extraction — combine the whole conversation, not just the latest message
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """You extract structured leave-request fields from a \
conversation between an employee and an HR chatbot.

Return ONLY a JSON object with these keys:

  employee_id      (string — leave "" if the user has NOT explicitly stated it)
  employee_name    (string — leave "" if not stated)
  leave_type       (string — leave "" if not stated; common values: \
"Vacation", "Sick", "Personal", "Casual")
  start_date       (string YYYY-MM-DD; "" if not derivable)
  end_date         (string YYYY-MM-DD; "" if not derivable)
  number_of_days   (integer; 0 if not computable)
  reason           (string; "" if not stated)

Rules:
  - Today's date is {today}. Resolve relative dates ("next week", "Monday", \
"June 10-12") against this.
  - Combine fields ACROSS turns: if the user gave a date earlier and a \
name later, both belong in the output.
  - DO NOT invent or default missing fields. If the user has not explicitly \
provided a field, return "" (empty string) for it. NEVER use placeholder \
values like "N/A" or "Vacation" unless the user actually said them.
  - If only one date is given, treat it as both start_date and end_date \
(number_of_days = 1).
  - Output JSON only. No prose, no markdown."""


def _today_iso() -> str:
    return date.today().isoformat()


def _history_to_text(history: list[dict] | None) -> str:
    """Render the recent conversation as readable text for the extractor."""
    if not history:
        return ""
    recent = [m for m in history if m.get("content") and m.get("role") in ("user", "assistant")]
    recent = recent[-12:]  # cap at last 12 turns
    lines = []
    for m in recent:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _extract_request(user_text: str, history: list[dict] | None = None) -> dict:
    """Ask the LLM to pull leave fields out of the conversation."""
    history_text = _history_to_text(history)
    if history_text:
        input_block = (
            "Conversation so far:\n"
            f"{history_text}\n\n"
            "Latest user message (already included above): "
            f"{user_text!r}\n\n"
            "Extract leave-request fields by combining ALL turns above."
        )
    else:
        input_block = user_text

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM.format(today=_today_iso())},
            {"role": "user",   "content": input_block},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    # Normalise — make sure every required key exists, as a string.
    for key, _label in REQUIRED_USER_FIELDS:
        v = data.get(key)
        data[key] = (v.strip() if isinstance(v, str) else "") if v else ""
    if "number_of_days" not in data or not isinstance(data["number_of_days"], int):
        data["number_of_days"] = 0

    return data


def _missing_fields(data: dict) -> list[str]:
    """Return the human-friendly labels of any required fields still blank."""
    return [label for key, label in REQUIRED_USER_FIELDS if not data.get(key)]


def _compute_days(start: str, end: str) -> int:
    """Inclusive day count between two YYYY-MM-DD dates (best effort)."""
    try:
        s = datetime.strptime(start, "%Y-%m-%d").date()
        e = datetime.strptime(end, "%Y-%m-%d").date()
        return (e - s).days + 1
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Public actions
# ---------------------------------------------------------------------------

def submit_leave_request(
    user_text: str,
    history: list[dict] | None = None,
) -> str:
    """Parse a leave request from the conversation. If ALL required fields
    are present, append a Pending row to the sheet. Otherwise, return a
    chat-friendly prompt asking for the missing ones — DO NOT write a row.
    """
    print(f"[leave] submit request: {user_text!r}")
    data = _extract_request(user_text, history)
    print(f"[leave] extracted: {data}")

    missing = _missing_fields(data)
    if missing:
        print(f"[leave] incomplete — missing: {missing}")
        if len(missing) == 1:
            return (
                f"Before I can log this leave request, I still need {missing[0]}. "
                "Could you share that?"
            )
        bullets = "\n".join(f"- {m}" for m in missing)
        return (
            "Before I can log this leave request, I still need a few details:\n\n"
            f"{bullets}\n\n"
            "Once you share these I'll log it right away."
        )

    # All required fields present — compute days and append the row.
    start = data["start_date"]
    end   = data["end_date"]
    days  = data["number_of_days"] or _compute_days(start, end) or 1

    row = [
        data["employee_id"],
        data["employee_name"],
        data["leave_type"],
        start,
        end,
        days,
        "Pending",
        _today_iso(),  # Requested On
        "",            # Approval Date (blank)
        data["reason"],
    ]
    assert len(row) == len(LEAVE_HEADERS), "row length must match header count"

    sheet_id = os.environ["LEAVE_SHEET_ID"]
    append_row(sheet_id, row)
    print(f"[leave] appended row: {row}")

    return (
        "Got it — I've logged your leave request as **Pending**:\n\n"
        f"- **Employee ID:** {data['employee_id']}\n"
        f"- **Name:** {data['employee_name']}\n"
        f"- **Leave type:** {data['leave_type']}\n"
        f"- **Dates:** {start} to {end} ({days} day{'s' if days != 1 else ''})\n"
        f"- **Reason:** {data['reason']}\n\n"
        "HR will review and update the status shortly."
    )


def check_leave(user_text: str) -> str:
    """Answer questions about existing leave records in the sheet."""
    print(f"[leave] check: {user_text!r}")
    sheet_id = os.environ["LEAVE_SHEET_ID"]
    rows = read_all_records(sheet_id)
    print(f"[leave] read {len(rows)} record(s)")

    # We hand the rows to the model and let it answer in natural language.
    # That keeps the agent flexible (balance by name, count pending, etc.)
    # without us having to hand-code every kind of query.
    context = (
        "Here are all current rows in the Leave Tracker sheet (JSON):\n"
        f"{json.dumps(rows, default=str, indent=2)}\n\n"
        f"Today's date is {_today_iso()}."
    )
    sys_prompt = (
        "You answer questions about an employee leave tracker. Use ONLY the "
        "rows provided. Be concise and friendly. If the question is about a "
        "specific employee, filter by name (case-insensitive). If the data "
        "is missing, say so."
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{context}\n\nQUESTION: {user_text}"},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Smoke test — `python -m backend.agents.leave`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 70)
    print("TEST 1: Bare request — should ask for ALL missing fields")
    print("=" * 70)
    print(submit_leave_request("I want to take leave"))

    print()
    print("=" * 70)
    print("TEST 2: Partial info — should ask only for what's still missing")
    print("=" * 70)
    print(submit_leave_request("I want leave June 10-12, family trip"))

    print()
    print("=" * 70)
    print("TEST 3: Complete request via fake conversation history")
    print("=" * 70)
    fake_history = [
        {"role": "user",      "content": "I want to take leave"},
        {"role": "assistant", "content": "Sure — could you share your ID, name, type, dates, and reason?"},
        {"role": "user",      "content": "EMP-501, Shreyash, Vacation, June 10-12, family trip"},
    ]
    print(submit_leave_request("EMP-501, Shreyash, Vacation, June 10-12, family trip", fake_history))

    print()
    print("=" * 70)
    print("TEST 4: Read existing leave records")
    print("=" * 70)
    print(check_leave("Show me all pending leave requests"))
