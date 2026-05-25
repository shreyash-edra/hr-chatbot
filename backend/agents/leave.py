"""Leave management agent — reads from and writes to the Leave Tracker sheet.

Leave Tracker column order (Sheet1):
  Employee ID | Employee Name | Leave Type | Start Date | End Date |
  Number of Days | Leave Status | Requested On | Approval Date |
  Comments/Reason

A new row is appended when the leave dates and reason are supplied. Employee
ID, employee name, and leave type are optional. If the supplied employee name
matches multiple known employee IDs in the leave tracker, the agent asks for
the Employee ID before writing the row. The auto-filled columns are:
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

USER_FIELDS: list[tuple[str, str]] = [
    ("employee_id",   "your **Employee ID**"),
    ("employee_name", "your **full name**"),
    ("leave_type",    "the **leave type** (e.g. Vacation, Sick, Personal, Casual)"),
    ("start_date",    "the **start date**"),
    ("end_date",      "the **end date**"),
    ("reason",        "a brief **reason** for the leave"),
]

REQUIRED_USER_FIELDS: list[tuple[str, str]] = [
    ("start_date", "the **start date**"),
    ("end_date", "the **end date**"),
    ("reason", "a brief **reason** for the leave"),
]
REQUIRED_USER_KEYS = {key for key, _label in REQUIRED_USER_FIELDS}
OPTIONAL_DETAILS_TEXT = (
    "Employee ID, name, and leave type are optional. I'll only ask for "
    "Employee ID if I need it to distinguish between matching names."
)


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

    # Normalise — make sure every known key exists, as a string.
    for key, _label in USER_FIELDS:
        v = data.get(key)
        data[key] = (v.strip() if isinstance(v, str) else "") if v else ""
    if "number_of_days" not in data or not isinstance(data["number_of_days"], int):
        data["number_of_days"] = 0

    return data


def _compute_days(start: str, end: str) -> int:
    """Inclusive day count between two YYYY-MM-DD dates (best effort)."""
    try:
        s = datetime.strptime(start, "%Y-%m-%d").date()
        e = datetime.strptime(end, "%Y-%m-%d").date()
        return (e - s).days + 1
    except (ValueError, TypeError):
        return 0


def _normalise_name(name: str) -> str:
    """Case-insensitive, whitespace-insensitive name comparison."""
    return " ".join(name.casefold().split())


def _matching_employee_ids(employee_name: str) -> list[str]:
    """Return unique known employee IDs for an exact name match in the sheet."""
    if not employee_name:
        return []

    sheet_id = os.environ["LEAVE_SHEET_ID"]
    rows = read_all_records(sheet_id)
    target = _normalise_name(employee_name)
    ids: list[str] = []
    seen: set[str] = set()

    for row in rows:
        row_name = str(row.get("Employee Name", "")).strip()
        employee_id = str(row.get("Employee ID", "")).strip()
        if not row_name or not employee_id:
            continue
        if _normalise_name(row_name) != target:
            continue
        folded_id = employee_id.casefold()
        if folded_id not in seen:
            ids.append(employee_id)
            seen.add(folded_id)

    return ids


# ---------------------------------------------------------------------------
# Public actions
# ---------------------------------------------------------------------------

def submit_leave_request(
    user_text: str,
    history: list[dict] | None = None,
) -> str:
    """Parse a leave request from the conversation and append a Pending row.

    Start date, end date, and reason are required. Employee ID, name, and
    leave type are optional unless the provided name is ambiguous.
    """
    print(f"[leave] submit request: {user_text!r}")
    data = _extract_request(user_text, history)
    print(f"[leave] extracted: {data}")

    if data["employee_name"] and not data["employee_id"]:
        matching_ids = _matching_employee_ids(data["employee_name"])
        if len(matching_ids) > 1:
            print(f"[leave] ambiguous employee name — matching IDs: {matching_ids}")
            bullets = "\n".join(f"- {employee_id}" for employee_id in matching_ids)
            return (
                f"I found more than one employee named **{data['employee_name']}**. "
                "Employee ID is needed here only to log this against the right "
                f"person:\n\n{bullets}\n\nCould you share the correct Employee ID?"
            )

    missing_required = []
    for key, label in REQUIRED_USER_FIELDS:
        if not data.get(key):
            missing_required.append(label)
    if missing_required:
        print(f"[leave] incomplete — missing required fields: {missing_required}")
        if len(missing_required) == 1:
            return (
                f"Before I can log this leave request, I still need {missing_required[0]}. "
                f"Could you share that?\n\n{OPTIONAL_DETAILS_TEXT}"
            )
        bullets = "\n".join(f"- {m}" for m in missing_required)
        return (
            "Before I can log this leave request, I need these details:\n\n"
            f"{bullets}\n\n"
            f"{OPTIONAL_DETAILS_TEXT}"
        )

    missing_optional = [
        key
        for key, _label in USER_FIELDS
        if key not in REQUIRED_USER_KEYS
        and not data.get(key)
    ]
    if missing_optional:
        print(f"[leave] optional fields not provided: {missing_optional}")

    # Compute days when dates are present; otherwise leave the count blank.
    start = data["start_date"]
    end   = data["end_date"]
    days  = data["number_of_days"] or _compute_days(start, end)
    days_cell = days or ""

    row = [
        data["employee_id"],
        data["employee_name"],
        data["leave_type"],
        start,
        end,
        days_cell,
        "Pending",
        _today_iso(),  # Requested On
        "",            # Approval Date (blank)
        data["reason"],
    ]
    assert len(row) == len(LEAVE_HEADERS), "row length must match header count"

    sheet_id = os.environ["LEAVE_SHEET_ID"]
    append_row(sheet_id, row)
    print(f"[leave] appended row: {row}")

    details = [
        f"- **Employee ID:** {data['employee_id'] or 'Not provided'}",
        f"- **Name:** {data['employee_name'] or 'Not provided'}",
        f"- **Leave type:** {data['leave_type'] or 'Not provided'}",
    ]
    if start or end:
        date_text = f"{start or 'Not provided'} to {end or 'Not provided'}"
        if days:
            date_text = f"{date_text} ({days} day{'s' if days != 1 else ''})"
        details.append(f"- **Dates:** {date_text}")
    else:
        details.append("- **Dates:** Not provided")
    details.append(f"- **Reason:** {data['reason'] or 'Not provided'}")

    return (
        "Got it — I've logged your leave request as **Pending** with the "
        "details available so far:\n\n"
        f"{chr(10).join(details)}\n\n"
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
    print("TEST 1: Bare request — should ask for dates and reason")
    print("=" * 70)
    print(submit_leave_request("I want to take leave"))

    print()
    print("=" * 70)
    print("TEST 2: Partial info — should log with optional fields blank")
    print("=" * 70)
    print(submit_leave_request("I want leave June 10-12, family trip"))

    print()
    print("=" * 70)
    print("TEST 3: Complete request via fake conversation history")
    print("=" * 70)
    fake_history = [
        {"role": "user",      "content": "I want to take leave"},
        {"role": "assistant", "content": "Sure — share whichever details you have."},
        {"role": "user",      "content": "EMP-501, Shreyash, Vacation, June 10-12, family trip"},
    ]
    print(submit_leave_request("EMP-501, Shreyash, Vacation, June 10-12, family trip", fake_history))

    print()
    print("=" * 70)
    print("TEST 4: Read existing leave records")
    print("=" * 70)
    print(check_leave("Show me all pending leave requests"))
