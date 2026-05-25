"""Google Sheets client using a service account (no OAuth dance).

Provides thin read/write helpers used by the leave and feedback agents.
The service account JSON path is taken from GOOGLE_SA_JSON; both target
sheets are referenced by their spreadsheet ID and (for now) the default
worksheet/tab name "Sheet1".
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_TAB = "Sheet1"


@lru_cache(maxsize=1)
def _client() -> gspread.Client:
    """Authenticate once per process and reuse the client."""
    sa_path = os.environ.get("GOOGLE_SA_JSON", "./service-account.json")
    if not os.path.isfile(sa_path):
        raise FileNotFoundError(
            f"Google service-account JSON not found at {sa_path!r}. "
            "Set GOOGLE_SA_JSON in .env to its absolute or relative path."
        )
    creds = Credentials.from_service_account_file(sa_path, scopes=SCOPES)
    return gspread.authorize(creds)


def _worksheet(sheet_id: str, tab: str = DEFAULT_TAB) -> gspread.Worksheet:
    try:
        ws = _client().open_by_key(sheet_id).worksheet(tab)
    except gspread.exceptions.APIError as e:
        # The most common cause is forgetting to share the sheet with the
        # service-account email — surface a helpful hint.
        raise RuntimeError(
            f"Google Sheets API error opening {sheet_id!r} (tab {tab!r}): {e}. "
            "If this is 403/PERMISSION_DENIED, share the sheet as Editor with "
            "the `client_email` from your service-account JSON."
        ) from e
    return ws


def append_row(sheet_id: str, row: list[Any], tab: str = DEFAULT_TAB) -> None:
    """Append a single row to the bottom of a sheet."""
    _worksheet(sheet_id, tab).append_row(row, value_input_option="USER_ENTERED")


def read_all_records(sheet_id: str, tab: str = DEFAULT_TAB) -> list[dict[str, Any]]:
    """Return all rows as list[dict] keyed by the header row."""
    return _worksheet(sheet_id, tab).get_all_records()


def read_headers(sheet_id: str, tab: str = DEFAULT_TAB) -> list[str]:
    """Return the header row (row 1)."""
    return _worksheet(sheet_id, tab).row_values(1)


def update_cell_by_header(
    sheet_id: str,
    row_number: int,
    header: str,
    value: Any,
    tab: str = DEFAULT_TAB,
) -> None:
    """Update one cell by 1-based sheet row number and header name."""
    ws = _worksheet(sheet_id, tab)
    headers = ws.row_values(1)
    if header not in headers:
        raise ValueError(f"Header {header!r} not found in sheet")
    col_number = headers.index(header) + 1
    ws.update_cell(row_number, col_number, value)


# ---------------------------------------------------------------------------
# Smoke test — `python -m backend.services.sheets_client`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    leave_id = os.environ.get("LEAVE_SHEET_ID")
    feedback_id = os.environ.get("FEEDBACK_SHEET_ID")

    if leave_id:
        print(f"[sheets] Leave sheet headers: {read_headers(leave_id)}")
        rows = read_all_records(leave_id)
        print(f"[sheets] Leave sheet has {len(rows)} data row(s)")
    else:
        print("[sheets] LEAVE_SHEET_ID not set — skipping")

    if feedback_id:
        print(f"[sheets] Feedback sheet headers: {read_headers(feedback_id)}")
        rows = read_all_records(feedback_id)
        print(f"[sheets] Feedback sheet has {len(rows)} data row(s)")
    else:
        print("[sheets] FEEDBACK_SHEET_ID not set — skipping")
