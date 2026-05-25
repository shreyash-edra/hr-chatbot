# HR Assistant Chatbot

A multi-agent HR chatbot built with Python, FastAPI, and raw OpenAI tool-calling.
One orchestrator routes each message to exactly one of three sub-agents:

1. **policy_questions** — RAG over Supabase pgvector (company policies, benefits, etc.)
2. **leave_management** — read/write a Google Sheet to log leave requests and check balances
3. **feedback_collection** — anonymous feedback capture with sentiment + action item

Off-topic questions (weather, news, trivia) get a polite decline.

---

## Architecture

```
User ──► /chat (FastAPI) ──► Orchestrator (OpenAI tool-calling)
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
        policy_questions   leave_management   feedback_collection
         (Supabase RAG)      (Google Sheets)    (Google Sheets)
```

---

## Project layout

```
hr-chatbot/
├── backend/
│   ├── main.py              # FastAPI app, serves frontend + POST /chat
│   ├── orchestrator.py      # tool-calling loop that routes to the 3 agents
│   ├── agents/
│   │   ├── policy.py
│   │   ├── leave.py
│   │   └── feedback.py
│   └── services/
│       ├── supabase_client.py
│       └── sheets_client.py
├── frontend/
│   └── index.html
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Python environment

```bash
cd hr-chatbot
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
```

Then fill in `.env`:

| Variable | What it is |
| --- | --- |
| `OPENAI_API_KEY` | Your OpenAI API key (from platform.openai.com) |
| `OPENAI_MODEL` | Defaults to `gpt-5-mini` — good cost/quality tradeoff |
| `SUPABASE_URL` | `https://<your-project>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | Supabase **service role** key (not anon) — used server-side only |
| `SUPABASE_TABLE` | Table holding embedded documents (default `documents`) |
| `SUPABASE_QUERY` | RPC function name for similarity search (default `match_documents`) |
| `GOOGLE_SA_JSON` | Path to your Google service-account JSON key |
| `LEAVE_SHEET_ID` | The Google Sheet ID for the Leave Tracker (from its URL) |
| `FEEDBACK_SHEET_ID` | The Google Sheet ID for the Feedback Tracker |

### 3. Google service account (one-time)

We use a Google **service account** (no OAuth dance) via `gspread`:

1. Go to https://console.cloud.google.com/ and create or select a project.
2. **APIs & Services → Library** → enable **Google Sheets API** and **Google Drive API**.
3. **APIs & Services → Credentials → Create credentials → Service account**.
   - Give it a name (e.g. `hr-assistant`). No roles needed for Sheets-only use.
4. Open the new service account → **Keys → Add key → Create new key → JSON**. Download it.
5. Save the JSON in this repo as `service-account.json` (it's already gitignored).
6. **Critical:** open each Google Sheet (Leave Tracker and Feedback Tracker), click **Share**, and share with the service account's email (the `client_email` field inside the JSON, e.g. `hr-assistant@your-project.iam.gserviceaccount.com`) as **Editor**. Without this, you'll see `403 PERMISSION_DENIED`.

### 4. Run

```bash
uvicorn backend.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

---

## Test scenarios

| Input | Expected behavior |
| --- | --- |
| "How many vacation days do I get?" | Policy agent answers from Supabase RAG |
| "I want to take leave June 10-12, I'm Shreyash, family trip" | Leave agent appends a Pending row to the sheet |
| "The office coffee is terrible" | Feedback agent appends an anonymous row with Negative sentiment |
| "What's the weather today?" | Polite decline — bot only handles policy/leave/feedback |

---

## Troubleshooting

- **403 from Google Sheets** → you forgot to share the sheet with the service account email.
- **Supabase RPC returns empty** → verify the `match_documents` function exists and the `documents` table has rows with non-null embeddings.
- **OpenAI 401** → check the API key in `.env`.
- **Module not found** → make sure your venv is activated and `pip install -r requirements.txt` succeeded.
