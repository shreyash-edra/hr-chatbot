"""FastAPI entry point — serves the frontend and exposes POST /chat.

Run with:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /            → returns the chat UI (frontend/index.html)
    POST /chat        → {message, session_id} → {reply}
    GET  /health      → simple liveness probe
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Load .env BEFORE importing the orchestrator (which reads env vars at call
# time, not import time, but loading early keeps logs honest).
load_dotenv()

from backend.orchestrator import handle_message  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

app = FastAPI(title="HR Assistant", version="1.0.0")

# CORS — permissive for local demo; tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or uuid.uuid4().hex
    try:
        reply = handle_message(session_id, req.message)
    except KeyError as e:
        # Missing env var — surface a clear 500 instead of a stack trace.
        raise HTTPException(
            status_code=500,
            detail=f"Server misconfigured: missing env var {e}",
        )
    except Exception as e:
        print(f"[main] /chat unexpected error: {e!r}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    return ChatResponse(reply=reply, session_id=session_id)


@app.get("/")
def root() -> FileResponse:
    if not INDEX_HTML.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Frontend not found at {INDEX_HTML}",
        )
    return FileResponse(INDEX_HTML)
