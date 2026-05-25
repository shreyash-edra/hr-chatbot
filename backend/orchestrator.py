"""Orchestrator — OpenAI tool-calling router for the three HR sub-agents.

Design:
  - We expose five tools to the model: policy_questions, submit_leave_request,
    check_leave, update_leave_status, submit_feedback.
  - The orchestrator's system prompt tells the model to pick exactly one tool
    per user message, or to politely decline if the message is off-topic
    (weather, news, trivia, etc.).
  - The conversation history is held per session_id in memory so multi-turn
    context is available to the router.
  - The user never sees the tool plumbing; we just return the model's final
    natural-language reply.

Why raw OpenAI tool-calling (no LangChain): the loop is ~30 lines, fully
debuggable, and there's no abstraction in the way when something goes wrong.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

from openai import OpenAI

from backend.agents.feedback import submit_feedback
from backend.agents.leave import check_leave, submit_leave_request, update_leave_status
from backend.agents.policy import answer_policy_question

# ---------------------------------------------------------------------------
# Tool schema exposed to the model
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "policy_questions",
            "description": (
                "Answer questions about company policies, benefits, vacation "
                "rules, holidays, expenses, leave entitlements, or anything "
                "documented in the HR policy materials. Use this for "
                "INFORMATIONAL questions — not for actually requesting leave."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's policy question, verbatim.",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_leave_request",
            "description": (
                "Try to log a NEW leave/time-off request to the Leave Tracker. "
                "Call this whenever the user is asking to TAKE leave. The "
                "tool will inspect the full conversation history. It requires "
                "start date, end date, and reason before logging. Employee ID, "
                "name, and leave type are optional, except the tool will ask "
                "for Employee ID when the supplied name is ambiguous and "
                "matches multiple known employee IDs. NEVER fabricate fields "
                "the user hasn't given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_text": {
                        "type": "string",
                        "description": (
                            "The user's latest leave-related message verbatim. "
                            "Don't try to compile earlier context yourself — "
                            "the tool reads conversation history on its own."
                        ),
                    }
                },
                "required": ["user_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_leave",
            "description": (
                "Read existing rows from the Leave Tracker sheet to answer "
                "questions like 'what is my leave status?', 'is my leave "
                "approved?', 'what leave have I taken?', 'how many days has X "
                "used?', 'show pending requests'. Use when the user wants "
                "information ABOUT existing leave records or approval status, "
                "not when they want to submit or update a leave request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_text": {
                        "type": "string",
                        "description": "The user's question about leave records.",
                    }
                },
                "required": ["user_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_leave_status",
            "description": (
                "Update an existing Leave Tracker row's Leave Status to "
                "Pending or Approved. Use when the user asks to approve, "
                "accept, mark approved, mark pending, or change the status of "
                "an existing leave request. The tool reads the sheet and uses "
                "the model to identify the correct row or ask for a concise "
                "clarifying detail if the row/status is ambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_text": {
                        "type": "string",
                        "description": "The user's leave status update request.",
                    }
                },
                "required": ["user_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_feedback",
            "description": (
                "Submit ANONYMOUS employee feedback to the Feedback Tracker. "
                "ONLY call this when the user has shared the ACTUAL feedback "
                "content (an opinion, complaint, compliment, suggestion, or "
                "observation). Do NOT call it when the user merely signals "
                "intent (e.g. 'I have feedback', 'I want to share something', "
                "'I have a suggestion') — in that case reply directly asking "
                "what they'd like to share. Sentiment is auto-classified. "
                "Never collects a name or employee ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feedback_text": {
                        "type": "string",
                        "description": (
                            "The actual feedback content verbatim from the "
                            "user (not their intent to share). Must contain "
                            "an opinion / observation / suggestion."
                        ),
                    }
                },
                "required": ["feedback_text"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are an HR Assistant chatbot. You help employees with \
exactly three things:

  1. **Policy questions** — company policies, benefits, vacation rules, \
holidays, expenses, leave entitlements. Use the `policy_questions` tool.
  2. **Leave management** — submitting new leave requests, asking about \
existing leave records/statuses, or updating an existing leave status. Use \
`submit_leave_request` for NEW requests, `check_leave` for questions about \
existing records/status, and `update_leave_status` for status changes.
  3. **Feedback collection** — anonymous suggestions, complaints, opinions, \
or compliments about ANYTHING in the workplace (food, equipment, \
processes, management, culture, environment, tools, perks, etc.). \
Use `submit_feedback`. Never ask for or include a name.

Routing rules:
  - Pick the tool(s) that fit the user's message. ONE tool when there's one \
intent; MULTIPLE tools (in the same response) when the user combines \
distinct intents in a single message. For example, "I want to take leave \
AND I have feedback that the coffee is bad" should call BOTH \
`submit_leave_request` AND `submit_feedback` in the same turn.
  - When you call multiple tools, you'll receive multiple tool results. \
Combine them into ONE friendly reply, summarising what was logged (or \
what's still needed) for each intent. Do not call one tool, wait, and \
then call another in a separate turn — issue all calls together.

LEAVE — logging rules:
  - A leave request can only be written to the sheet when these three \
details have been provided by the user in this conversation:
      1. Start Date
      2. End Date
      3. Reason for the leave
  - These details are optional for logging:
      1. Employee ID
      2. Employee Name
      3. Leave Type (e.g. Vacation, Sick, Personal, Casual)
      4. Leave Status (Pending or Approved)
  - If the user gives a status while logging a new leave request, such as \
"approved", "approve it", or "pending", include that in the logged row. If \
no status is given, the leave request is logged as Pending.
  - Employee ID is only required before writing when the supplied employee \
name is ambiguous and matches multiple known employee IDs in the leave \
tracker. In that case, the tool will ask the user for Employee ID so the \
entry can be differentiated.
  - When the user mentions wanting leave, call `submit_leave_request` with \
their LATEST message — the tool reads conversation history itself and will \
either log the request or ask for Employee ID if the name is ambiguous.
  - Do not ask the user for missing leave details before calling the tool. \
For leave submission intent, always call `submit_leave_request`; the tool \
will ask for missing required leave dates/reason or ambiguous Employee ID.
  - NEVER invent, default, or guess any of these fields. Never assume an \
employee ID like "N/A" or a leave type like "Vacation" on the user's behalf.

LEAVE — status read/update rules:
  - Use `check_leave` when the user asks whether a leave request is Pending \
or Approved, or asks to show/filter leave records by status.
  - Use `update_leave_status` when the user asks to approve, accept, mark \
approved, mark pending, or otherwise change an existing leave request status.

FEEDBACK — content required:
  - **Any opinion or complaint about the workplace is feedback** (food, \
equipment, processes, management, culture, environment, tools, perks). \
Route those to `submit_feedback`. Examples: "the office coffee is \
terrible", "the new chairs are amazing", "meetings run too long", "the \
wifi is slow".
  - But ONLY call `submit_feedback` when the user has shared the ACTUAL \
feedback content. If they only signal INTENT to give feedback — e.g. "I \
have feedback", "I want to share some feedback", "I have a suggestion" — \
do NOT call the tool. Reply directly asking what they'd like to share. \
Only after they share the actual content do you call `submit_feedback`.

Other rules:
  - Only decline when the topic is genuinely unrelated to the workplace: \
weather, news, sports, trivia, general chit-chat, coding help, math \
problems. In those cases reply that you can only help with HR policies, \
leave, and feedback.
  - For ambiguous non-leave messages, use conversation history to \
disambiguate. If still unclear, ask ONE short clarifying question before \
tool-calling.
  - For greetings ("hi", "hello"), reply warmly and briefly suggest the \
three things you can help with — do not call any tool.
  - If the user asks the current date, day, or time, answer directly using \
the CURRENT DATE/TIME provided below. Do NOT decline these. (Other off-topic \
questions still get a polite decline.)

When you do call a tool, you'll receive its result. The tool's result is \
already formatted nicely — deliver it verbatim as your reply. You may add \
at most ONE brief friendly opener sentence. Do NOT rewrite, summarize, or \
re-format the tool result."""


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _dispatch(name: str, args: dict, history: list[dict]) -> str:
    """Run the named tool with the given JSON args + conversation history."""
    if name == "policy_questions":
        return answer_policy_question(args.get("question", ""))
    if name == "submit_leave_request":
        # Pass history so the extractor can combine fields across turns
        # (date in one turn, name in the next, etc.).
        return submit_leave_request(args.get("user_text", ""), history)
    if name == "check_leave":
        return check_leave(args.get("user_text", ""))
    if name == "update_leave_status":
        return update_leave_status(args.get("user_text", ""))
    if name == "submit_feedback":
        return submit_feedback(args.get("feedback_text", ""))
    return f"(internal: unknown tool {name!r})"


# ---------------------------------------------------------------------------
# Per-session conversation history (in-memory; demo only)
# ---------------------------------------------------------------------------

_HISTORY: dict[str, list[dict]] = defaultdict(list)
_MAX_TURNS = 20  # keep recent context, drop older


def _trim_history(session_id: str) -> None:
    msgs = _HISTORY[session_id]
    if len(msgs) > _MAX_TURNS * 2:
        _HISTORY[session_id] = msgs[-_MAX_TURNS * 2 :]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle_message(session_id: str, user_message: str) -> str:
    """Route a user message through the tool-calling loop and return a reply."""
    print(f"\n[orchestrator] session={session_id!r} user={user_message!r}")

    history = _HISTORY[session_id]
    history.append({"role": "user", "content": user_message})

    # Inject today's date and time into the system prompt so the model can
    # resolve "today", "tomorrow", "next Monday" etc. and answer date-aware
    # questions correctly.
    now = datetime.now().astimezone()
    today_block = (
        f"\n\nCURRENT DATE/TIME: {now.strftime('%A, %B %d, %Y')} "
        f"({now.strftime('%Y-%m-%d')}) at {now.strftime('%I:%M %p %Z')}."
    )
    system_with_date = SYSTEM_PROMPT + today_block

    messages: list[dict] = [{"role": "system", "content": system_with_date}] + history

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

    # Tool-calling loop. In practice the model picks 0 or 1 tools per turn,
    # but we allow up to 3 iterations as a safety net.
    for step in range(3):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            # No tool — the model is replying directly (decline, greeting,
            # clarifying question).
            reply = (msg.content or "").strip()
            history.append({"role": "assistant", "content": reply})
            _trim_history(session_id)
            print(f"[orchestrator] direct reply (no tool) step={step}")
            return reply

        # Append the assistant turn (with the tool calls) to message list so
        # the subsequent tool-result messages have something to reference.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Execute each tool call and feed results back.
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"[orchestrator] tool call: {tool_name}({args})")
            try:
                result = _dispatch(tool_name, args, history)
            except Exception as e:
                print(f"[orchestrator] tool {tool_name} raised: {e!r}")
                result = (
                    f"Sorry, something went wrong while running {tool_name}: {e}. "
                    "Please try again, or contact HR if it persists."
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    # If we somehow exhausted the loop, fall back to the last tool result text.
    fallback = "Sorry — I had trouble completing that. Could you try rephrasing?"
    history.append({"role": "assistant", "content": fallback})
    _trim_history(session_id)
    return fallback


# ---------------------------------------------------------------------------
# Smoke test — `python -m backend.orchestrator`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    scenarios = [
        ("s1", "How many vacation days do I get?"),
        ("s2", "I want leave June 15-16, I'm Riya, attending a wedding"),
        ("s3", "The office coffee is terrible"),
        ("s4", "What's the weather today?"),
        ("s5", "Hi there"),
    ]
    for sid, msg in scenarios:
        print("\n" + "#" * 70)
        print(f"# session={sid}  user={msg}")
        print("#" * 70)
        print(handle_message(sid, msg))
