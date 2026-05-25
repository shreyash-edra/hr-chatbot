"""Orchestrator — OpenAI tool-calling router for the three HR sub-agents.

Design:
  - We expose four tools to the model: policy_questions, submit_leave_request,
    check_leave, submit_feedback.
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
from backend.agents.leave import check_leave, submit_leave_request
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
                "tool will inspect the full conversation history; if any of "
                "the six required fields are still missing (Employee ID, "
                "Employee Name, Leave Type, Start Date, End Date, Reason), "
                "the tool will return a message asking the user for the "
                "missing ones WITHOUT writing to the sheet. It only logs a "
                "row when every required field has been provided. NEVER "
                "fabricate fields the user hasn't given — let the tool ask."
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
                "questions like 'what leave have I taken?', 'how many days "
                "has X used?', 'show pending requests'. Use when the user "
                "wants information ABOUT existing leave records, not when "
                "they want to submit a new one."
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
  2. **Leave management** — submitting new leave requests OR asking about \
existing leave records. Use `submit_leave_request` for NEW requests and \
`check_leave` for questions about existing records.
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

LEAVE — strict completeness rules:
  - A leave request can ONLY be written to the sheet when ALL six of these \
have been provided by the user in this conversation:
      1. Employee ID
      2. Employee Name (full name)
      3. Leave Type (e.g. Vacation, Sick, Personal, Casual)
      4. Start Date
      5. End Date
      6. Reason for the leave
  - When the user mentions wanting leave, call `submit_leave_request` with \
their LATEST message — the tool reads conversation history itself and will \
return a friendly prompt asking for any missing fields WITHOUT writing a \
row. Keep calling it as the user provides more info; it only writes when \
all six fields are complete.
  - NEVER invent, default, or guess any of these fields. Never assume an \
employee ID like "N/A" or a leave type like "Vacation" on the user's behalf. \
If a field hasn't been stated by the user, it is missing.

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
  - For ambiguous messages, use conversation history to disambiguate. If \
still unclear, ask ONE short clarifying question before tool-calling.
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
