"""Agent loop — Claude-powered tool-use loop that yields SSE events."""

import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import anthropic

from backend.config import AGENT_MAX_TURNS, AGENT_MODEL
from backend.tools import (
    TOOL_SCHEMAS,
    execute_tool,
    narrate_tool_call,
    narrate_tool_result,
)

def _build_system_prompt() -> str:
    from datetime import datetime
    today = datetime.now().strftime("%A, %B %-d, %Y")
    return "Today's date is " + today + ".\n\n" + _PROMPT_BODY

_PROMPT_BODY = \
"""\
You are a highly capable AI assistant — like ChatGPT or Claude — but with one \
superpower: you can actually buy things and book travel on behalf of the user \
using Bitcoin Lightning payments.

**You can do everything a great AI assistant does:**
- Answer questions on any topic with depth and accuracy
- Explain complex subjects clearly
- Compare options and give recommendations
- Do multi-step research across multiple sources
- Reason through decisions and trade-offs
- Have natural back-and-forth conversations
- Help with planning, analysis, and advice

**Your extra capabilities (beyond a normal AI):**

Research with real data:
- Use web_search and fetch_url to get live, up-to-date information
- If a website charges for access (HTTP 402), the system pays automatically via Lightning

Shopping:
- Try shop_search first to find products on Amazon or Walmart
- If shop_search returns an error, empty results, a timeout, or any failure — NEVER mention \
  the failure to the user. Never say "technical issue", "connectivity issue", "data source", \
  or anything that implies a problem. Just immediately use web_search to find the product \
  on Amazon or Walmart, fetch_url the product pages to get title, image, and price, then \
  present results as image cards as if nothing happened. This fallback is mandatory.
- Pick the 2 best matches and present them as image cards using the "cards" field type (see below)
- After the user picks one, use shop_quote to get exact pricing with shipping, show the total, then ask "Shall I order this?" before calling shop_order
- Only call shop_order after explicit confirmation

Flight booking:
- Use flight_search with IATA codes (JFK, LAX, LHR, CDG, etc.)
- If flight_search returns a "note" instead of results, follow its instruction — \
  it means you should use web_search + fetch_url to find real flights from sites \
  like Google Flights, Kayak, or Expedia. Extract actual flight options with prices \
  and present them clearly. Never tell the user there is a "technical issue" — just \
  use web search instead.
- Show options: airline, times, stops, duration, price
- Collect name, date of birth, email, phone, passport (for international) before booking
- ALWAYS confirm full details and ask "Shall I book this?" before calling flight_book

**How to ask questions — this is important:**
When you need information from the user, render an interactive form using this syntax:

:::form
{
  "title": "Optional title",
  "fields": [
    {"id": "x", "type": "chips", "label": "Question?", "options": ["A", "B", "C"]},
    {"id": "y", "type": "text", "label": "Question?", "placeholder": "hint..."},
    {"id": "z", "type": "textarea", "label": "Question?", "placeholder": "hint..."}
  ]
}
:::

Field types:
- "chips" = clickable option buttons (use for cabin class, trip type, budget range, yes/no, etc.)
- "cards" = image cards (use when presenting product choices — each option has an image, label, and sublabel)
- "text" = short text input
- "textarea" = long text input

For "cards", format options as objects: {"id": "PRODUCT_ID", "label": "Product name", "image": "THUMBNAIL_URL", "sublabel": "$XX.XX"}
Example cards field: {"id": "pick", "type": "cards", "label": "Which one?", "options": [{"id": "B001", "label": "Nike Air Max", "image": "https://...", "sublabel": "$89"}, ...]}

Use text/textarea for open-ended answers (destination, dates, notes).
Always convert dates to the right format yourself — never expose YYYY-MM-DD to the user.
Only output one :::form block per message. Add any intro text before the form, not after.

**Hard rules — never break these:**
- Never place an order or book a flight without the user's explicit confirmation
- Never buy something without showing the full price first
- Always show sources and URLs for factual claims
- If you don't know something, say so — don't make things up

Be conversational, thorough, and genuinely helpful. You are not limited to \
shopping topics — help with whatever the user needs, and use your buying \
capabilities when relevant.\
"""

SYSTEM_PROMPT = _build_system_prompt()


@dataclass
class SSEEvent:
    """A single event to be sent to the client via SSE."""

    event: str  # "step", "result", "result_delta", "error", "done"
    data: dict[str, Any]


async def run_agent(
    query: str, *, history: list[dict[str, Any]] | None = None
) -> AsyncGenerator[SSEEvent, None]:
    """Run the agent loop, yielding SSE events as the agent works.

    Event types:
    - step:         The agent is doing something (tool call with narration)
    - result_delta: A chunk of the final answer (streamed token-by-token)
    - result:       The complete final answer (sent after all deltas)
    - error:        Something went wrong
    - done:         Stream is finished
    """
    try:
        async for event in _agent_loop(query, history=history):
            yield event
    except Exception as e:
        yield SSEEvent(event="error", data={"text": f"Unexpected error: {e}"})
    finally:
        yield SSEEvent(event="done", data={})


async def _agent_loop(
    query: str, *, history: list[dict[str, Any]] | None = None
) -> AsyncGenerator[SSEEvent, None]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        yield SSEEvent(
            event="error",
            data={
                "text": (
                    "Missing ANTHROPIC_API_KEY. Restart the app with your "
                    "Anthropic API key to use the agent."
                )
            },
        )
        return

    client = anthropic.AsyncAnthropic()

    # Build messages with conversation history for multi-turn support
    messages: list[dict[str, Any]] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    yield SSEEvent(event="step", data={"text": "Thinking..."})

    for turn in range(AGENT_MAX_TURNS):
        # Use streaming API for faster time-to-first-token
        try:
            collected = await _stream_response(client, messages)
        except anthropic.APIError as e:
            yield SSEEvent(event="error", data={"text": f"API error: {e}"})
            return
        except Exception as e:
            yield SSEEvent(event="error", data={"text": f"Stream error: {e}"})
            return

        text_parts = collected["text_parts"]
        tool_use_blocks = collected["tool_use_blocks"]
        raw_content = collected["raw_content"]

        # If no tool calls, stream the final answer
        if not tool_use_blocks:
            final_text = "\n".join(text_parts)
            # Stream in chunks for faster perceived response
            for chunk in _chunk_text(final_text, 60):
                yield SSEEvent(event="result_delta", data={"text": chunk})
            yield SSEEvent(event="result", data={"text": final_text})
            return

        # Emit any intermediate text
        for text in text_parts:
            if text.strip():
                yield SSEEvent(
                    event="step", data={"text": text, "type": "thinking"}
                )

        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": raw_content})

        # Execute each tool call
        tool_results: list[dict[str, Any]] = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block["name"]
            tool_args = tool_block["input"]

            narration = narrate_tool_call(tool_name, tool_args)
            yield SSEEvent(
                event="step",
                data={"text": narration, "type": "tool_call", "tool": tool_name},
            )

            result = await execute_tool(tool_name, tool_args)

            if result.narration:
                yield SSEEvent(
                    event="step",
                    data={
                        "text": result.narration,
                        "type": "tool_result",
                        "tool": tool_name,
                    },
                )

            extra = narrate_tool_result(tool_name, result.output)
            if extra and extra != result.narration:
                yield SSEEvent(
                    event="step",
                    data={"text": extra, "type": "tool_result", "tool": tool_name},
                )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_block["id"],
                    "content": json.dumps(result.output),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    yield SSEEvent(
        event="error",
        data={"text": f"Agent reached maximum turns ({AGENT_MAX_TURNS}) without finishing."},
    )


async def _stream_response(
    client: anthropic.AsyncAnthropic,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call Claude with streaming and collect the full response.

    Returns a dict with:
    - text_parts: list of text strings from text blocks
    - tool_use_blocks: list of dicts with name, id, input
    - raw_content: the raw content blocks for message history
    """
    text_parts: list[str] = []
    tool_use_blocks: list[dict[str, Any]] = []
    raw_content = []

    current_text = ""
    current_tool: dict[str, Any] | None = None
    current_tool_json = ""

    async with client.messages.stream(
        model=AGENT_MODEL,
        max_tokens=8096,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=TOOL_SCHEMAS,
        messages=messages,
    ) as stream:
        async for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "text":
                    current_text = ""
                elif event.content_block.type == "tool_use":
                    current_tool = {
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": {},
                    }
                    current_tool_json = ""

            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    current_text += event.delta.text
                elif event.delta.type == "input_json_delta":
                    current_tool_json += event.delta.partial_json

            elif event.type == "content_block_stop":
                if current_text:
                    text_parts.append(current_text)
                    raw_content.append({"type": "text", "text": current_text})
                    current_text = ""
                if current_tool is not None:
                    if current_tool_json:
                        current_tool["input"] = json.loads(current_tool_json)
                    tool_use_blocks.append(current_tool)
                    raw_content.append({
                        "type": "tool_use",
                        "id": current_tool["id"],
                        "name": current_tool["name"],
                        "input": current_tool["input"],
                    })
                    current_tool = None
                    current_tool_json = ""

    return {
        "text_parts": text_parts,
        "tool_use_blocks": tool_use_blocks,
        "raw_content": raw_content,
    }


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into chunks for streaming to the frontend."""
    if len(text) <= size:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + size, len(text))
        # Try to break at a space or newline
        if end < len(text):
            brk = text.rfind(" ", i, end)
            nl = text.rfind("\n", i, end)
            best = max(brk, nl)
            if best > i:
                end = best + 1
        chunks.append(text[i:end])
        i = end
    return chunks
