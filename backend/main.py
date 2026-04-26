"""FastAPI application — SSE endpoint for the agentic payments agent."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.agent import run_agent
from backend.budget import (
    DAILY_LIMIT_KEY,
    PER_TX_LIMIT_KEY,
    get_daily_limit,
    get_per_tx_limit,
    get_remaining_budget,
)
from backend.db import (
    close_db,
    get_db,
    get_last_session_id,
    get_session_messages,
    get_transactions,
    list_sessions,
    save_message,
    set_setting,
)
from backend.l402 import L402Error, create_receive_invoice, get_wallet_balance

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure DB is initialized
    await get_db()
    yield
    # Shutdown: close DB
    await close_db()


app = FastAPI(
    title="Mintty",
    description="AI agent that can browse the web and make Lightning payments via L402",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


class AgentRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.post("/agent")
async def agent_endpoint(req: AgentRequest):
    """Stream agent responses as SSE events with persistent conversation memory."""
    sid = req.session_id or "default"

    # Load history from DB
    history = await get_session_messages(sid)

    # Persist the user message
    await save_message(sid, "user", req.query)

    async def event_generator():
        assistant_text = ""
        async for event in run_agent(req.query, history=history):
            if event.event == "result":
                assistant_text = event.data.get("text", "")
            yield {"event": event.event, "data": json.dumps(event.data)}

        # Persist the assistant response
        if assistant_text:
            await save_message(sid, "assistant", assistant_text)

    return EventSourceResponse(event_generator())


@app.get("/session/last")
async def last_session_endpoint():
    """Return the most recently active session ID and its messages."""
    sid = await get_last_session_id()
    if not sid:
        return {"session_id": None, "messages": []}
    messages = await get_session_messages(sid)
    return {"session_id": sid, "messages": messages}


@app.get("/sessions")
async def sessions_list_endpoint():
    """List all past sessions."""
    sessions = await list_sessions()
    return {"sessions": sessions}


@app.get("/session/{session_id}")
async def session_endpoint(session_id: str):
    """Return messages for a specific session."""
    messages = await get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@app.get("/budget")
async def budget_endpoint():
    """Get remaining daily budget."""
    remaining = await get_remaining_budget()
    return {"remaining_sats": remaining}


@app.get("/transactions")
async def transactions_endpoint(limit: int = 50):
    """Get recent transactions."""
    txns = await get_transactions(limit=limit)
    return {"transactions": txns}


class ReceiveRequest(BaseModel):
    amount_sats: int | None = None


@app.get("/wallet/balance")
async def wallet_balance_endpoint():
    """Get actual wallet balance from MDK."""
    try:
        balance = await get_wallet_balance()
        return {"balance_sats": balance}
    except L402Error as e:
        return {"balance_sats": 0, "error": str(e)}


@app.post("/wallet/receive")
async def wallet_receive_endpoint(req: ReceiveRequest):
    """Generate a Lightning invoice to fund the wallet."""
    try:
        result = await create_receive_invoice(req.amount_sats)
        return result
    except L402Error as e:
        return {"error": str(e)}


@app.get("/settings/budget")
async def get_budget_settings():
    """Get current budget limits."""
    return {
        "daily_budget_sats": await get_daily_limit(),
        "max_per_purchase_sats": await get_per_tx_limit(),
        "remaining_today_sats": await get_remaining_budget(),
    }


class BudgetSettingsRequest(BaseModel):
    daily_budget_sats: int | None = None
    max_per_purchase_sats: int | None = None


@app.post("/settings/budget")
async def update_budget_settings(req: BudgetSettingsRequest):
    """Update budget limits."""
    if req.daily_budget_sats is not None:
        if req.daily_budget_sats < 0:
            return {"error": "daily_budget_sats must be non-negative"}
        await set_setting(DAILY_LIMIT_KEY, str(req.daily_budget_sats))
    if req.max_per_purchase_sats is not None:
        if req.max_per_purchase_sats < 0:
            return {"error": "max_per_purchase_sats must be non-negative"}
        await set_setting(PER_TX_LIMIT_KEY, str(req.max_per_purchase_sats))
    return await get_budget_settings()


@app.get("/")
async def index():
    """Serve the frontend."""
    return HTMLResponse((_FRONTEND_DIR / "index.html").read_text())


@app.get("/audit")
async def audit():
    """Serve the audit log page."""
    return HTMLResponse((_FRONTEND_DIR / "audit.html").read_text())
