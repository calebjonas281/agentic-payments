"""Application configuration — all tunables in one place."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Paths
DATA_DIR = Path(os.environ.get("AP_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DB_PATH = DATA_DIR / "payments.db"

# Budget
DAILY_BUDGET_SATS = int(os.environ.get("AP_DAILY_BUDGET_SATS", "1000"))

# Agent
AGENT_MODEL = os.environ.get("AP_AGENT_MODEL", "claude-haiku-4-5-20251001")
AGENT_MAX_TURNS = int(os.environ.get("AP_AGENT_MAX_TURNS", "25"))

# Lightning — MoneyDevKit agent-wallet CLI
# Default invocation: npx @moneydevkit/agent-wallet@latest <command>
# Override with AP_MDK_CMD for custom installs (e.g. a global binary path)
MDK_CMD: list[str] = os.environ.get(
    "AP_MDK_CMD", "npx @moneydevkit/agent-wallet@latest"
).split()

# Demo mode — simulate a funded wallet without real Lightning
DEMO_MODE: bool = os.environ.get("AP_DEMO_MODE", "true").lower() in ("1", "true", "yes")
DEMO_BALANCE_SATS: int = int(os.environ.get("AP_DEMO_BALANCE_SATS", "50000"))

# Vendor blocklist — comma-separated domains the agent must never pay
_blocklist_raw = os.environ.get("AP_VENDOR_BLOCKLIST", "")
VENDOR_BLOCKLIST: set[str] = {d.strip() for d in _blocklist_raw.split(",") if d.strip()}

# Amadeus flight API (https://developers.amadeus.com — free self-service tier)
AMADEUS_CLIENT_ID = os.environ.get("AMADEUS_CLIENT_ID", "")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET", "")
AMADEUS_BASE_URL = os.environ.get("AMADEUS_BASE_URL", "https://test.api.amadeus.com")

# SerpAPI — Google Flights scraper (https://serpapi.com — 100 free searches/month)
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
