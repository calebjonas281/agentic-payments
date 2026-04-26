"""Budget enforcement — prevents the agent from overspending."""

from backend.config import DAILY_BUDGET_SATS
from backend.db import get_setting, sum_spending_today

# Setting keys
DAILY_LIMIT_KEY = "daily_budget_sats"
PER_TX_LIMIT_KEY = "max_per_purchase_sats"

# Defaults (used if the user hasn't configured anything)
DEFAULT_DAILY = DAILY_BUDGET_SATS
DEFAULT_PER_TX = 50_000  # ~$50 safety cap per single purchase


class BudgetExceeded(Exception):
    def __init__(self, requested: int, spent: int, limit: int) -> None:
        self.requested = requested
        self.spent = spent
        self.limit = limit
        remaining = max(0, limit - spent)
        super().__init__(
            f"Over daily budget — this purchase costs {requested:,} sats but only "
            f"{remaining:,} sats remain today (spent {spent:,} of {limit:,} sats). "
            f"Add funds to the wallet or raise the daily budget in wallet settings."
        )


class PurchaseTooLarge(Exception):
    def __init__(self, requested: int, limit: int) -> None:
        super().__init__(
            f"Purchase too large — {requested:,} sats exceeds the single-purchase "
            f"limit of {limit:,} sats. Raise the per-purchase cap in wallet settings."
        )


async def get_daily_limit() -> int:
    raw = await get_setting(DAILY_LIMIT_KEY, str(DEFAULT_DAILY))
    return int(raw)


async def get_per_tx_limit() -> int:
    raw = await get_setting(PER_TX_LIMIT_KEY, str(DEFAULT_PER_TX))
    return int(raw)


async def check_budget(amount_sats: int) -> int:
    """Check whether *amount_sats* fits within today's daily budget.

    Returns the remaining budget (after this payment) on success.
    Raises BudgetExceeded if the payment would bust the daily limit.
    Raises PurchaseTooLarge if it exceeds the per-transaction cap.
    """
    per_tx = await get_per_tx_limit()
    if amount_sats > per_tx:
        raise PurchaseTooLarge(amount_sats, per_tx)

    daily = await get_daily_limit()
    spent = await sum_spending_today()
    if spent + amount_sats > daily:
        raise BudgetExceeded(amount_sats, spent, daily)
    return daily - spent - amount_sats


async def get_remaining_budget() -> int:
    """Return how many sats the agent can still spend today."""
    daily = await get_daily_limit()
    spent = await sum_spending_today()
    return max(0, daily - spent)
