"""Stripe integration — card-funded wallet top-ups with automatic tax (Stripe Tax).

Mintty's wallet is normally funded over Lightning (see l402.py). This module adds
a card on-ramp: the user pays in USD via Stripe Checkout, Stripe Tax calculates
any tax owed based on their billing address, and once the payment succeeds the
webhook credits the same demo balance that Lightning top-ups use.
"""

import stripe

from backend.config import (
    BASE_URL,
    DEMO_MODE,
    SATS_PER_USD,
    STRIPE_SECRET_KEY,
    STRIPE_WEBHOOK_SECRET,
)
from backend.db import (
    credit_demo_funded_sats,
    record_stripe_topup,
    stripe_session_processed,
)

stripe.api_key = STRIPE_SECRET_KEY


class StripeError(Exception):
    pass


def usd_to_sats(amount_usd: float) -> int:
    return round(amount_usd * SATS_PER_USD)


async def create_topup_checkout_session(amount_usd: float) -> dict:
    """Create a Stripe Checkout Session to top up the wallet with a card.

    automatic_tax is enabled so Stripe Tax calculates any tax owed from the
    billing address collected at checkout.
    """
    if not STRIPE_SECRET_KEY:
        raise StripeError("Stripe is not configured — set STRIPE_SECRET_KEY")
    if amount_usd <= 0:
        raise StripeError("amount_usd must be positive")

    sats = usd_to_sats(amount_usd)
    cents = round(amount_usd * 100)

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": f"Mintty wallet top-up — {sats:,} sats"},
                        "unit_amount": cents,
                    },
                    "quantity": 1,
                }
            ],
            automatic_tax={"enabled": True},
            billing_address_collection="required",
            success_url=f"{BASE_URL}/?funded=1&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{BASE_URL}/?funded=0",
            metadata={"sats": str(sats)},
        )
    except stripe.error.StripeError as e:
        raise StripeError(str(e)) from e

    return {"url": session.url, "session_id": session.id}


async def handle_webhook_event(payload: bytes, sig_header: str) -> dict:
    """Verify a Stripe webhook event and credit the wallet on a completed checkout.

    Idempotent — Stripe retries webhook deliveries, so a session already
    recorded in stripe_topups is skipped rather than double-credited.
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise StripeError("Stripe webhook is not configured — set STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        raise StripeError(f"Invalid webhook payload: {e}") from e

    if event["type"] != "checkout.session.completed":
        return {"handled": False}

    # event["data"]["object"] is a StripeObject, not a plain dict — .to_dict()
    # first so the .get() calls below work.
    session = event["data"]["object"].to_dict()
    if session.get("payment_status") != "paid":
        return {"handled": False}

    session_id = session["id"]
    if await stripe_session_processed(session_id):
        return {"handled": True, "duplicate": True}

    sats = int((session.get("metadata") or {}).get("sats") or 0)
    amount_total = session.get("amount_total") or 0
    tax_amount = (session.get("total_details") or {}).get("amount_tax") or 0

    # Demo mode only: Mintty has no live exchange integration, so a card top-up
    # can't yet mint real sats in a real Lightning wallet. It still records the
    # payment for the audit trail either way.
    if sats > 0 and DEMO_MODE:
        await credit_demo_funded_sats(sats)

    await record_stripe_topup(
        session_id=session_id,
        amount_usd_cents=amount_total,
        tax_usd_cents=tax_amount,
        sats_credited=sats if DEMO_MODE else 0,
    )
    return {"handled": True, "sats_credited": sats if DEMO_MODE else 0}
