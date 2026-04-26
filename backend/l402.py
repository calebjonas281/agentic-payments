"""L402 protocol handler — detects 402 responses, extracts invoices, pays via mdk."""

import asyncio
import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from backend.budget import get_daily_limit, get_per_tx_limit, PurchaseTooLarge
from backend.config import MDK_CMD, VENDOR_BLOCKLIST, DEMO_MODE, DEMO_BALANCE_SATS
from backend.db import (
    check_budget_and_reserve,
    payment_hash_exists,
    update_transaction_status,
)


class L402Error(Exception):
    pass


class UnsafeURL(L402Error):
    pass


class VendorBlocked(L402Error):
    pass


class DuplicatePayment(L402Error):
    pass


@dataclass
class L402Challenge:
    """Parsed L402 challenge from a WWW-Authenticate header."""

    macaroon: str
    invoice: str
    payment_hash: str
    amount_sats: int | None  # decoded from invoice if possible


def parse_l402_challenge(
    www_authenticate: str, response_body: str | None = None
) -> L402Challenge:
    """Parse an L402 challenge from a 402 response.

    MDK servers return structured JSON in the response body:
        {"macaroon":"eyJ...", "invoice":"lnbc...", "paymentHash":"abc...", "amountSats":100}

    Falls back to parsing the WWW-Authenticate header if the body is missing
    or doesn't contain the expected fields.
    """
    macaroon: str | None = None
    invoice: str | None = None
    payment_hash: str | None = None
    amount_sats: int | None = None

    # Prefer the structured JSON body (MDK format)
    if response_body:
        try:
            body = json.loads(response_body)
            macaroon = body.get("macaroon")
            invoice = body.get("invoice")
            payment_hash = body.get("paymentHash")
            amount_sats = body.get("amountSats")
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to WWW-Authenticate header parsing
    if not macaroon or not invoice:
        mac_match = re.search(r'macaroon="([^"]+)"', www_authenticate)
        inv_match = re.search(r'invoice="([^"]+)"', www_authenticate)

        if not mac_match or not inv_match:
            raise L402Error(
                f"Could not parse L402 challenge from body or header: "
                f"{www_authenticate[:200]}"
            )

        macaroon = macaroon or mac_match.group(1)
        invoice = invoice or inv_match.group(1)

    # If we still don't have a payment_hash, extract from invoice
    if not payment_hash:
        payment_hash = _extract_payment_hash_from_invoice(invoice)

    # If we still don't have amount, try bolt11 parsing
    if amount_sats is None:
        amount_sats = _extract_amount_sats(invoice)

    return L402Challenge(
        macaroon=macaroon,
        invoice=invoice,
        payment_hash=payment_hash,
        amount_sats=amount_sats,
    )


def _extract_payment_hash_from_invoice(invoice: str) -> str:
    """Derive a unique identifier from a bolt11 invoice string.

    A proper bolt11 decoder would extract the real payment hash from the
    tagged fields. For now we use a SHA-256 of the invoice as a stand-in
    for deduplication purposes.
    """
    import hashlib

    return hashlib.sha256(invoice.encode()).hexdigest()


def _extract_amount_sats(invoice: str) -> int | None:
    """Best-effort amount extraction from a bolt11 invoice.

    Bolt11 invoices encode the amount in the human-readable part:
      lnbc10u...  = 10 micro-BTC = 1000 sats
      lnbc1500n... = 1500 nano-BTC = 150 sats

    Uses integer arithmetic to avoid floating-point precision issues.
    """
    match = re.match(r"ln(?:bc|tb|tbs)(\d+)([munp])?", invoice.lower())
    if not match:
        return None

    amount = int(match.group(1))
    multiplier = match.group(2)

    # Sats per unit — integer multipliers to avoid float math
    sats_per_unit: dict[str | None, tuple[int, int]] = {
        "m": (100_000, 1),      # amount * 100_000
        "u": (100, 1),          # amount * 100
        "n": (1, 10),           # amount / 10
        "p": (1, 10_000),       # amount / 10_000
        None: (100_000_000, 1), # whole BTC
    }

    numerator, denominator = sats_per_unit.get(multiplier, (100_000_000, 1))
    return (amount * numerator) // denominator


@dataclass
class PaymentResult:
    """Result of paying a Lightning invoice."""

    preimage: str
    payment_hash: str


async def pay_invoice(invoice: str) -> PaymentResult:
    """Pay a Lightning invoice.

    In demo mode, simulates a successful payment instantly.
    """
    if DEMO_MODE:
        import hashlib, secrets
        preimage = secrets.token_hex(32)
        payment_hash = hashlib.sha256(preimage.encode()).hexdigest()
        return PaymentResult(preimage=preimage, payment_hash=payment_hash)

    proc = await asyncio.create_subprocess_exec(
        *MDK_CMD,
        "send",
        invoice,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except asyncio.TimeoutError:
        proc.kill()
        raise L402Error("mdk send timed out after 60s")

    if proc.returncode != 0:
        raise L402Error(
            f"mdk send failed (exit {proc.returncode}): {stderr.decode().strip()}"
        )

    raw = stdout.decode().strip()
    if not raw:
        raise L402Error("mdk send returned empty output")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise L402Error(f"mdk send returned non-JSON output: {raw[:200]}")

    # Extract preimage (proof of payment) — try common field names
    preimage = (
        result.get("preimage")
        or result.get("payment_preimage")
        or result.get("proof")
    )
    payment_hash = result.get("payment_hash", "")

    if not preimage:
        raise L402Error(
            f"mdk send response missing preimage — cannot construct L402 "
            f"authorization. Got keys: {list(result.keys())}"
        )

    return PaymentResult(preimage=preimage, payment_hash=payment_hash)


async def get_wallet_balance() -> int:
    """Get the wallet balance in sats.

    In demo mode, returns a simulated balance minus any completed transactions today.
    """
    if DEMO_MODE:
        from backend.db import get_db
        db = await get_db()
        spent_row = await db.execute_fetchall(
            "SELECT COALESCE(SUM(amount_sats),0) FROM transactions WHERE status='completed'"
        )
        funded_row = await db.execute_fetchall(
            "SELECT COALESCE(value,'0') FROM settings WHERE key='demo_funded_sats'"
        )
        spent = spent_row[0][0] if spent_row else 0
        funded = int(funded_row[0][0]) if funded_row else 0
        return max(0, DEMO_BALANCE_SATS + funded - spent)

    proc = await asyncio.create_subprocess_exec(
        *MDK_CMD,
        "balance",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        raise L402Error("mdk balance timed out after 10s — is the wallet configured?")

    if proc.returncode != 0:
        raise L402Error(
            f"mdk balance failed (exit {proc.returncode}): {stderr.decode().strip()}"
        )

    try:
        result = json.loads(stdout.decode().strip())
    except json.JSONDecodeError:
        return 0

    return result.get("balance_sats", 0)


def _demo_invoice(amount_sats: int | None) -> dict:
    """Generate a realistic-looking mainnet Lightning invoice for demo mode."""
    import hashlib
    import secrets
    from datetime import datetime, timedelta, timezone

    token = secrets.token_hex(16)
    payment_hash = hashlib.sha256(token.encode()).hexdigest()

    # Build a realistic lnbc invoice string (not payable, but looks real)
    amt = amount_sats or 0
    if amt > 0:
        # Encode amount in millisats as micro-BTC (u suffix)
        micro_btc = (amt * 100_000) // 100  # amt sats → micro-BTC units
        amt_str = f"{micro_btc}u"
    else:
        amt_str = ""

    rand_body = secrets.token_hex(100)
    invoice = f"lnbc{amt_str}1{rand_body}"

    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    return {"invoice": invoice, "payment_hash": payment_hash, "expires_at": expires_at}


async def create_receive_invoice(amount_sats: int | None = None) -> dict:
    """Generate a Lightning invoice to receive funds.

    In demo mode, returns a realistic-looking (but non-payable) mainnet invoice
    and auto-credits the wallet after a short delay.
    """
    if DEMO_MODE:
        result = _demo_invoice(amount_sats)
        # Auto-credit the demo balance by reducing total spent (no-op since
        # balance = DEMO_BALANCE_SATS - spent; funding just raises the ceiling)
        # Store the funded amount so balance goes up
        if amount_sats:
            from backend.db import get_db
            db = await get_db()
            await db.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)",
                ("demo_funded_sats", "0"),
            )
            await db.execute(
                "UPDATE settings SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT) WHERE key = ?",
                (amount_sats, "demo_funded_sats"),
            )
            await db.commit()
        return result

    cmd = [*MDK_CMD, "receive"]
    if amount_sats is not None:
        cmd.append(str(amount_sats))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    except asyncio.TimeoutError:
        proc.kill()
        raise L402Error("mdk receive timed out after 15s")

    if proc.returncode != 0:
        raise L402Error(
            f"mdk receive failed (exit {proc.returncode}): {stderr.decode().strip()}"
        )

    raw = stdout.decode().strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise L402Error(f"mdk receive returned non-JSON output: {raw[:200]}")


@dataclass
class L402Result:
    """Result of an L402 fetch — either the resource body or an error description."""

    success: bool
    body: str
    amount_sats: int
    vendor: str
    status_code: int


def _validate_url(url: str) -> None:
    """Reject URLs that point to private/internal networks (SSRF protection)."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise UnsafeURL(f"Unsupported scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURL("No hostname in URL")

    # Resolve hostname and check all addresses
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise UnsafeURL(f"Cannot resolve hostname: {hostname}")

    for family, _, _, _, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise UnsafeURL(
                f"URL resolves to non-public address {ip} — "
                f"refusing to fetch (SSRF protection)"
            )


async def fetch_with_l402(
    url: str,
    *,
    method: str = "GET",
    json_body: dict | None = None,
) -> L402Result:
    """Fetch a URL, handling L402 payment challenges transparently.

    Flow:
    1. Send initial request
    2. If 402 → parse challenge, check budget, check dedup, pay invoice
    3. Retry request with Authorization: L402 <macaroon>:<preimage>
    4. Return the final response body
    """
    vendor = urlparse(url).netloc

    # SSRF protection — reject private/internal URLs
    _validate_url(url)

    # Blocklist check
    if vendor in VENDOR_BLOCKLIST:
        raise VendorBlocked(f"Vendor {vendor} is on the blocklist — skipping")

    req_kwargs: dict = {}
    if json_body is not None:
        req_kwargs["json"] = json_body

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        # Initial request
        resp = await client.request(method, url, **req_kwargs)

        # Not a 402 — return directly
        if resp.status_code != 402:
            return L402Result(
                success=resp.status_code < 400,
                body=resp.text,
                amount_sats=0,
                vendor=vendor,
                status_code=resp.status_code,
            )

        # Parse L402 challenge from body (MDK format) or header
        www_auth = resp.headers.get("www-authenticate", "")
        body_text = resp.text

        if not www_auth and not body_text:
            raise L402Error("Got 402 but no WWW-Authenticate header or body")

        challenge = parse_l402_challenge(www_auth, body_text)

        # Dedup: refuse to pay the same invoice twice
        if await payment_hash_exists(challenge.payment_hash):
            raise DuplicatePayment(
                f"Invoice already attempted (payment_hash={challenge.payment_hash[:16]}...)"
            )

        amount = challenge.amount_sats
        if amount is None:
            raise L402Error(
                "Cannot determine invoice amount — refusing to pay an invoice "
                "of unknown value. A bolt11 decoder is needed for this invoice format."
            )

        # Per-transaction limit check
        per_tx = await get_per_tx_limit()
        if amount > per_tx:
            raise L402Error(
                f"Purchase too large — {amount:,} sats exceeds your single-purchase "
                f"limit of {per_tx:,} sats. Raise the per-purchase cap in wallet settings."
            )

        # Atomically check budget and reserve the payment in one transaction
        daily_limit = await get_daily_limit()
        try:
            await check_budget_and_reserve(
                daily_limit=daily_limit,
                vendor=vendor,
                url=url,
                amount_sats=amount,
                payment_hash=challenge.payment_hash,
                description=f"L402 payment to {vendor}",
            )
        except ValueError as e:
            raise L402Error(str(e)) from e

        # Pay the invoice
        try:
            payment = await pay_invoice(challenge.invoice)
        except L402Error:
            await update_transaction_status(challenge.payment_hash, "failed")
            raise

        # Mark as completed
        await update_transaction_status(challenge.payment_hash, "completed")

        # Retry with L402 authorization: L402 <macaroon>:<preimage>
        auth_header = f"L402 {challenge.macaroon}:{payment.preimage}"
        resp2 = await client.request(
            method, url, headers={"Authorization": auth_header}, **req_kwargs
        )

        return L402Result(
            success=resp2.status_code < 400,
            body=resp2.text,
            amount_sats=amount,
            vendor=vendor,
            status_code=resp2.status_code,
        )
