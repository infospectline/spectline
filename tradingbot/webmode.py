"""Web-mode authentication and customer connection helpers for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from client import ClientIdentity, identity_from_verified_claims


# ============================================================================
# CONFIGURATION
# ============================================================================

WEB_MODE = os.getenv("WEB_MODE", "false").strip().lower() == "true"

TICKET_SECRET_ENV = "TRADINGBOT_SHARED_SECRET"
TICKET_AUDIENCE = "spectline-tradingbot"
TICKET_ISSUER_ENV = "TRADINGBOT_TICKET_ISSUER"

MAX_TICKET_LENGTH = 8_192
CLOCK_SKEW_SECONDS = 5

_BASE64URL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


# ============================================================================
# AUTHENTICATION ERRORS AND CLAIMS
# ============================================================================

class AuthenticationError(Exception):
    """Raised when a TradingBot authentication ticket is invalid."""


@dataclass(frozen=True)
class VerifiedTradingBotClaims:
    """Identity claims extracted from a successfully verified ticket."""

    user_id: str
    strategy_id: str
    tradingbot_id: str
    expires_at: int
    issued_at: Optional[int] = None
    not_before: Optional[int] = None
    token_id: Optional[str] = None
    issuer: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return the trusted identity claims expected by client.py."""
        return {
            "sub": self.user_id,
            "strategy_id": self.strategy_id,
            "tradingbot_id": self.tradingbot_id,
            "exp": self.expires_at,
            "iat": self.issued_at,
            "nbf": self.not_before,
            "jti": self.token_id,
            "iss": self.issuer,
        }

    def identity_for_market(self, market: str) -> ClientIdentity:
        """Build a validated client identity for the requested market."""
        return identity_from_verified_claims(
            self.as_dict(),
            market=market,
        )


# ============================================================================
# TOKEN DECODING
# ============================================================================

def _decode_base64url(value: str, field_name: str) -> bytes:
    """Decode canonical URL-safe Base64 without accepting invalid characters."""
    if not isinstance(value, str) or not value:
        raise AuthenticationError(f"Invalid {field_name} encoding.")

    if not _BASE64URL_PATTERN.fullmatch(value):
        raise AuthenticationError(f"Invalid {field_name} encoding.")

    padded_value = value + ("=" * (-len(value) % 4))

    try:
        decoded = base64.b64decode(
            padded_value.encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error) as error:
        raise AuthenticationError(
            f"Invalid {field_name} encoding."
        ) from error

    canonical_value = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if not hmac.compare_digest(canonical_value, value):
        raise AuthenticationError(f"Invalid {field_name} encoding.")

    return decoded


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
    """Reject duplicate JSON keys to avoid ambiguous ticket claims."""
    result: Dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise AuthenticationError("Duplicate ticket claim.")
        result[key] = value

    return result


def _read_integer_claim(
    payload: Mapping[str, Any],
    claim_name: str,
    *,
    required: bool = False,
) -> Optional[int]:
    """Read an integer timestamp claim without accepting booleans."""
    value = payload.get(claim_name)

    if value is None:
        if required:
            raise AuthenticationError(
                f"Missing {claim_name} claim."
            )
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthenticationError(
            f"Invalid {claim_name} claim."
        )

    if not float(value).is_integer():
        raise AuthenticationError(
            f"Invalid {claim_name} claim."
        )

    return int(value)


def _required_string_claim(
    payload: Mapping[str, Any],
    claim_name: str,
) -> str:
    """Read a required, non-empty string identity claim."""
    value = payload.get(claim_name)

    if not isinstance(value, str) or not value.strip():
        raise AuthenticationError(
            f"Missing or invalid {claim_name} claim."
        )

    return value.strip()


# ============================================================================
# TICKET VERIFICATION
# ============================================================================

def verify_tradingbot_ticket(
    ticket: str,
    secret: Optional[str] = None,
    *,
    now: Optional[int] = None,
) -> VerifiedTradingBotClaims:
    """Verify a signed Spectline TradingBot ticket and return trusted claims.

    Ticket format:
        BASE64URL(JSON payload).BASE64URL(HMAC-SHA256 signature)

    Required payload claims:
        aud, sub, strategy_id, tradingbot_id, exp

    Optional payload claims:
        iat, nbf, jti, iss
    """
    if not isinstance(ticket, str) or not ticket:
        raise AuthenticationError("WebSocket ticket is missing.")

    if len(ticket) > MAX_TICKET_LENGTH:
        raise AuthenticationError("WebSocket ticket is too large.")

    shared_secret = secret or os.getenv(TICKET_SECRET_ENV)
    if not shared_secret:
        raise AuthenticationError("TradingBot ticket secret is not configured.")

    parts = ticket.split(".")
    if len(parts) != 2:
        raise AuthenticationError("Invalid WebSocket ticket format.")

    payload_part, signature_part = parts

    payload_bytes = _decode_base64url(payload_part, "payload")
    supplied_signature = _decode_base64url(signature_part, "signature")

    expected_signature = hmac.new(
        shared_secret.encode("utf-8"),
        payload_part.encode("ascii"),
        hashlib.sha256,
    ).digest()

    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise AuthenticationError("WebSocket ticket signature is invalid.")

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except AuthenticationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthenticationError("Invalid WebSocket ticket payload.") from error

    if not isinstance(payload, dict):
        raise AuthenticationError("Invalid WebSocket ticket payload.")

    if payload.get("aud") != TICKET_AUDIENCE:
        raise AuthenticationError("WebSocket ticket audience is invalid.")

    configured_issuer = os.getenv(TICKET_ISSUER_ENV)
    issuer = payload.get("iss")

    if configured_issuer:
        if not isinstance(issuer, str) or issuer != configured_issuer:
            raise AuthenticationError("WebSocket ticket issuer is invalid.")
    elif issuer is not None and not isinstance(issuer, str):
        raise AuthenticationError("Invalid iss claim.")

    user_id = _required_string_claim(payload, "sub")
    strategy_id = _required_string_claim(payload, "strategy_id")
    tradingbot_id = _required_string_claim(payload, "tradingbot_id")

    expires_at = _read_integer_claim(
        payload,
        "exp",
        required=True,
    )
    issued_at = _read_integer_claim(payload, "iat")
    not_before = _read_integer_claim(payload, "nbf")

    current_time = int(time.time()) if now is None else int(now)

    if expires_at is None or expires_at <= current_time - CLOCK_SKEW_SECONDS:
        raise AuthenticationError("WebSocket ticket has expired.")

    if not_before is not None and not_before > current_time + CLOCK_SKEW_SECONDS:
        raise AuthenticationError("WebSocket ticket is not active yet.")

    if issued_at is not None and issued_at > current_time + CLOCK_SKEW_SECONDS:
        raise AuthenticationError("WebSocket ticket issue time is invalid.")

    token_id = payload.get("jti")
    if token_id is not None and (
        not isinstance(token_id, str) or not token_id.strip()
    ):
        raise AuthenticationError("Invalid jti claim.")

    claims = VerifiedTradingBotClaims(
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        expires_at=expires_at,
        issued_at=issued_at,
        not_before=not_before,
        token_id=token_id.strip() if isinstance(token_id, str) else None,
        issuer=issuer if isinstance(issuer, str) else None,
    )

    # Validate the identifier formats using the same rules as client.py.
    # The placeholder market is replaced when identity_for_market() is called.
    claims.identity_for_market("EURUSD")

    return claims


# ============================================================================
# WEBSOCKET AUTHENTICATION
# ============================================================================

async def authenticate_websocket(
    websocket: Any,
    secret: Optional[str] = None,
) -> Optional[VerifiedTradingBotClaims]:
    """Authenticate a WebSocket before accepting it in WEB_MODE.

    Local mode does not require a ticket and returns None.
    """
    if not WEB_MODE:
        return None

    ticket = websocket.query_params.get("ticket", "")

    return verify_tradingbot_ticket(
        ticket,
        secret=secret,
    )


async def reject_websocket(
    websocket: Any,
    code: int = 1008,
    reason: str = "Authentication failed.",
) -> None:
    """Close a WebSocket that failed authentication or authorization."""
    try:
        await websocket.close(
            code=code,
            reason=reason,
        )
    except Exception:
        pass


# ============================================================================
# CUSTOMER CONNECTION MANAGEMENT
# ============================================================================

async def accept_customer_connection(
    websocket: Any,
    runtime: Any,
) -> None:
    """Accept a WebSocket and register it with its TradingBot runtime."""
    if runtime is None:
        raise ValueError("A TradingBot runtime is required.")

    if not callable(getattr(runtime, "add_websocket_client", None)):
        raise TypeError("runtime does not support WebSocket registration.")

    await websocket.accept()
    runtime.add_websocket_client(websocket)


def disconnect_customer_connection(
    websocket: Any,
    runtime: Any,
) -> None:
    """Remove a WebSocket from its TradingBot runtime."""
    if runtime is None:
        return

    remove_client = getattr(runtime, "remove_websocket_client", None)
    if callable(remove_client):
        remove_client(websocket)


# ============================================================================
# PUBLIC HELPERS
# ============================================================================

def is_web_mode() -> bool:
    """Return whether the application is running in WEB_MODE."""
    return WEB_MODE


def ticket_identity_for_market(
    claims: VerifiedTradingBotClaims,
    market: str,
) -> ClientIdentity:
    """Build the complete validated identity for one selected market."""
    if not isinstance(claims, VerifiedTradingBotClaims):
        raise TypeError("claims must come from verify_tradingbot_ticket().")

    return claims.identity_for_market(market)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "WEB_MODE",
    "TICKET_AUDIENCE",
    "AuthenticationError",
    "VerifiedTradingBotClaims",
    "verify_tradingbot_ticket",
    "authenticate_websocket",
    "reject_websocket",
    "accept_customer_connection",
    "disconnect_customer_connection",
    "is_web_mode",
    "ticket_identity_for_market",
]
