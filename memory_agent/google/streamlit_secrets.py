"""Load Google credentials from Streamlit Cloud secrets."""

from __future__ import annotations

import json
from typing import Any


def _read_streamlit_root() -> Any | None:
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            return st.secrets
    except Exception:
        pass
    return None


def read_streamlit_table(name: str) -> dict[str, Any] | None:
    """Return a nested Streamlit secrets table as a plain dict."""
    secrets = _read_streamlit_root()
    if secrets is None or name not in secrets:
        return None

    value = secrets[name]
    if isinstance(value, dict):
        return _normalize_secret_value(dict(value))
    return None


def read_streamlit_json_string(*keys: str) -> dict[str, Any] | None:
    """Parse a JSON blob stored as a Streamlit secret string."""
    secrets = _read_streamlit_root()
    if secrets is None:
        return None

    for key in keys:
        if key not in secrets:
            continue
        raw = secrets[key]
        if raw is None:
            continue
        if isinstance(raw, dict):
            return _normalize_secret_value(dict(raw))
        text = str(raw).strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_secret_value(value: Any) -> Any:
    """Convert Streamlit SecretDict / AttrDict values to plain Python types."""
    if isinstance(value, dict):
        return {str(k): _normalize_secret_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_secret_value(item) for item in value]
    return value


def load_google_credentials_from_streamlit() -> dict[str, Any] | None:
    """Resolve Google credential JSON from Streamlit secrets."""
    service_account = read_streamlit_table("google_service_account")
    if service_account and service_account.get("type") == "service_account":
        return service_account

    for key in (
        "GOOGLE_CREDENTIALS_JSON",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "google_credentials_json",
    ):
        payload = read_streamlit_json_string(key)
        if payload:
            return payload

    client_secret = read_streamlit_table("google_client_secret")
    if client_secret:
        return client_secret

    return read_streamlit_json_string("GOOGLE_CLIENT_SECRET_JSON")


def load_google_token_from_streamlit() -> dict[str, Any] | None:
    """Resolve OAuth token JSON from Streamlit secrets."""
    token_table = read_streamlit_table("google_token")
    if token_table:
        return token_table

    return read_streamlit_json_string("GOOGLE_TOKEN_JSON", "google_token_json")
