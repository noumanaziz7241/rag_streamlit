"""Inspect configuration for setup UI — never exposes secret values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from memory_agent.config import CONFIG_ALIASES, read_secret_optional
from memory_agent.google.credentials import get_google_auth_error, resolve_google_auth
from memory_agent.google.streamlit_secrets import (
    load_google_credentials_from_streamlit,
    load_google_token_from_streamlit,
)

KeyStatus = Literal["ok", "missing", "empty"]


@dataclass
class ConfigDiagnostics:
    secrets_available: bool
    secrets_load_error: str | None
    keys: dict[str, KeyStatus]
    google_auth_ok: bool
    google_auth_mode: str | None
    google_error: str | None
    hints: list[str] = field(default_factory=list)


def _streamlit_secrets_status() -> tuple[bool, str | None]:
    try:
        import streamlit as st

        if not hasattr(st, "secrets"):
            return False, None
        # Force load — invalid TOML raises here on Streamlit Cloud.
        _ = st.secrets.keys()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _key_status(*candidates: str) -> KeyStatus:
    for name in candidates:
        value = read_secret_optional(name)
        if value is None:
            continue
        if str(value).strip():
            return "ok"
        return "empty"
    return "missing"


def get_config_diagnostics() -> ConfigDiagnostics:
    """Summarize which required settings are present (values are never returned)."""
    secrets_available, secrets_load_error = _streamlit_secrets_status()
    hints: list[str] = []

    keys = {
        "GEMINI_API_KEY": _key_status("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "PINECONE_API_KEY": _key_status("PINECONE_API_KEY"),
        "PINECONE_INDEX_NAME": _key_status("PINECONE_INDEX_NAME"),
        "PINECONE_MEMORY_INDEX_NAME": _key_status("PINECONE_MEMORY_INDEX_NAME"),
    }

    google_auth_ok = False
    google_auth_mode: str | None = None
    google_error = get_google_auth_error()

    if google_error is None:
        try:
            auth = resolve_google_auth()
            google_auth_ok = True
            google_auth_mode = auth.mode
        except ValueError as exc:
            google_error = str(exc)

    if secrets_load_error:
        hints.append(
            "Streamlit could not parse your Secrets TOML. Fix syntax errors in "
            "**App settings → Secrets**, then reboot the app. Use `key = \"value\"` "
            "format — not JSON."
        )

    if keys["GEMINI_API_KEY"] == "empty":
        hints.append("`GEMINI_API_KEY` is present but empty. Paste a real key from Google AI Studio.")

    if keys["GEMINI_API_KEY"] == "missing":
        creds = load_google_credentials_from_streamlit()
        token = load_google_token_from_streamlit()
        if creds and not token:
            hints.append(
                "Found `[google_client_secret]` in secrets but no `[google_token]`. "
                "Either add a `[google_token]` table OR remove OAuth sections and set "
                "only `GEMINI_API_KEY` (recommended)."
            )
        elif creds and creds.get("type") == "service_account":
            private_key = creds.get("private_key") or ""
            if not str(private_key).strip():
                hints.append(
                    "`[google_service_account]` is incomplete — fill `private_key` and "
                    "`client_email`, or switch to `GEMINI_API_KEY` only."
                )

    if any(keys[name] == "missing" for name in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_MEMORY_INDEX_NAME")):
        hints.append("All three Pinecone keys are required in Streamlit secrets.")

    if not hints and not google_auth_ok:
        hints.append(
            "Set `GEMINI_API_KEY` in Streamlit secrets (simplest), then click "
            "**Manage app → Reboot app**."
        )

    return ConfigDiagnostics(
        secrets_available=secrets_available,
        secrets_load_error=secrets_load_error,
        keys=keys,
        google_auth_ok=google_auth_ok,
        google_auth_mode=google_auth_mode,
        google_error=google_error,
        hints=hints,
    )


def format_key_status(status: KeyStatus) -> str:
    return {"ok": "✅ set", "missing": "❌ missing", "empty": "⚠️ empty"}[status]
