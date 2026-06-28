"""Google authentication helpers for Gemini API and Vertex AI."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from memory_agent.config import PROJECT_ROOT
from memory_agent.google.streamlit_secrets import (
    load_google_credentials_from_streamlit,
    load_google_token_from_streamlit,
)

GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_CREDENTIALS_FILENAME = "google_client_secret.json"
DEFAULT_TOKEN_FILENAME = "google_token.json"

GoogleAuthMode = Literal["api_key", "vertex_service_account", "vertex_oauth", "vertex_adc"]


@dataclass(frozen=True)
class GoogleAuthConfig:
    mode: GoogleAuthMode
    project_id: Optional[str] = None
    location: str = DEFAULT_VERTEX_LOCATION
    api_key: Optional[str] = None
    credentials: object | None = None


def _read_streamlit_secret(key: str) -> str | None:
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            value = st.secrets[key]
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass
    return None


def _optional_env(*keys: str) -> str | None:
    for key in keys:
        secret = _read_streamlit_secret(key)
        if secret:
            return secret
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def get_credentials_path() -> Path | None:
    """Resolve path to Google credentials JSON on disk."""
    candidates = [
        _optional_env("GOOGLE_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"),
        str(PROJECT_ROOT / DEFAULT_CREDENTIALS_FILENAME),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def get_token_path() -> Path | None:
    """Resolve path to OAuth token JSON on disk."""
    custom = _optional_env("GOOGLE_TOKEN_PATH")
    if custom and Path(custom).is_file():
        return Path(custom)

    default = PROJECT_ROOT / DEFAULT_TOKEN_FILENAME
    if default.is_file():
        return default
    return None


def _load_api_key() -> str | None:
    return _optional_env("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _load_credentials_dict() -> dict[str, Any] | None:
    """Load credentials JSON from Streamlit secrets or a local file."""
    streamlit_payload = load_google_credentials_from_streamlit()
    if streamlit_payload:
        return streamlit_payload

    credentials_path = get_credentials_path()
    if credentials_path is not None:
        return json.loads(credentials_path.read_text(encoding="utf-8"))
    return None


def _load_token_dict() -> dict[str, Any] | None:
    """Load OAuth token JSON from Streamlit secrets or a local file."""
    streamlit_token = load_google_token_from_streamlit()
    if streamlit_token:
        return streamlit_token

    token_path = get_token_path()
    if token_path is not None:
        return json.loads(token_path.read_text(encoding="utf-8"))
    return None


def _credentials_from_token_dict(
    token_data: dict[str, Any],
    project_id: str | None,
) -> GoogleAuthConfig:
    from google.oauth2.credentials import Credentials

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(token_data, handle)
        token_file = handle.name

    credentials = Credentials.from_authorized_user_file(
        token_file,
        scopes=[GOOGLE_CLOUD_SCOPE],
    )
    Path(token_file).unlink(missing_ok=True)

    return GoogleAuthConfig(
        mode="vertex_oauth",
        project_id=project_id or token_data.get("project_id"),
        location=_optional_env("GOOGLE_VERTEX_LOCATION") or DEFAULT_VERTEX_LOCATION,
        credentials=credentials,
    )


def _load_credentials_from_data(data: dict[str, Any]) -> GoogleAuthConfig:
    location = _optional_env("GOOGLE_VERTEX_LOCATION") or DEFAULT_VERTEX_LOCATION

    if data.get("type") == "service_account":
        from google.oauth2 import service_account

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(data, handle)
            credentials_file = handle.name

        credentials = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=[GOOGLE_CLOUD_SCOPE],
        )
        Path(credentials_file).unlink(missing_ok=True)

        return GoogleAuthConfig(
            mode="vertex_service_account",
            project_id=data.get("project_id"),
            location=location,
            credentials=credentials,
        )

    project_id = data.get("project_id")
    if not project_id:
        section = data.get("web") or data.get("installed") or {}
        project_id = section.get("project_id")

    token_data = _load_token_dict()
    if token_data is not None:
        return _credentials_from_token_dict(token_data, project_id)

    raise ValueError(
        "OAuth client secret found but no token is configured. "
        "For Streamlit Cloud, add a `[google_token]` table in secrets "
        "(from `python scripts/google_auth.py` locally), "
        "or set `GEMINI_API_KEY` from Google AI Studio "
        "(recommended when org policy blocks service account keys or OAuth is internal-only)."
    )


def _prefer_api_key() -> bool:
    """When true, skip Vertex/OAuth and use Google AI Studio API key."""
    value = _optional_env("GOOGLE_AUTH_MODE", "GOOGLE_USE_API_KEY")
    return value is not None and value.strip().lower() in {"api_key", "1", "true", "yes"}


def resolve_google_auth() -> GoogleAuthConfig:
    """Resolve Google auth for embeddings, multimodal, and chat models."""
    api_key = _load_api_key()

    if _prefer_api_key():
        if api_key:
            return GoogleAuthConfig(mode="api_key", api_key=api_key)
        raise ValueError(
            "GOOGLE_AUTH_MODE=api_key is set but GEMINI_API_KEY / GOOGLE_API_KEY is missing."
        )

    # Prefer API key when set — avoids broken partial OAuth blocks in Streamlit secrets.
    if api_key:
        return GoogleAuthConfig(mode="api_key", api_key=api_key)

    credentials_data = _load_credentials_dict()
    if credentials_data is not None:
        return _load_credentials_from_data(credentials_data)

    raise ValueError(
        "Google credentials not configured. Provide one of:\n"
        "  • `GEMINI_API_KEY` / `GOOGLE_API_KEY` in Streamlit secrets (recommended)\n"
        f"  • Streamlit secrets: `[google_service_account]` or `[google_client_secret]` + `[google_token]`\n"
        f"  • Local file: `{DEFAULT_CREDENTIALS_FILENAME}` + `{DEFAULT_TOKEN_FILENAME}`"
    )


def has_google_auth() -> bool:
    try:
        resolve_google_auth()
        return True
    except ValueError:
        return False


def get_google_auth_error() -> str | None:
    try:
        resolve_google_auth()
        return None
    except ValueError as exc:
        return str(exc)
