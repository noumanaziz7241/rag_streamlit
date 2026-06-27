"""Shared google-genai client factory."""

from __future__ import annotations

from functools import lru_cache

from google import genai

from memory_agent.google.credentials import GoogleAuthConfig, resolve_google_auth


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    """Return a cached google-genai client using API key or Vertex credentials."""
    auth = resolve_google_auth()
    return _build_client(auth)


def _build_client(auth: GoogleAuthConfig) -> genai.Client:
    if auth.mode == "api_key":
        if not auth.api_key:
            raise ValueError("Google API key is missing.")
        return genai.Client(api_key=auth.api_key)

    if not auth.project_id:
        raise ValueError(
            "Google Vertex project_id is missing. "
            "Ensure your credentials JSON includes project_id or run "
            "`python scripts/google_auth.py` to create google_token.json."
        )

    return genai.Client(
        vertexai=True,
        project=auth.project_id,
        location=auth.location,
        credentials=auth.credentials,
    )


def clear_genai_client_cache() -> None:
    get_genai_client.cache_clear()
