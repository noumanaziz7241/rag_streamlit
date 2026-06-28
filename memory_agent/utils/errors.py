"""User-facing error messages for API failures."""

from __future__ import annotations


def format_chat_error(error: str) -> str:
    """Turn raw API exceptions into actionable chat UI messages."""
    lowered = (error or "").lower()

    if (
        "503" in error
        or "unavailable" in lowered
        or "high demand" in lowered
        or "resource_exhausted" in lowered
    ):
        return (
            "Gemini is temporarily at capacity (503). The app retried with fallback models; "
            "please wait a minute and try again. You can also set "
            "`GEMINI_CHAT_MODEL=gemini-2.0-flash` in `.env` or Streamlit secrets."
        )

    if "429" in error or "rate limit" in lowered or "quota" in lowered:
        return (
            "Gemini rate limit reached. Please wait a moment before sending another message."
        )

    return error or "Unknown error"
