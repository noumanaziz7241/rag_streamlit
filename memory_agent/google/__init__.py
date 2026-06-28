"""Google Gemini / Vertex authentication and clients."""

from memory_agent.google.chat_model import create_chat_model
from memory_agent.google.credentials import GoogleAuthConfig, has_google_auth, resolve_google_auth
from memory_agent.google.genai_client import get_genai_client

__all__ = [
    "GoogleAuthConfig",
    "create_chat_model",
    "get_genai_client",
    "has_google_auth",
    "resolve_google_auth",
]
