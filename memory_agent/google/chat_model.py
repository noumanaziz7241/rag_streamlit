"""LangChain chat model factory for Gemini (DeepSeek fallback commented out)."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from memory_agent.config import get_gemini_chat_model, has_google_auth
from memory_agent.google.credentials import resolve_google_auth

# DeepSeek is kept in the codebase for future use but disabled by default.
# from langchain_deepseek import ChatDeepSeek


def create_chat_model() -> BaseChatModel:
    """Create the chat LLM — Gemini via Google credentials (default)."""
    if has_google_auth():
        auth = resolve_google_auth()
        model_name = get_gemini_chat_model()
        if auth.mode == "api_key":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=auth.api_key,
                temperature=0,
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            vertexai=True,
            project=auth.project_id,
            location=auth.location,
            credentials=auth.credentials,
            temperature=0,
        )

    # --- DeepSeek fallback (disabled — set DEEPSEEK_ENABLED=True when API key is available) ---
    # if DEEPSEEK_ENABLED and has_deepseek_api_key():
    #     from langchain_deepseek import ChatDeepSeek
    #     from memory_agent.config import get_config_value
    #
    #     return ChatDeepSeek(
    #         model="deepseek-chat",
    #         api_key=get_config_value("DEEPSEEK_API_KEY"),
    #         base_url="https://api.deepseek.com",
    #         temperature=0,
    #     )

    raise ValueError(
        "No chat LLM configured. Add Google credentials to Streamlit secrets "
        "(see docs/DEPLOY.md), place google_client_secret.json locally, or set GEMINI_API_KEY."
    )
