from memory_agent.config_diagnostics import format_key_status, get_config_diagnostics
from memory_agent.google.credentials import resolve_google_auth


def test_api_key_preferred_over_incomplete_oauth(monkeypatch):
    monkeypatch.setattr(
        "memory_agent.google.credentials._load_api_key",
        lambda: "test-key",
    )
    monkeypatch.setattr(
        "memory_agent.google.credentials.load_google_credentials_from_streamlit",
        lambda: {"web": {"project_id": "demo"}},
    )
    monkeypatch.setattr(
        "memory_agent.google.credentials.load_google_token_from_streamlit",
        lambda: None,
    )
    auth = resolve_google_auth()
    assert auth.mode == "api_key"
    assert auth.api_key == "test-key"


def test_diagnostics_flags_missing_gemini_key(monkeypatch):
    monkeypatch.setattr(
        "memory_agent.config_diagnostics._streamlit_secrets_status",
        lambda: (True, None),
    )
    monkeypatch.setattr(
        "memory_agent.config_diagnostics.read_secret_optional",
        lambda key: "value" if key.startswith("PINECONE") else None,
    )
    monkeypatch.setattr(
        "memory_agent.config_diagnostics.load_google_credentials_from_streamlit",
        lambda: None,
    )
    monkeypatch.setattr(
        "memory_agent.config_diagnostics.load_google_token_from_streamlit",
        lambda: None,
    )
    monkeypatch.setattr(
        "memory_agent.config_diagnostics.get_google_auth_error",
        lambda: "missing key",
    )

    diag = get_config_diagnostics()
    assert diag.keys["GEMINI_API_KEY"] == "missing"
    assert diag.hints


def test_format_key_status():
    assert format_key_status("ok") == "✅ set"
