from memory_agent.google.streamlit_secrets import (
    _normalize_secret_value,
    load_google_credentials_from_streamlit,
    load_google_token_from_streamlit,
    read_streamlit_json_string,
)


def test_read_streamlit_json_string_parses_dict(monkeypatch):
    class FakeSecrets(dict):
        pass

    payload = '{"type": "service_account", "project_id": "demo"}'
    monkeypatch.setattr(
        "memory_agent.google.streamlit_secrets._read_streamlit_root",
        lambda: FakeSecrets(GOOGLE_CREDENTIALS_JSON=payload),
    )
    parsed = read_streamlit_json_string("GOOGLE_CREDENTIALS_JSON")
    assert parsed == {"type": "service_account", "project_id": "demo"}


def test_load_google_credentials_returns_none_without_streamlit(monkeypatch):
    monkeypatch.setattr(
        "memory_agent.google.streamlit_secrets._read_streamlit_root",
        lambda: None,
    )
    assert load_google_credentials_from_streamlit() is None
    assert load_google_token_from_streamlit() is None


def test_normalize_secret_value_flattens_nested_dict():
    nested = {"web": {"project_id": "demo", "client_id": "abc"}}
    assert _normalize_secret_value(nested) == nested
