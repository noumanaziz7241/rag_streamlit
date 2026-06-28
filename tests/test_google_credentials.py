import json
from pathlib import Path

from memory_agent.google.credentials import (
    DEFAULT_CREDENTIALS_FILENAME,
    get_credentials_path,
    resolve_google_auth,
)


def test_get_credentials_path_finds_default_file(tmp_path, monkeypatch):
    creds_file = tmp_path / DEFAULT_CREDENTIALS_FILENAME
    creds_file.write_text(
        json.dumps({"web": {"project_id": "demo-project"}, "project_id": "demo-project"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_CREDENTIALS_PATH", str(creds_file))
    assert get_credentials_path() == creds_file


def test_resolve_api_key_when_no_credentials_file(monkeypatch):
    monkeypatch.setattr(
        "memory_agent.google.credentials.get_credentials_path",
        lambda: None,
    )
    monkeypatch.setattr(
        "memory_agent.google.credentials._load_api_key",
        lambda: "test-key",
    )
    auth = resolve_google_auth()
    assert auth.mode == "api_key"
    assert auth.api_key == "test-key"


def test_resolve_api_key_when_oauth_client_missing_token(tmp_path, monkeypatch):
    creds_file = tmp_path / DEFAULT_CREDENTIALS_FILENAME
    creds_file.write_text(
        json.dumps({"web": {"project_id": "demo-project"}, "project_id": "demo-project"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_CREDENTIALS_PATH", str(creds_file))
    monkeypatch.setattr(
        "memory_agent.google.credentials._load_api_key",
        lambda: "fallback-key",
    )
    auth = resolve_google_auth()
    assert auth.mode == "api_key"
    assert auth.api_key == "fallback-key"
