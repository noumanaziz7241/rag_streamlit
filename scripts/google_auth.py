#!/usr/bin/env python3
"""One-time OAuth setup for google_client_secret.json (web/installed app)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_agent.config import PROJECT_ROOT
from memory_agent.google.credentials import (
    DEFAULT_CREDENTIALS_FILENAME,
    DEFAULT_TOKEN_FILENAME,
    GOOGLE_CLOUD_SCOPE,
    get_credentials_path,
)


def main() -> None:
    credentials_path = get_credentials_path()
    if credentials_path is None:
        raise SystemExit(
            f"Missing credentials file. Place `{DEFAULT_CREDENTIALS_FILENAME}` in the project root "
            "or set GOOGLE_CREDENTIALS_PATH."
        )

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise SystemExit(
            "Install OAuth dependencies first: pip install google-auth-oauthlib"
        ) from exc

    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path),
        scopes=[GOOGLE_CLOUD_SCOPE],
    )
    credentials = flow.run_local_server(port=8080, prompt="consent")

    token_path = PROJECT_ROOT / DEFAULT_TOKEN_FILENAME
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    print(f"Saved OAuth token to {token_path}")
    print("Restart the app to use Vertex AI with your Google credentials.")
    print()
    print("For Streamlit Cloud: copy google_token.json into [google_token] in App secrets.")
    print("See docs/DEPLOY.md for the full secrets template.")


if __name__ == "__main__":
    main()
