"""Backward-compatible entry point for Streamlit."""

from app.bootstrap import bootstrap

bootstrap()

from app.main import main

if __name__ == "__main__":
    main()
