"""Backward-compatible entry point for Streamlit."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.bootstrap import bootstrap

bootstrap()

from app.main import main

if __name__ == "__main__":
    main()
