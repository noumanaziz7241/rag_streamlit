"""Application bootstrap: environment setup and warning suppression."""

from __future__ import annotations

import logging
import os
import warnings

from dotenv import load_dotenv


def bootstrap() -> None:
    """Initialize runtime environment before importing heavy dependencies."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*pydantic.*")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("grpc").setLevel(logging.ERROR)
    load_dotenv()
