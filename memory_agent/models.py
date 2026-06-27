"""Request and response models for the chat API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatRequest:
    message: str
    user_id: str
    session_id: str


@dataclass
class ChatResponse:
    response: str
    user_id: str
    session_id: str
    success: bool
    error: Optional[str] = None
