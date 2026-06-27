"""Request and response models for the chat API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SourceCitation:
    source: str
    modality: str
    chunk_index: int
    preview: str
    storage_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "modality": self.modality,
            "chunk_index": self.chunk_index,
            "preview": self.preview,
            "storage_path": self.storage_path,
        }


@dataclass
class ToolActivity:
    tool: str
    summary: str
    input: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "summary": self.summary,
            "input": self.input,
        }
