"""Request and response models for the chat API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IngestResult:
    chunks: int
    skipped: bool
    source: str


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
    relevance_score: float = 0.0
    citation_index: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    total_pages: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source": self.source,
            "modality": self.modality,
            "chunk_index": self.chunk_index,
            "preview": self.preview,
            "storage_path": self.storage_path,
            "relevance_score": self.relevance_score,
        }
        if self.citation_index is not None:
            payload["citation_index"] = self.citation_index
        if self.page_start is not None:
            payload["page_start"] = self.page_start
        if self.page_end is not None:
            payload["page_end"] = self.page_end
        if self.total_pages is not None:
            payload["total_pages"] = self.total_pages
        return payload


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
