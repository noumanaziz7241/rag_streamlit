"""Shared types for multimodal RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RagChunk:
    """A single retrievable unit indexed in the vector store."""

    chunk_id: str
    source: str
    modality: str
    mime_type: str
    text: str = ""
    media_bytes: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_media(self) -> bool:
        return self.media_bytes is not None
