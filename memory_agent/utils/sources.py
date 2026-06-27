"""Format and consolidate RAG source citations for display."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SOURCE_PREVIEW_MAX = 140
MAX_DISPLAY_DOCUMENTS = 4
MAX_CHUNKS_PER_DOCUMENT = 1

_TRAILING_LIST_ARTIFACT = re.compile(r"\s*['\"]?\]\s*$")
_INLINE_SOURCES_BLOCK = re.compile(
    r"\n+(?:Source|Sources)\s*:\s*.+$",
    re.IGNORECASE | re.DOTALL,
)


def truncate_preview(text: str, max_len: int = SOURCE_PREVIEW_MAX) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def display_filename(source: str) -> str:
    """Short label for long upload paths."""
    name = Path(source).name or source
    if len(name) <= 72:
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    keep = 68 - len(suffix)
    return f"{stem[:keep]}…{suffix}"


def build_source_preview(metadata: dict[str, Any], page_content: str = "") -> str:
    """One-line preview for the Sources panel — not full chunk text."""
    modality = str(metadata.get("modality", "text"))
    if modality == "pdf":
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        total_pages = metadata.get("total_pages")
        if page_start and page_end:
            label = f"Pages {page_start}–{page_end}"
            if total_pages:
                label += f" of {total_pages}"
            return label

    preview = str(metadata.get("text_preview") or "").strip()
    if preview and not preview.lower().startswith("pdf pages"):
        return truncate_preview(preview)

    if page_content and modality != "pdf":
        return truncate_preview(page_content)

    note = metadata.get("note")
    if note:
        return truncate_preview(str(note))

    return "Referenced section"


def consolidate_sources_for_display(
    sources: list[dict[str, Any]],
    *,
    max_documents: int = MAX_DISPLAY_DOCUMENTS,
    max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
) -> list[dict[str, Any]]:
    """Keep the strongest, diverse citations — one entry per document by default."""
    if not sources:
        return []

    ranked = sorted(
        sources,
        key=lambda item: float(item.get("relevance_score", 0.0)),
        reverse=True,
    )

    consolidated: list[dict[str, Any]] = []
    per_document: dict[str, int] = {}

    for source in ranked:
        doc_key = str(source.get("source", "unknown"))
        count = per_document.get(doc_key, 0)
        if count >= max_chunks_per_document:
            continue

        per_document[doc_key] = count + 1
        consolidated.append(source)

        if len(per_document) >= max_documents:
            break

    return consolidated


def polish_assistant_text(text: str) -> str:
    """Remove parser artifacts and duplicate inline source dumps."""
    if not text:
        return ""

    cleaned = _INLINE_SOURCES_BLOCK.sub("", text.strip())
    cleaned = _TRAILING_LIST_ARTIFACT.sub("", cleaned).strip()
    return cleaned
