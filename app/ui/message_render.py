"""Render assistant metadata: tool transparency and source citations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

TOOL_ICONS = {
    "save_memory": "💾",
    "recall_memory": "🧠",
    "retrieve_domain": "📚",
}


def render_tool_transparency(tools_used: List[Dict[str, Any]]) -> None:
    """Show which agent tools ran for an assistant turn."""
    if not tools_used:
        return

    with st.expander(f"Agent tools ({len(tools_used)})", expanded=False):
        for item in tools_used:
            icon = TOOL_ICONS.get(item.get("tool", ""), "🔧")
            tool_name = item.get("tool", "unknown")
            summary = item.get("summary", "")
            st.markdown(f"{icon} **{tool_name}** — {summary}")


def render_citations(sources: List[Dict[str, Any]]) -> None:
    """Show retrieved source documents with previews."""
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            filename = source.get("source", "unknown")
            modality = source.get("modality", "text")
            chunk_index = source.get("chunk_index", 0)
            preview = source.get("preview", "")

            st.markdown(
                f"**[{index}] {filename}** · `{modality}` · chunk {chunk_index}"
            )
            if preview:
                st.caption(preview)

            storage_path = source.get("storage_path")
            if modality == "image" and storage_path and Path(storage_path).exists():
                st.image(storage_path, caption=filename, width=320)
