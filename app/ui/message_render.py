"""Render assistant metadata: tool transparency and source citations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from memory_agent.utils.sources import (
    consolidate_sources_for_display,
    format_source_label,
)

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
    """Show retrieved documents as a compact reference list."""
    if not sources:
        return

    display_sources = consolidate_sources_for_display(sources)
    doc_count = len({item.get("source", "") for item in display_sources})

    with st.expander(f"References ({doc_count})", expanded=False):
        for fallback_index, source in enumerate(display_sources, start=1):
            ref_num = source.get("citation_index") or fallback_index
            label = format_source_label(source)
            modality = source.get("modality", "text")
            preview = source.get("preview", "")

            st.markdown(f"**[{ref_num}] {label}**")
            st.caption(modality)
            if preview:
                st.markdown(f"> {preview}")

            storage_path = source.get("storage_path")
            if modality == "image" and storage_path and Path(storage_path).exists():
                st.image(storage_path, caption=label, width=320)
