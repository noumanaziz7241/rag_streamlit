"""Streamlit chat interface."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.ui.message_render import render_citations, render_tool_transparency


def _render_message(message: Dict[str, Any]) -> None:
    """Render a single chat message with optional tool and source metadata."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_tool_transparency(message.get("tools_used", []))
            render_citations(message.get("sources", []))


def _update_tool_entry(
    tools_used: List[Dict[str, Any]],
    *,
    tool_call_id: str | None,
    tool: str,
    summary: str,
    status: str,
) -> None:
    """Insert or update a tool activity row keyed by tool_call_id when available."""
    if tool_call_id:
        for item in tools_used:
            if item.get("tool_call_id") == tool_call_id:
                item["summary"] = summary
                item["status"] = status
                return

    if status == "done":
        for item in reversed(tools_used):
            if item.get("tool") == tool and item.get("status") == "running":
                item["summary"] = summary
                item["status"] = status
                if tool_call_id:
                    item["tool_call_id"] = tool_call_id
                return

    tools_used.append({
        "tool_call_id": tool_call_id,
        "tool": tool,
        "summary": summary,
        "status": status,
    })


def _render_live_tools(tools_used: List[Dict[str, Any]]) -> str:
    return " · ".join(f"{item['tool']}: {item['summary']}" for item in tools_used)


def _stream_assistant_response(prompt: str) -> None:
    """Stream the assistant response with live tool and citation updates."""
    status_placeholder = st.empty()
    tools_placeholder = st.empty()
    response_placeholder = st.empty()
    meta_placeholder = st.container()

    full_response = ""
    tools_used: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    seen_source_keys: set[tuple[str, int]] = set()
    is_streaming_text = False

    for event in st.session_state.chat_api.chat_stream(
        message=prompt,
        user_id=st.session_state.user_id,
        session_id=st.session_state.active_session_id,
    ):
        event_type = event.get("type")

        if event_type == "status":
            if not is_streaming_text:
                status_placeholder.caption(event.get("message", ""))

        elif event_type == "tool_start":
            _update_tool_entry(
                tools_used,
                tool_call_id=event.get("tool_call_id"),
                tool=event.get("tool", "unknown"),
                summary=event.get("summary", "Running…"),
                status="running",
            )
            tools_placeholder.info(_render_live_tools(tools_used))

        elif event_type == "tool_done":
            _update_tool_entry(
                tools_used,
                tool_call_id=event.get("tool_call_id"),
                tool=event.get("tool", "unknown"),
                summary=event.get("summary", "Completed"),
                status="done",
            )
            tools_placeholder.info(_render_live_tools(tools_used))

        elif event_type == "source":
            source = event.get("source", {})
            key = (source.get("source", ""), int(source.get("chunk_index", 0)))
            if key not in seen_source_keys:
                seen_source_keys.add(key)
                sources.append(source)

        elif event_type == "token":
            is_streaming_text = True
            status_placeholder.empty()
            tools_placeholder.empty()
            full_response += event.get("content", "")
            response_placeholder.markdown(full_response + "▌")

        elif event_type == "error":
            status_placeholder.empty()
            tools_placeholder.empty()
            st.error(f"Error: {event.get('error', 'Unknown error')}")
            return

        elif event_type == "done":
            full_response = event.get("response") or full_response
            tools_used = event.get("tools_used") or tools_used
            sources = event.get("sources") or sources

            status_placeholder.empty()
            tools_placeholder.empty()
            response_placeholder.markdown(full_response)

            with meta_placeholder:
                render_tool_transparency(tools_used)
                render_citations(sources)

    st.session_state.messages = st.session_state.chat_api.get_history(
        user_id=st.session_state.user_id,
        session_id=st.session_state.active_session_id,
    )
    st.session_state.history_session_id = st.session_state.active_session_id


def render_chat() -> None:
    """Render the main chat area and handle user input."""
    active_session = st.session_state.chat_api.sessions.get_session(
        st.session_state.active_session_id
    )
    title = active_session.title if active_session else "Memory Agent Chat"

    st.title(title)
    st.caption(
        "Streaming chat with tool transparency, source citations, memory, and document retrieval"
    )

    for message in st.session_state.messages:
        _render_message(message)

    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            _stream_assistant_response(prompt)
