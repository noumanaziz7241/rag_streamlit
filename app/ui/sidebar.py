"""Streamlit sidebar: sessions and document ingestion."""

from __future__ import annotations

import streamlit as st

from memory_agent.config import NAMESPACE


def render_sidebar() -> None:
    """Render session controls and document ingestion."""
    with st.sidebar:
        st.title("Sessions")

        sessions = st.session_state.chat_api.list_sessions(st.session_state.user_id)
        session_options = {s.session_id: s.title for s in sessions}

        if st.session_state.active_session_id not in session_options:
            new_session = st.session_state.chat_api.create_session(
                user_id=st.session_state.user_id,
                title="New chat",
            )
            st.session_state.active_session_id = new_session.session_id
            session_options[new_session.session_id] = new_session.title

        selected_title = st.selectbox(
            "Conversation",
            options=list(session_options.keys()),
            format_func=lambda sid: session_options[sid],
            index=list(session_options.keys()).index(st.session_state.active_session_id),
        )

        if selected_title != st.session_state.active_session_id:
            st.session_state.active_session_id = selected_title
            st.session_state.history_session_id = None
            st.rerun()

        col_new, col_clear, col_delete = st.columns(3)
        with col_new:
            if st.button("New", use_container_width=True):
                session = st.session_state.chat_api.create_session(
                    user_id=st.session_state.user_id,
                )
                st.session_state.active_session_id = session.session_id
                st.session_state.history_session_id = None
                st.session_state.messages = []
                st.rerun()

        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_api.clear_session_history(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.active_session_id,
                )
                st.session_state.messages = []
                st.session_state.history_session_id = st.session_state.active_session_id
                st.rerun()

        with col_delete:
            if st.button("Delete", use_container_width=True):
                session_id = st.session_state.active_session_id
                st.session_state.chat_api.delete_session(
                    session_id=session_id,
                    user_id=st.session_state.user_id,
                )
                remaining = st.session_state.chat_api.list_sessions(st.session_state.user_id)
                if remaining:
                    st.session_state.active_session_id = remaining[0].session_id
                else:
                    session = st.session_state.chat_api.create_session(
                        user_id=st.session_state.user_id,
                    )
                    st.session_state.active_session_id = session.session_id
                st.session_state.history_session_id = None
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.markdown("### Knowledge Base")
        st.caption(f"Namespace: `{NAMESPACE}`")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "markdown", "csv"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Index documents", use_container_width=True):
            total_chunks = 0
            with st.spinner("Chunking and indexing documents..."):
                for uploaded in uploaded_files:
                    total_chunks += st.session_state.chat_api.ingest_file(
                        uploaded.name,
                        uploaded.getvalue(),
                    )
            st.success(f"Indexed {total_chunks} chunks from {len(uploaded_files)} file(s).")

        st.divider()
        st.markdown("### Agent Capabilities")
        st.markdown(
            """
            - **Sessions**: Each conversation has its own persistent history
            - **Memory**: Remembers user-specific facts per session
            - **RAG**: MMR retrieval over chunked domain documents
            """
        )
