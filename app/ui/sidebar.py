"""Streamlit sidebar: sessions and document ingestion."""

from __future__ import annotations

import streamlit as st

from memory_agent.config import NAMESPACE

from app.ui.documents import render_document_manager


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
        st.markdown("### Sample corpus")
        st.caption("Bundled demo docs in `sample_data/` — no upload needed.")

        if st.button("Index sample corpus", use_container_width=True):
            with st.spinner("Indexing sample corpus…"):
                summary = st.session_state.chat_api.index_sample_corpus()
            if summary.indexed_chunks:
                st.success(
                    f"Indexed {summary.indexed_chunks} chunks from "
                    f"{summary.indexed_files} file(s)."
                )
            elif summary.skipped_files:
                st.info("Sample corpus already indexed (unchanged files skipped).")
            else:
                st.warning("No sample files were indexed.")

        st.divider()
        st.markdown("### Knowledge Base")
        st.caption(f"Namespace: `{NAMESPACE}`")

        render_document_manager()

        uploaded_files = st.file_uploader(
            "Upload files (any format)",
            accept_multiple_files=True,
            help=(
                "Indexed with Gemini Embedding 2: text, PDF, images, audio, video, "
                "Office (DOCX/XLSX/PPTX), code, and more."
            ),
        )

        if uploaded_files and st.button("Index documents", use_container_width=True):
            total_chunks = 0
            skipped = 0
            with st.spinner("Chunking and indexing documents..."):
                for uploaded in uploaded_files:
                    result = st.session_state.chat_api.ingest_file_detailed(
                        uploaded.name,
                        uploaded.getvalue(),
                    )
                    if result.skipped:
                        skipped += 1
                    else:
                        total_chunks += result.chunks
            if total_chunks:
                st.success(f"Indexed {total_chunks} chunks from {len(uploaded_files) - skipped} file(s).")
            if skipped:
                st.info(f"Skipped {skipped} unchanged file(s) already in the knowledge base.")
            if not total_chunks and not skipped:
                st.warning("No chunks were indexed.")

        st.divider()
        st.markdown("### Agent Capabilities")
        st.markdown(
            """
            - **Streaming**: Responses appear token-by-token
            - **Sources**: Expand citations for retrieved documents
            - **Tools**: See memory and retrieval tool activity per reply
            - **Sessions**: Each conversation has its own persistent history
            - **Memory**: Remembers user-specific facts per session
            - **RAG**: Multimodal retrieval (text, image, audio, video, PDF) via Gemini Embedding 2
            """
        )
