"""Streamlit chat interface."""

from __future__ import annotations

from datetime import datetime

import streamlit as st


def render_chat() -> None:
    """Render the main chat area and handle user input."""
    active_session = st.session_state.chat_api.sessions.get_session(
        st.session_state.active_session_id
    )
    title = active_session.title if active_session else "Memory Agent Chat"

    st.title(title)
    st.caption("Chat with persistent sessions, memory, and document retrieval")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp,
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_api.chat_dict(
                    message=prompt,
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.active_session_id,
                )

            if response["success"]:
                st.markdown(response["response"])
                st.session_state.messages = st.session_state.chat_api.get_history(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.active_session_id,
                )
                st.session_state.history_session_id = st.session_state.active_session_id
                st.rerun()
            else:
                st.error(f"Error: {response['error']}")
