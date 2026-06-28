"""Document management UI for the knowledge base."""

from __future__ import annotations

import streamlit as st


def render_document_manager() -> None:
    """List indexed documents and allow deletion."""
    documents = st.session_state.chat_api.list_documents()

    with st.expander(f"Indexed documents ({len(documents)})", expanded=bool(documents)):
        if not documents:
            st.caption("No documents indexed yet. Upload files below to build your knowledge base.")
            return

        for doc in documents:
            col_info, col_action = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{doc.source}**")
                st.caption(
                    f"`{doc.modality}` · {doc.chunk_count} chunks · "
                    f"indexed {doc.indexed_at[:19].replace('T', ' ')} UTC"
                )
            with col_action:
                if st.button("Delete", key=f"delete_doc_{doc.doc_id}", use_container_width=True):
                    deleted = st.session_state.chat_api.delete_document(doc.doc_id)
                    if deleted:
                        st.success(f"Removed {doc.source}")
                        st.rerun()
                    else:
                        st.error("Could not delete document")
