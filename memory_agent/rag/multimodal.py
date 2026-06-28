"""Gemini multimodal model for understanding retrieved media."""

from __future__ import annotations

from google.genai import types

from memory_agent.config import get_gemini_model_chain, get_gemini_multimodal_model
from memory_agent.google.genai_client import get_genai_client
from memory_agent.google.retry import call_with_model_fallback
from memory_agent.rag.media_store import MediaStore


class GeminiMultimodalClient:
    """Uses Gemini Flash to turn media chunks into text context for the chat LLM."""

    def __init__(self, media_store: MediaStore | None = None):
        self.client = get_genai_client()
        self.model = get_gemini_multimodal_model()
        self.model_chain = get_gemini_model_chain(self.model)
        self.media_store = media_store or MediaStore()

    def describe_media(self, data: bytes, mime_type: str, source: str, modality: str) -> str:
        prompt = (
            f"Describe this {modality} file ({source}) in detail for a retrieval system. "
            "Include visible objects, text, spoken content, actions, and any facts useful for Q&A."
        )
        contents = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=data, mime_type=mime_type),
        ]

        def _generate(model_name: str):
            return self.client.models.generate_content(
                model=model_name,
                contents=contents,
            )

        response = call_with_model_fallback(self.model_chain, _generate)
        return (response.text or "").strip() or f"{modality} content from {source}"

    def enrich_metadata(self, metadata: dict) -> str:
        """Build LLM-readable context for a retrieved chunk."""
        source = metadata.get("source", "unknown")
        modality = metadata.get("modality", "text")
        text_preview = metadata.get("text_preview", "")

        if modality == "text":
            storage_path = metadata.get("storage_path")
            if storage_path:
                return self.media_store.load_text(storage_path)
            return text_preview

        storage_path = metadata.get("storage_path")
        mime_type = metadata.get("mime_type", "application/octet-stream")
        if not storage_path:
            return text_preview or f"{modality} content from {source}"

        data = self.media_store.load(storage_path)
        if modality == "pdf":
            return self.describe_media(data, mime_type, source, "PDF document")
        return self.describe_media(data, mime_type, source, modality)
