"""Normalize LangChain / Gemini message content to plain markdown text."""

from __future__ import annotations

import ast
from typing import Any


def extract_text_content(content: Any) -> str:
    """Return user-visible text from heterogeneous model message payloads."""
    if content is None:
        return ""

    if isinstance(content, str):
        recovered = _recover_stringified_blocks(content)
        if recovered is not content:
            return _polish(recovered)
        return _polish(content)

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = _text_from_block(block)
            if text:
                parts.append(text)
        return _polish("".join(parts))

    text = _text_from_block(content)
    return _polish(text if text else str(content))


def _polish(text: str) -> str:
    from memory_agent.utils.sources import polish_assistant_text

    return polish_assistant_text(text)


def _recover_stringified_blocks(value: str) -> str:
    """Recover text when legacy code stored ``str(list[dict])`` in checkpoints."""
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return value
    if "'type'" not in stripped and '"type"' not in stripped:
        return value
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value
    if isinstance(parsed, list):
        return extract_text_content(parsed)
    return value


def _text_from_block(block: Any) -> str:
    if isinstance(block, str):
        return block

    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type not in (None, "text"):
            return ""
        text = block.get("text")
        return str(text) if text else ""

    text = getattr(block, "text", None)
    if text:
        return str(text)

    return ""
