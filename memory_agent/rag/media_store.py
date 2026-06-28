"""Persist uploaded media bytes for retrieval-time multimodal understanding."""

from __future__ import annotations

from pathlib import Path

from memory_agent.config import UPLOADS_DIR


class MediaStore:
    """Filesystem store for binary RAG chunks (images, audio, video, PDF)."""

    def __init__(self, base_dir: str = UPLOADS_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, chunk_id: str, data: bytes, extension: str) -> str:
        """Save bytes and return a path relative to ``base_dir`` (portable across hosts)."""
        suffix = extension if extension.startswith(".") else f".{extension}"
        filename = f"{chunk_id}{suffix}"
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return filename

    def resolve_path(self, storage_path: str) -> Path:
        """Resolve stored path — supports legacy absolute paths and relative filenames."""
        path = Path(storage_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def exists(self, storage_path: str) -> bool:
        return self.resolve_path(storage_path).is_file()

    def load(self, storage_path: str) -> bytes:
        path = self.resolve_path(storage_path)
        return path.read_bytes()

    def try_load(self, storage_path: str) -> bytes | None:
        try:
            if not self.exists(storage_path):
                return None
            return self.load(storage_path)
        except OSError:
            return None

    def load_text(self, storage_path: str) -> str:
        path = self.resolve_path(storage_path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def try_load_text(self, storage_path: str) -> str | None:
        try:
            if not self.exists(storage_path):
                return None
            return self.load_text(storage_path)
        except OSError:
            return None
