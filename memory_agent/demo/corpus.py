"""Index bundled sample documents for demos and live deployments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from memory_agent.config import PROJECT_ROOT, SAMPLE_DATA_DIR

SAMPLE_CORPUS_FILES = (
    "memory_agent_overview.md",
    "rag_pipeline_guide.md",
    "agent_tools_reference.md",
    "demo_faq.md",
)


@dataclass
class SampleCorpusResult:
    source: str
    chunks: int
    skipped: bool


@dataclass
class SampleCorpusSummary:
    indexed_chunks: int
    skipped_files: int
    results: list[SampleCorpusResult]

    @property
    def indexed_files(self) -> int:
        return sum(1 for item in self.results if item.chunks > 0)

    @property
    def total_files(self) -> int:
        return len(self.results)


def list_sample_corpus_files() -> list[Path]:
    """Return paths to bundled sample files that exist on disk."""
    files: list[Path] = []
    for name in SAMPLE_CORPUS_FILES:
        path = SAMPLE_DATA_DIR / name
        if path.is_file():
            files.append(path)
    return files


def index_sample_corpus(chat_api) -> SampleCorpusSummary:
    """Index all bundled sample files into the domain knowledge base."""
    results: list[SampleCorpusResult] = []
    indexed_chunks = 0
    skipped_files = 0

    for path in list_sample_corpus_files():
        raw_bytes = path.read_bytes()
        ingest = chat_api.ingest_file_detailed(path.name, raw_bytes)
        results.append(
            SampleCorpusResult(
                source=path.name,
                chunks=ingest.chunks,
                skipped=ingest.skipped,
            )
        )
        if ingest.skipped:
            skipped_files += 1
        else:
            indexed_chunks += ingest.chunks

    return SampleCorpusSummary(
        indexed_chunks=indexed_chunks,
        skipped_files=skipped_files,
        results=results,
    )


def sample_corpus_is_indexed(chat_api) -> bool:
    """True when every bundled sample file is registered in the document registry."""
    registered = {doc.source for doc in chat_api.list_documents()}
    expected = {path.name for path in list_sample_corpus_files()}
    return bool(expected) and expected.issubset(registered)
