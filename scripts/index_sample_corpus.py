#!/usr/bin/env python3
"""Index bundled sample_data/ files into Pinecone (CLI helper for live demo prep)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_agent.api import ChatAPI
from memory_agent.config import DEFAULT_DB_PATH, get_missing_config_keys
from memory_agent.demo.corpus import index_sample_corpus, list_sample_corpus_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Index sample_data/ into the knowledge base.")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"SQLite path (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    missing = get_missing_config_keys()
    if missing:
        print("Missing configuration:", ", ".join(missing), file=sys.stderr)
        print("Copy .env.example to .env and add API keys, then retry.", file=sys.stderr)
        return 1

    files = list_sample_corpus_files()
    if not files:
        print("No sample corpus files found under sample_data/", file=sys.stderr)
        return 1

    print(f"Indexing {len(files)} sample file(s)…")
    api = ChatAPI(db_path=args.db_path)
    summary = index_sample_corpus(api)

    for item in summary.results:
        if item.skipped:
            print(f"  skip  {item.source} (unchanged)")
        else:
            print(f"  index {item.source} ({item.chunks} chunks)")

    print(
        f"Done: {summary.indexed_chunks} new chunks, "
        f"{summary.skipped_files} skipped, "
        f"{summary.indexed_files}/{summary.total_files} files indexed this run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
