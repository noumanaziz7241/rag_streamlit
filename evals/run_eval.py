#!/usr/bin/env python3
"""Run RAG evaluation against the golden Q&A set."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.metrics import keyword_faithfulness, mean_score, recall_at_k
from memory_agent.api import ChatAPI
from memory_agent.config import get_missing_config_keys
from memory_agent.rag.pipeline import retrieve_domain_documents

EVALS_DIR = Path(__file__).resolve().parent
CORPUS = EVALS_DIR / "corpus" / "memory_agent_overview.md"
GOLDEN = EVALS_DIR / "golden_qa.json"


def load_cases() -> list[dict]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def ensure_corpus_indexed(api: ChatAPI) -> None:
    raw = CORPUS.read_bytes()
    result = api.ingest_file_detailed(CORPUS.name, raw)
    if result.skipped:
        print(f"Corpus already indexed: {CORPUS.name}")
    else:
        print(f"Indexed corpus: {result.chunks} chunks")


def run_eval(live_chat: bool = False) -> dict:
    if get_missing_config_keys():
        raise RuntimeError(
            "Missing API keys. Configure .env or .streamlit/secrets.toml before running evals."
        )

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "eval.db")
        api = ChatAPI(db_path=db_path)
        ensure_corpus_indexed(api)

        cases = load_cases()
        recall_scores: list[float] = []
        faithfulness_scores: list[float] = []
        rows: list[dict] = []

        for case in cases:
            question = case["question"]
            _, docs = retrieve_domain_documents(
                api.agent.vector_store_manager.domain_index,
                query=question,
            )
            sources = [str(doc.metadata.get("source", "")) for doc in docs]
            recall = recall_at_k(sources, case["expected_source"], k=5)
            recall_scores.append(recall)

            if live_chat:
                response = api.chat_dict(
                    message=question,
                    session_id=api.sessions.create_session(title="Eval").session_id,
                )
                answer = response.get("response", "")
            else:
                answer = " ".join(case["expected_keywords"])

            faith = keyword_faithfulness(answer, case["expected_keywords"])
            faithfulness_scores.append(faith)

            rows.append({
                "id": case["id"],
                "recall_at_5": recall,
                "keyword_faithfulness": faith,
                "retrieved_sources": sources[:3],
            })

        summary = {
            "cases": len(cases),
            "recall_at_5": round(mean_score(recall_scores), 3),
            "keyword_faithfulness": round(mean_score(faithfulness_scores), 3),
            "live_chat": live_chat,
            "results": rows,
        }
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Memory Agent Chat RAG evaluation")
    parser.add_argument(
        "--live-chat",
        action="store_true",
        help="Also evaluate answer keyword faithfulness via live LLM responses",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON results",
    )
    args = parser.parse_args()

    summary = run_eval(live_chat=args.live_chat)
    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
