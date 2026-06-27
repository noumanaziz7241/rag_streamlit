#!/usr/bin/env python3
"""Create Pinecone indexes required by Memory Agent Chat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from memory_agent.config import EMBEDDING_DIMENSION, get_config_value  # noqa: E402
from pinecone import Pinecone, ServerlessSpec  # noqa: E402

DEFAULT_DOMAIN_INDEX = "memory-agent-domain"
DEFAULT_MEMORY_INDEX = "memory-agent-memory"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_METRIC = "cosine"


def _resolve_index_names(
    domain_name: str | None,
    memory_name: str | None,
) -> tuple[str, str]:
    domain = domain_name
    memory = memory_name

    if not domain:
        try:
            domain = get_config_value("PINECONE_INDEX_NAME")
        except ValueError:
            domain = DEFAULT_DOMAIN_INDEX

    if not memory:
        try:
            memory = get_config_value("PINECONE_MEMORY_INDEX_NAME")
        except ValueError:
            memory = DEFAULT_MEMORY_INDEX

    return domain, memory


def ensure_index(
    pc: Pinecone,
    name: str,
    *,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
) -> str:
    if pc.has_index(name):
        return f"exists: {name}"

    pc.create_index(
        name=name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    return f"created: {name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Pinecone indexes for domain RAG and session memory.",
    )
    parser.add_argument(
        "--domain-index",
        help=f"Domain knowledge index name (default: env or {DEFAULT_DOMAIN_INDEX})",
    )
    parser.add_argument(
        "--memory-index",
        help=f"Session memory index name (default: env or {DEFAULT_MEMORY_INDEX})",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=EMBEDDING_DIMENSION,
        help=f"Vector dimension (default: {EMBEDDING_DIMENSION} for gemini-embedding-2)",
    )
    parser.add_argument("--cloud", default=DEFAULT_CLOUD, choices=("aws", "gcp", "azure"))
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing indexes in the Pinecone project",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without creating indexes",
    )
    args = parser.parse_args()

    try:
        api_key = get_config_value("PINECONE_API_KEY")
    except ValueError:
        if args.dry_run:
            api_key = None
        else:
            print(
                "Missing PINECONE_API_KEY. Set it in .env or .streamlit/secrets.toml.",
                file=sys.stderr,
            )
            return 1

    domain_name, memory_name = _resolve_index_names(args.domain_index, args.memory_index)

    print(f"Pinecone project indexes to ensure:")
    print(f"  domain: {domain_name}")
    print(f"  memory: {memory_name}")
    print(
        f"  spec: dimension={args.dimension}, metric={args.metric}, "
        f"cloud={args.cloud}, region={args.region}"
    )

    if args.dry_run:
        for label, name in (("domain", domain_name), ("memory", memory_name)):
            print(
                f"[{label}] would create: {name} "
                f"(dimension={args.dimension}, metric={args.metric}, "
                f"cloud={args.cloud}, region={args.region})"
            )
        return 0

    pc = Pinecone(api_key=api_key)

    if args.list:
        indexes = pc.list_indexes()
        names = [item.name for item in indexes]
        if not names:
            print("No indexes found in this Pinecone project.")
        else:
            print("Existing indexes:")
            for item in indexes:
                print(
                    f"  - {item.name}: dimension={item.dimension}, "
                    f"metric={item.metric}, "
                    f"host={getattr(item, 'host', 'n/a')}"
                )
        return 0

    for label, name in (("domain", domain_name), ("memory", memory_name)):
        result = ensure_index(
            pc,
            name,
            dimension=args.dimension,
            metric=args.metric,
            cloud=args.cloud,
            region=args.region,
        )
        print(f"[{label}] {result}")

    print("\nAdd these to .env if not already set:")
    print(f"PINECONE_INDEX_NAME={domain_name}")
    print(f"PINECONE_MEMORY_INDEX_NAME={memory_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
