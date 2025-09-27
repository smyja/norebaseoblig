#!/usr/bin/env python3
"""Quick, provider-agnostic RAG evaluation using Ragas faithfulness.

This script runs a few queries against the local Autocomply retrieval/LLM stack,
collects contexts, and computes the Ragas faithfulness score.

Requirements (install once):
  pip install -e .[eval]

Usage examples:
  python scripts/evaluate_faithfulness.py \
      --industry banking_fintech --regulator CBN \
      --queries examples/queries.txt \
      --top-k 10

Environment:
  TOGETHER_API_KEY must be set.
  Optional: PHOENIX_ENABLED=true to launch Phoenix UI & instrumentation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv


def load_env() -> None:
    load_dotenv()
    load_dotenv("src/.env.local", override=False)
    load_dotenv("src/.env", override=False)


def read_queries(path: Path | None) -> List[str]:
    if not path:
        # Fallback starter queries
        return [
            "FX reporting obligations for banks",
            "deadlines for settlement under EFEMS",
            "penalties for non-compliance in FX matching",
        ]
    if not path.exists():
        raise SystemExit(f"queries file not found: {path}")
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            out.append(s)
    if not out:
        raise SystemExit("no queries found in file")
    return out


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(description="Evaluate RAG faithfulness on Autocomply")
    parser.add_argument("--industry", default=os.getenv("EVAL_INDUSTRY"))
    parser.add_argument("--regulator", default=os.getenv("EVAL_REGULATOR"))
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # Lazy imports so script remains optional
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import faithfulness  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Missing dependencies. Install with `pip install -e .[eval]` ({exc})"
        )

    # Phoenix (optional)
    try:
        from src.app.observability.phoenix_setup import enable_phoenix

        enable_phoenix(
            host=os.getenv("PHOENIX_HOST"),
            port=int(os.getenv("PHOENIX_PORT", "0") or 0) or None,
        )
    except Exception:
        pass

    # Use the same internals as the API for a faithful run
    from src.app.api.v1.autocomply import (
        QueryRequest,
        chat,
        _load_nodes_from_docstore,
    )
    from llama_index.retrievers.bm25 import BM25Retriever

    industry = args.industry
    regulator = args.regulator
    queries = read_queries(args.queries)
    top_k = int(args.top_k)

    if not os.getenv("TOGETHER_API_KEY"):
        raise SystemExit("TOGETHER_API_KEY missing; set it in your environment or src/.env.local")

    # Preload nodes and build a BM25 retriever mirroring /chat
    nodes = _load_nodes_from_docstore(industry, regulator)
    bm25 = BM25Retriever.from_defaults(nodes=nodes)

    rows = []
    for q in queries:
        # Answer via our standard pipeline
        req = QueryRequest(question=q, industry=industry, regulator=regulator, similarity_top_k=top_k)
        resp = chat(req)

        # Build contexts consistent with /chat BM25 path
        results = bm25.retrieve(q)
        top = results[:top_k]
        contexts: List[str] = []
        for r in top:
            n = r.node if hasattr(r, "node") else r
            text = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            if text:
                contexts.append(text)

        rows.append({
            "question": q,
            "answer": resp.answer,
            "contexts": contexts,
            # ground_truth optional for faithfulness-only eval
            "ground_truth": "",
        })

    ds = Dataset.from_list(rows)
    result = evaluate(dataset=ds, metrics=[faithfulness])
    print(result)


if __name__ == "__main__":
    main()

