#!/usr/bin/env python3
"""Build a Knowledge Graph (Property Graph Index) from parsed pages.

This is distinct from build_llamaindex_graph.py, which builds a graph of per-regulator
vector indices (for hierarchical routing). This script builds an actual knowledge graph
of entities and relations extracted from the text and persists it under
storage_autocomply/kg/graph_store.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv

# Load envs (repo root first, then src/.env.local and src/.env)
load_dotenv()
load_dotenv("src/.env.local", override=False)
load_dotenv("src/.env", override=False)

from llama_index.core import Document, Settings, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.indices.property_graph import (
    PropertyGraphIndex,
    SchemaLLMPathExtractor,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.together import TogetherLLM

PAGES_DIR = Path("parsed_pages")
KG_DIR = Path("storage_autocomply/kg")


def configure_models() -> None:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise SystemExit("TOGETHER_API_KEY missing. Add it to .env or export it.")
    Settings.llm = TogetherLLM(
        model=os.getenv("TOGETHER_LLM_MODEL", "moonshotai/Kimi-K2-Instruct-0905"),
        api_key=api_key,
    )


def load_documents() -> List[Document]:
    splitter = SentenceSplitter(chunk_size=int(os.getenv("KG_CHUNK_SIZE", "1200")), chunk_overlap=150)
    docs: List[Document] = []
    for jsonl in sorted(PAGES_DIR.glob("*.jsonl")):
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                meta: Dict[str, Any] = {
                    "industry": item.get("industry"),
                    "regulator": item.get("regulator"),
                    "filename": item.get("filename"),
                    "file_path": item.get("file_path"),
                    "page_no": item.get("page_no"),
                    "page_count": item.get("page_count"),
                }
                for pos, chunk in enumerate(splitter.split_text(text), start=1):
                    docs.append(Document(text=chunk, metadata={**meta, "chunk_in_page": pos}))
    return docs


def main() -> None:
    if not PAGES_DIR.exists():
        raise SystemExit("parsed_pages missing. Run parse_corpus_with_ocr.py first.")

    configure_models()
    docs = load_documents()
    if not docs:
        raise SystemExit("No docs loaded from parsed_pages.")

    # Define a light schema for regulatory obligations
    Entities = Literal["REGULATOR", "INSTRUMENT", "OBLIGATION", "ACTOR", "DEADLINE", "PENALTY", "CITATION"]
    Relations = Literal["ISSUED", "APPLIES_TO", "HAS_DEADLINE", "HAS_PENALTY", "CITES", "CONTAINS"]
    schema = {
        "REGULATOR": ["ISSUED", "CONTAINS"],
        "INSTRUMENT": ["CONTAINS", "CITES"],
        "OBLIGATION": ["APPLIES_TO", "HAS_DEADLINE", "HAS_PENALTY", "CITES"],
        "ACTOR": ["APPLIES_TO"],
        "DEADLINE": ["HAS_DEADLINE"],
        "PENALTY": ["HAS_PENALTY"],
        "CITATION": ["CITES"],
    }

    extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=Entities,
        possible_relations=Relations,
        kg_validation_schema=schema,  # constrain relation types
        strict=True,
        max_triplets_per_chunk=int(os.getenv("KG_MAX_TRIPLETS_PER_CHUNK", "10")),
        num_workers=int(os.getenv("KG_NUM_WORKERS", "4")),
    )

    graph_store = SimpleGraphStore()
    storage = StorageContext.from_defaults(graph_store=graph_store)

    print(f"[kg] extracting triplets over {len(docs)} chunks …")
    index = PropertyGraphIndex.from_documents(
        docs,
        kg_extractors=[extractor],
        storage_context=storage,
    )

    KG_DIR.mkdir(parents=True, exist_ok=True)
    storage.persist(persist_dir=str(KG_DIR))
    print(f"[ok] KG persisted to {KG_DIR.resolve()}/graph_store.json")


if __name__ == "__main__":
    main()

