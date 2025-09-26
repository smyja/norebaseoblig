#!/usr/bin/env python3
"""Build per-regulator LlamaIndex stores and a simple registry for hierarchical routing.

Output structure:
  storage_autocomply/
    indices/
      <industry>__<regulator>/  # individual persisted index
    registry.json                # list of children with paths and counts
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()
load_dotenv("src/.env.local", override=False)
load_dotenv("src/.env", override=False)

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.together import TogetherLLM
from src.app.core.ai.embeddings import BatchedTogetherEmbedding

PAGES_DIR = Path("parsed_pages")
PERSIST_ROOT = Path("storage_autocomply")
INDICES_DIR = PERSIST_ROOT / "indices"


def configure_models() -> None:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise SystemExit("TOGETHER_API_KEY missing. Add it to your environment or .env file.")
    together_model = os.getenv("TOGETHER_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    batch_size = int(os.getenv("TOGETHER_EMBED_BATCH", "32"))
    embedder = BatchedTogetherEmbedding(
        model_name=together_model,
        api_key=api_key,
        batch_size=batch_size,
    )
    Settings.embed_model = embedder

    Settings.llm = TogetherLLM(
        model=os.getenv("TOGETHER_LLM_MODEL", "moonshotai/Kimi-K2-Instruct-0905"),
        api_key=api_key,
    )


def iter_pages() -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for jf in sorted(PAGES_DIR.glob("*.jsonl")):
        items.append((jf.stem, jf))
    return items


def load_documents(jsonl_path: Path) -> List[Document]:
    splitter = SentenceSplitter(chunk_size=1_500, chunk_overlap=150)
    docs: List[Document] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = (item.get("text") or "").strip()
            if not text:
                continue
            metadata: Dict[str, Any] = {
                "industry": item.get("industry"),
                "regulator": item.get("regulator"),
                "filename": item.get("filename"),
                "file_path": item.get("file_path"),
                "page_no": item.get("page_no"),
                "page_count": item.get("page_count"),
                "listed_date": item.get("listed_date"),
                "category": item.get("category"),
                "department": item.get("department"),
                "sha256": item.get("sha256"),
            }
            for pos, chunk in enumerate(splitter.split_text(text), start=1):
                docs.append(Document(text=chunk, metadata={**metadata, "chunk_in_page": pos}))
    return docs


def main() -> None:
    if not PAGES_DIR.exists():
        raise SystemExit("parsed_pages directory missing. Run parse_corpus_with_ocr.py first.")

    configure_models()
    INDICES_DIR.mkdir(parents=True, exist_ok=True)

    registry_children: List[Dict[str, Any]] = []
    force_rebuild = os.getenv("FORCE_REBUILD", "false").lower() in {"1", "true", "yes"}
    only = os.getenv("ONLY_REGULATOR")
    for name, jf in iter_pages():
        if only and not (name == only or name.startswith(only)):
            continue
        docs = load_documents(jf)
        if not docs:
            continue
        persist_dir = INDICES_DIR / name
        if persist_dir.exists() and not force_rebuild:
            print(f"[skip] {name} exists at {persist_dir}; set FORCE_REBUILD=true to overwrite")
            parts = name.split("__", 1)
            industry = parts[0] if len(parts) == 2 else None
            regulator = parts[1] if len(parts) == 2 else name
            registry_children.append(
                {
                    "name": name,
                    "persist_dir": str(persist_dir),
                    "industry": industry,
                    "regulator": regulator,
                    "doc_count": None,
                }
            )
            continue

        print(f"[index] building {name} over {len(docs)} chunks")
        try:
            embedder.set_progress_total(len(docs), name=name)
        except Exception:
            pass
        idx = VectorStoreIndex.from_documents(docs)
        idx.storage_context.persist(persist_dir=str(persist_dir))
        # attempt to split name
        parts = name.split("__", 1)
        industry = parts[0] if len(parts) == 2 else None
        regulator = parts[1] if len(parts) == 2 else name
        registry_children.append(
            {
                "name": name,
                "persist_dir": str(persist_dir),
                "industry": industry,
                "regulator": regulator,
                "doc_count": len(docs),
            }
        )

    (PERSIST_ROOT).mkdir(parents=True, exist_ok=True)
    registry = {"children": registry_children}
    with (PERSIST_ROOT / "registry.json").open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote registry with {len(registry_children)} children -> {str((PERSIST_ROOT / 'registry.json').resolve())}")


if __name__ == "__main__":
    main()
