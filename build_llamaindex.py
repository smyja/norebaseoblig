#!/usr/bin/env python3
"""Build a persisted LlamaIndex instance from parsed_pages JSONL files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load env from repo root, then fall back to src/.env.local and src/.env
load_dotenv()
load_dotenv("src/.env.local", override=False)
load_dotenv("src/.env", override=False)

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.together import TogetherLLM
from src.app.core.ai.embeddings import BatchedTogetherEmbedding

PAGES_DIR = Path("parsed_pages")
PERSIST_DIR = Path("storage_autocomply")


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

    # LLM isn't required for building, but keep Settings.llm consistent for downstream usage
    Settings.llm = TogetherLLM(
        model=os.getenv("TOGETHER_LLM_MODEL", "moonshotai/Kimi-K2-Instruct-0905"),
        api_key=api_key,
    )


def load_documents() -> List[Document]:
    chunk_size = int(os.getenv("SENTENCE_CHUNK_SIZE", "1500"))
    chunk_overlap = int(os.getenv("SENTENCE_CHUNK_OVERLAP", "150"))
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents: List[Document] = []

    for jsonl_path in sorted(PAGES_DIR.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                text = item.get("text", "")
                if not text.strip():
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

                for position, chunk in enumerate(splitter.split_text(text), start=1):
                    documents.append(
                        Document(
                            text=chunk,
                            metadata={**metadata, "chunk_in_page": position},
                        )
                    )

    return documents


def main() -> None:
    if not PAGES_DIR.exists():
        raise SystemExit("parsed_pages directory missing. Run parse_corpus_with_ocr.py first.")

    configure_models()

    documents = load_documents()
    if not documents:
        raise SystemExit("No parsed pages available. Did parse_corpus_with_ocr.py produce any output?")

    # Provide progress context to the embedder (optional)
    try:
        from src.app.core.ai.embeddings import BatchedTogetherEmbedding as _BTE

        if isinstance(Settings.embed_model, _BTE):
            Settings.embed_model.set_progress_total(len(documents), name="build")
    except Exception:
        pass

    print(f"[index] building over {len(documents)} chunks")
    index = VectorStoreIndex.from_documents(documents)

    storage_context: StorageContext = index.storage_context
    storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"[ok] persisted to {PERSIST_DIR.resolve()}")


if __name__ == "__main__":
    main()
