#!/usr/bin/env python3
"""Serve a FastAPI layer exposing chat and obligation extraction over LlamaIndex."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load env from repo root, then fall back to src/.env.local and src/.env
load_dotenv()
load_dotenv("src/.env.local", override=False)
load_dotenv("src/.env", override=False)

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.together import TogetherLLM
from src.app.core.ai.embeddings import BatchedTogetherEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from src.app.observability.phoenix_setup import enable_phoenix, instrument_fastapi
from src.app.schemas.autocomply import (
    Obligation,
    ExtractionResult,
    QueryRequest,
    QueryResponse,
    ExtractRequest,
    ExtractResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autocomply")

PERSIST_DIR = Path("storage_autocomply")


def configure_models() -> None:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise SystemExit("TOGETHER_API_KEY missing. Add it to your environment or .env file.")
    together_model = os.getenv("TOGETHER_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    batch_size = int(os.getenv("TOGETHER_EMBED_BATCH", "32"))
    Settings.embed_model = BatchedTogetherEmbedding(
        model_name=together_model,
        api_key=api_key,
        batch_size=batch_size,
    )

    Settings.llm = TogetherLLM(
        model=os.getenv("TOGETHER_LLM_MODEL", "moonshotai/Kimi-K2-Instruct-0905"),
        api_key=api_key,
    )


configure_models()

# Optionally enable Phoenix tracing/instrumentation
enable_phoenix(
    host=os.getenv("PHOENIX_HOST"),
    port=int(os.getenv("PHOENIX_PORT", "0") or 0) or None,
)

if not PERSIST_DIR.exists():
    raise SystemExit("storage_autocomply not found. Run build_llamaindex.py first.")

storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
index = load_index_from_storage(storage_context)

app = FastAPI()

# Instrument FastAPI + HTTP clients for request traces (best‑effort)
instrument_fastapi(app)


def metadata_filter(industry: Optional[str], regulator: Optional[str]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if industry:
        filters["industry"] = industry
    if regulator:
        filters["regulator"] = regulator
    return filters


def hybrid_retrieve(query: str, top_k: int, filters: Dict[str, Any]):
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters or None)
    vector_nodes = retriever.retrieve(query)

    keyword_nodes = []
    if vector_nodes:
        bm25 = BM25Retriever.from_defaults(nodes=[candidate.node for candidate in vector_nodes])
        keyword_nodes = bm25.retrieve(query)

    merged = []
    seen = set()
    for candidate in vector_nodes + keyword_nodes:
        identifier = candidate.node.node_id
        if identifier in seen:
            continue
        seen.add(identifier)
        merged.append(candidate)

    return merged


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "index_path": str(PERSIST_DIR.resolve())}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest) -> QueryResponse:
    filters = metadata_filter(request.industry, request.regulator)
    nodes = hybrid_retrieve(request.question, request.similarity_top_k, filters)

    if not nodes:
        raise HTTPException(status_code=404, detail="No results")

    query_engine = index.as_query_engine(
        similarity_top_k=request.similarity_top_k,
        filters=filters or None,
        llm=Settings.llm,
    )
    answer = query_engine.query(request.question)

    # Conform to shared schema: list[str] labels for sources
    src_labels: List[str] = []
    seen: set[str] = set()
    for node_with_score in getattr(answer, "source_nodes", []):
        metadata = node_with_score.metadata or {}
        file_path = metadata.get("file_path") or metadata.get("filename")
        page_no = metadata.get("page_no")
        if not file_path:
            continue
        label = f"{file_path}#p{page_no}" if page_no else str(file_path)
        if label in seen:
            continue
        src_labels.append(label)
        seen.add(label)
        if len(src_labels) >= (request.max_sources or 3):
            break

    return QueryResponse(answer=str(answer), sources=src_labels)


@app.post("/extract_obligations", response_model=ExtractResponse)
def extract_obligations(request: ExtractRequest) -> ExtractResponse:
    filters = metadata_filter(request.industry, request.regulator)
    nodes = hybrid_retrieve(request.query, request.similarity_top_k, filters)

    if not nodes:
        return ExtractResponse()

    context_lines: List[str] = []
    used_sources: List[Dict[str, Any]] = []

    for candidate in nodes[: request.max_return]:
        metadata = candidate.node.metadata or {}
        header = f"[{metadata.get('regulator', '?')} | {metadata.get('filename', '?')} | p.{metadata.get('page_no', '?')}]"
        context_lines.append(header + "\n" + candidate.node.get_content())
        used_sources.append(
            {
                "file": metadata.get("file_path") or metadata.get("filename"),
                "page": metadata.get("page_no"),
                "industry": metadata.get("industry"),
                "regulator": metadata.get("regulator"),
            }
        )

    instructions = (
        "You are an extraction assistant. From the context below, extract explicit compliance obligations and return them under the key 'obligations'. "
        "Each obligation must include: regulator, instrument_name (infer from filename if needed), "
        "instrument_type (Act, Regulation, Guideline, Circular, Code), citation (section/reg number), "
        "actor (who must comply), obligation_text, trigger (if conditional), deadline (if any), penalty (if any), "
        "effective_date (if stated), and source_file/source_page from metadata. "
        "Explicitly search for penalties/sanctions (e.g., fine, penalty, sanctions, liable, contravention, violation, revocation, suspension) and populate the penalty field with the precise text or a concise summary tied to the obligation. "
        "If no explicit penalty is specified, leave penalty null. Do not invent."
    )

    example = (
        "Example (illustrative only):\n"
        '{"obligations":[{"regulator":"CBN","instrument_name":"XYZ Guidelines","instrument_type":"Guideline",'
        '"citation":"s.5(2)","actor":"Banks","obligation_text":"Banks shall maintain records of all domestic and cross-border transfers.",'
        '"trigger":null,"deadline":"within 30 days","penalty":null,"effective_date":null,"source_file":"xyz.pdf","source_page":12}]}'
    )

    prompt_template = PromptTemplate(
        "{instructions}\n\n"
        "Follow the format instructions precisely. {output_instructions}\n\n"
        f"{example}\n\n"
        "Context:\n{context}"
    )

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=ExtractionResult,
        prompt=prompt_template,
        llm=Settings.llm,
        temperature=0.1,
        input_key="context",
    )
    result = program(context="\n\n".join(context_lines), instructions=instructions)

    return ExtractResponse(obligations=result.obligations, used_sources=used_sources[:20])
