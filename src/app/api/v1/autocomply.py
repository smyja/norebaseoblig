from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.together import TogetherLLM
from src.app.core.ai.embeddings import BatchedTogetherEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

# Lazy imports of heavy dependencies to avoid import-time crashes
_llama_ready = False
_index = None
_child_indices: dict[str, Any] | None = None


def _configure_models() -> None:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="TOGETHER_API_KEY is not configured")
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


def _get_index():
    global _llama_ready, _index
    if _index is not None:
        return _index

    try:
        _configure_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"LlamaIndex not available: {exc}")

    persist_dir = Path(os.getenv("AUTOCOMPLY_STORAGE", "storage_autocomply"))
    if not persist_dir.exists():
        raise HTTPException(status_code=503, detail="Index storage not found. Run build_llamaindex.py first.")

    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        _index = load_index_from_storage(storage_context)
        _llama_ready = True
        return _index
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load index: {exc}")


from ...schemas.autocomply import (
    ExtractRequest,
    ExtractResponse,
    ExtractionResult,
    Obligation,
    QueryRequest,
    QueryResponse,
)


router = APIRouter(prefix="/autocomply", tags=["autocomply"])


def _metadata_filter(industry: Optional[str], regulator: Optional[str]) -> Dict[str, Any]:
    flt: Dict[str, Any] = {}
    if industry:
        flt["industry"] = industry
    if regulator:
        flt["regulator"] = regulator
    return flt


def _hybrid_retrieve(query: str, top_k: int, filters: Dict[str, Any]):
    index = _get_index()

    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters or None)
    vector_nodes = retriever.retrieve(query)

    keyword_nodes = []
    if vector_nodes:
        bm25 = BM25Retriever.from_defaults(nodes=[candidate.node for candidate in vector_nodes])
        keyword_nodes = bm25.retrieve(query)

    merged = []
    seen = set()
    for candidate in vector_nodes + keyword_nodes:
        nid = candidate.node.node_id
        if nid in seen:
            continue
        seen.add(nid)
        merged.append(candidate)

    return merged


# ---------- Hierarchical (per-regulator) fan-out retrieval ----------
def _load_child_indices() -> dict[str, Any]:
    global _child_indices
    if _child_indices is not None:
        return _child_indices

    registry_path = Path(os.getenv("AUTOCOMPLY_REGISTRY", "storage_autocomply/registry.json"))
    if not registry_path.exists():
        raise HTTPException(status_code=503, detail="Registry not found. Run build_llamaindex_graph.py.")

    try:
        import json
        from llama_index.core import StorageContext, load_index_from_storage

        with registry_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)

        child_indices: dict[str, Any] = {}
        for item in registry.get("children", []):
            name = item["name"]
            persist_dir = item["persist_dir"]
            sc = StorageContext.from_defaults(persist_dir=persist_dir)
            child_indices[name] = load_index_from_storage(sc)
        _child_indices = child_indices
        return child_indices
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load child indices: {exc}")


def _select_children(children: dict[str, Any], limit: int = 6) -> List[str]:
    names = sorted(children.keys())
    return names[:limit]


def _fanout_retrieve(
    query: str,
    similarity_top_k: int,
    industry: Optional[str],
    regulator: Optional[str],
    per_child: int = 4,
    max_children: int = 6,
):
    children = _load_child_indices()

    if regulator:
        explicit = [
            k
            for k in children
            if k.endswith(f"__{regulator}") and (industry is None or k.startswith(f"{industry}__"))
        ]
        selected_names = explicit or _select_children(children, limit=max_children)
    else:
        selected_names = _select_children(children, limit=max_children)

    merged = []
    for name in selected_names:
        idx = children[name]
        retriever = idx.as_retriever(similarity_top_k=per_child)
        try:
            nodes = retriever.retrieve(query)
        except Exception:
            nodes = []
        merged.extend(nodes)

    merged.sort(key=lambda n: (getattr(n, "score", None) or 0.0), reverse=True)

    seen = set()
    result = []
    for cand in merged:
        nid = cand.node.node_id
        if nid in seen:
            continue
        seen.add(nid)
        result.append(cand)
        if len(result) >= similarity_top_k:
            break

    return result


@router.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest) -> QueryResponse:
    use_graph = os.getenv("AUTOCOMPLY_USE_GRAPH", "false").lower() in {"1", "true", "yes"}

    if use_graph:
        nodes = _fanout_retrieve(
            query=request.question,
            similarity_top_k=request.similarity_top_k,
            industry=request.industry,
            regulator=request.regulator,
        )
        if not nodes:
            raise HTTPException(status_code=404, detail="No results")

        context_lines: List[str] = []
        sources: List[Dict[str, Any]] = []
        for cand in nodes:
            meta = cand.node.metadata or {}
            header = f"[{meta.get('regulator','?')} | {meta.get('filename','?')} | p.{meta.get('page_no','?')}]"
            context_lines.append(header + "\n" + cand.node.get_content())
            sources.append(
                {
                    "file": meta.get("file_path") or meta.get("filename"),
                    "page": meta.get("page_no"),
                    "industry": meta.get("industry"),
                    "regulator": meta.get("regulator"),
                    "url": meta.get("source_url"),
                }
            )

        prompt = (
            "You are a compliance analyst. Use the context snippets to answer the user's question conservatively. "
            "Prefer citing clearly scoped obligations. If uncertain, say so. "
            "Include brief inline citations like [regulator | file | p.X] when referencing specifics.\n\n"
            f"Question: {request.question}\n\nContext:\n" + "\n\n".join(context_lines)
        )
        completion = Settings.llm.complete(prompt)
        answer_text = getattr(completion, "text", str(completion))
        return QueryResponse(answer=answer_text, sources=sources[:5])
    else:
        index = _get_index()

        filters = _metadata_filter(request.industry, request.regulator)
        nodes = _hybrid_retrieve(request.question, request.similarity_top_k, filters)
        if not nodes:
            raise HTTPException(status_code=404, detail="No results")

        qe = index.as_query_engine(
            similarity_top_k=request.similarity_top_k,
            filters=filters or None,
            llm=Settings.llm,
        )
        answer = qe.query(request.question)

        sources: List[Dict[str, Any]] = []
    for n in getattr(answer, "source_nodes", []):
        meta = n.metadata or {}
        sources.append(
            {
                "file": meta.get("file_path") or meta.get("filename"),
                "page": meta.get("page_no"),
                "industry": meta.get("industry"),
                "regulator": meta.get("regulator"),
            }
        )

        return QueryResponse(answer=str(answer), sources=sources[:5])


@router.post("/extract_obligations", response_model=ExtractResponse)
def extract_obligations(request: ExtractRequest) -> ExtractResponse:
    use_graph = os.getenv("AUTOCOMPLY_USE_GRAPH", "false").lower() in {"1", "true", "yes"}

    if use_graph:
        nodes = _fanout_retrieve(
            query=request.query,
            similarity_top_k=request.similarity_top_k,
            industry=request.industry,
            regulator=request.regulator,
            per_child=4,
        )
    else:
        filters = _metadata_filter(request.industry, request.regulator)
        nodes = _hybrid_retrieve(request.query, request.similarity_top_k, filters)

    if not nodes:
        return ExtractResponse()

    context_lines: List[str] = []
    used_sources: List[Dict[str, Any]] = []

    for candidate in nodes[: request.max_return]:
        meta = candidate.node.metadata or {}
        header = f"[{meta.get('regulator', '?')} | {meta.get('filename', '?')} | p.{meta.get('page_no', '?')}]"
        context_lines.append(header + "\n" + candidate.node.get_content())
        used_sources.append(
            {
                "file": meta.get("file_path") or meta.get("filename"),
                "page": meta.get("page_no"),
                "industry": meta.get("industry"),
                "regulator": meta.get("regulator"),
            }
        )

    prompt = (
        "From the context, extract explicit compliance obligations and return an array under 'obligations'. "
        "Each obligation must include: regulator, instrument_name (infer from filename if needed), "
        "instrument_type (Act, Regulation, Guideline, Circular, Code), citation (section/reg number), "
        "actor (who must comply), obligation_text, trigger (if conditional), deadline (if any), penalty (if any), "
        "effective_date (if stated), and source_file/source_page from metadata. "
        "Focus on Nigeria regulators. If unsure, leave a field null. Do not invent."
    )

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=ExtractionResult,
        prompt=prompt,
        llm=Settings.llm,
        temperature=0.1,
        input_key="context",
    )
    result = program(context="\n\n".join(context_lines))

    return ExtractResponse(obligations=result.obligations, used_sources=used_sources[:20])
