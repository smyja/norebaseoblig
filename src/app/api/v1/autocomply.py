from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode
from ...schemas.autocomply import (
    ExtractRequest,
    ExtractResponse,
    ExtractionResult,
)


router = APIRouter(prefix="/autocomply", tags=["autocomply"])


# Minimal model configuration using Together
api_key = os.getenv("TOGETHER_API_KEY")
if api_key:
    Settings.embed_model = TogetherEmbedding(
        model_name=os.getenv("TOGETHER_EMBED_MODEL", "togethercomputer/m2-bert-80M-32k-retrieval"),
        api_key=api_key,
    )
    Settings.llm = TogetherLLM(
        model=os.getenv("TOGETHER_LLM_MODEL", "moonshotai/Kimi-K2-Instruct-0905"),
        api_key=api_key,
    )


class QueryRequest(BaseModel):
    question: str
    similarity_top_k: Optional[int] = 10
    reranker_top_n: Optional[int] = 3
    use_llm_reranker: Optional[bool] = False
    max_sources: Optional[int] = 3
    industry: Optional[str] = None
    regulator: Optional[str] = None
    keyword_only: Optional[bool] = None  # keyword/BM25 only retrieval


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []


def _resolve_persist_dir(industry: Optional[str], regulator: Optional[str]) -> Path:
    """Load a single persisted index.

    Preference order:
      1) AUTOCOMPLY_CHILD_INDEX (explicit child name e.g., banking_fintech__CBN)
      2) storage_autocomply/indices/<industry>__<regulator>
      3) storage_autocomply (root index)
    """

    root_dir = Path(os.getenv("AUTOCOMPLY_STORAGE", "storage_autocomply"))
    child_name = os.getenv("AUTOCOMPLY_CHILD_INDEX")

    candidate_dirs = []
    if child_name:
        candidate_dirs.append(root_dir / "indices" / child_name)
    if industry and regulator:
        candidate_dirs.append(root_dir / "indices" / f"{industry}__{regulator}")
    candidate_dirs.append(root_dir)

    for pd in candidate_dirs:
        if not pd.exists():
            continue
        return pd

    if not root_dir.exists():
        raise HTTPException(status_code=503, detail="Index storage not found. Build an index first.")
    raise HTTPException(status_code=503, detail="No suitable index found. Build root or child index.")


def _load_index(industry: Optional[str], regulator: Optional[str]):
    persist_dir = _resolve_persist_dir(industry, regulator)
    try:
        sc = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(sc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load index: {exc}")


_nodes_cache: dict[str, List[TextNode]] = {}


def _load_nodes_from_docstore(industry: Optional[str], regulator: Optional[str]) -> List[TextNode]:
    """Load all TextNodes directly from docstore.json in the selected persist dir."""
    persist_dir = _resolve_persist_dir(industry, regulator)
    key = str(persist_dir)
    cached = _nodes_cache.get(key)
    if cached:
        return cached
    docstore_path = persist_dir / "docstore.json"
    if not docstore_path.exists():
        raise HTTPException(status_code=503, detail=f"docstore.json not found under {persist_dir}")
    try:
        import json

        with docstore_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("docstore/data", {})
        nodes: List[TextNode] = []
        for nid, wrapper in records.items():
            payload = wrapper.get("__data__", {})
            text = payload.get("text") or ""
            if not text.strip():
                continue
            md = payload.get("metadata") or {}
            nodes.append(TextNode(text=text, metadata=md, id_=nid))
        if not nodes:
            raise HTTPException(status_code=503, detail="No nodes found in docstore.json")
        _nodes_cache[key] = nodes
        return nodes
    except HTTPException:
        # Bubble up known HTTP errors as-is
        raise
    except Exception as exc:  # noqa: BLE001
        # Wrap unexpected errors with a 500 to keep the API consistent
        raise HTTPException(status_code=500, detail=f"Failed to read docstore: {exc}")
 

@router.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest) -> QueryResponse:
    if not os.getenv("TOGETHER_API_KEY"):
        raise HTTPException(status_code=500, detail="TOGETHER_API_KEY is not configured")

    # Decide retrieval mode (keyword-only default if env set)
    keyword_only = request.keyword_only
    if keyword_only is None:
        keyword_only = os.getenv("AUTOCOMPLY_KEYWORD_ONLY", "true").lower() in {"1", "true", "yes"}

    if keyword_only:
        # BM25 over all nodes in the selected child/root index
        nodes = _load_nodes_from_docstore(request.industry, request.regulator)
        bm25 = BM25Retriever.from_defaults(nodes=nodes)
        results = bm25.retrieve(request.question)
        # Optional LLM rerank
        if request.use_llm_reranker:
            reranker = LLMRerank(choice_batch_size=5, top_n=request.reranker_top_n or 3, llm=Settings.llm)
            try:
                results = reranker.postprocess_nodes(results, query_str=request.question)  # type: ignore[arg-type]
            except Exception:
                pass
        # Build context: pure BM25 top-K, no heuristic filtering
        k = request.similarity_top_k or 10
        top = results[:k]
        context = []
        sources: List[str] = []
        seen: set[str] = set()
        for r in top:
            n = r.node if hasattr(r, "node") else r
            meta = getattr(n, "metadata", None) or {}
            file_path = meta.get("file_path") or meta.get("filename")
            page_no = meta.get("page_no")
            header = f"[{meta.get('regulator','?')} | {meta.get('filename','?')} | p.{page_no or '?'}]"
            context.append(header + "\n" + (n.get_content() if hasattr(n, "get_content") else n.text))
            if file_path:
                label = f"{file_path}#p{page_no}" if page_no else str(file_path)
                if label not in seen:
                    sources.append(label)
                    seen.add(label)
            if len(sources) >= (request.max_sources or 3):
                pass

        # Ask LLM with simple prompt
        prompt = (
            "You are a compliance analyst. Use the context snippets to answer the user's question conservatively. "
            "Prefer citing clearly scoped obligations. If uncertain, say so. "
            "Include brief inline citations like [regulator | file | p.X] when referencing specifics.\n\n"
            f"Question: {request.question}\n\nContext:\n" + "\n\n".join(context)
        )
        completion = Settings.llm.complete(prompt)
        answer_text = getattr(completion, "text", str(completion))
        return QueryResponse(answer=answer_text, sources=sources[: (request.max_sources or 3)])
    else:
        # Vector path (kept for completeness)
        index = _load_index(request.industry, request.regulator)
        node_postprocessors = []
        if request.use_llm_reranker:
            node_postprocessors.append(
                LLMRerank(choice_batch_size=5, top_n=request.reranker_top_n or 3, llm=Settings.llm)
            )
        query_engine = index.as_query_engine(
            similarity_top_k=request.similarity_top_k or 10,
            llm=Settings.llm,
            node_postprocessors=node_postprocessors,
        )
        response = query_engine.query(request.question)
        if not response or str(response).strip() == "" or str(response) == "Empty Response":
            raise HTTPException(status_code=404, detail="No results found")
        sources: List[str] = []
        seen: set[str] = set()
        for node in getattr(response, "source_nodes", []) or []:
            meta = getattr(node, "metadata", None) or {}
            file_path = meta.get("file_path") or meta.get("filename")
            page_no = meta.get("page_no")
            if not file_path:
                continue
            label = f"{file_path}#p{page_no}" if page_no else str(file_path)
            if label in seen:
                continue
            sources.append(label)
            seen.add(label)
            if len(sources) >= (request.max_sources or 3):
                break
    return QueryResponse(answer=str(response), sources=sources)


@router.post("/extract_obligations", response_model=ExtractResponse)
def extract_obligations(request: ExtractRequest) -> ExtractResponse:
    """Keyword-only retrieval + LLM schema extraction. No heuristics."""
    if not os.getenv("TOGETHER_API_KEY"):
        raise HTTPException(status_code=500, detail="TOGETHER_API_KEY is not configured")

    nodes = _load_nodes_from_docstore(request.industry, request.regulator)
    bm25 = BM25Retriever.from_defaults(nodes=nodes)
    results = bm25.retrieve(request.query)

    max_return = request.max_return if request.max_return is not None else 30
    picked = results[: max_return]

    context_lines: List[str] = []
    used_sources: List[dict] = []
    for r in picked:
        n = r.node if hasattr(r, "node") else r
        meta = getattr(n, "metadata", None) or {}
        text = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
        if not text:
            continue
        header = f"[{meta.get('regulator', '?')} | {meta.get('filename', '?')} | p.{meta.get('page_no', '?')}]"
        context_lines.append(header + "\n" + text)
        used_sources.append(
            {
                "file": meta.get("file_path") or meta.get("filename"),
                "page": meta.get("page_no"),
                "industry": meta.get("industry"),
                "regulator": meta.get("regulator"),
            }
        )

    if not context_lines:
        return ExtractResponse()

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
    try:
        result = program(context="\n\n".join(context_lines))
        obligations = result.obligations
    except Exception:
        obligations = []

    return ExtractResponse(obligations=obligations, used_sources=used_sources[:20])
