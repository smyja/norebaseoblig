from __future__ import annotations

import os
from pathlib import Path
import logging
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode
from ...schemas.autocomply import (
    ExtractRequest,
    ExtractResponse,
    ExtractionResult,
    Obligation,
)


router = APIRouter(prefix="/autocomply", tags=["autocomply"])
logger = logging.getLogger("autocomply.api")


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
    # Use request.similarity_top_k (default defined in schema) to control BM25 depth
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=request.similarity_top_k)
    results = bm25.retrieve(request.query)

    # Optional hybrid retrieval: combine vector results if an index is available
    use_hybrid = request.hybrid
    if use_hybrid is None:
        use_hybrid = os.getenv("AUTOCOMPLY_HYBRID", "true").lower() in {"1", "true", "yes"}
    if use_hybrid:
        try:
            index = _load_index(request.industry, request.regulator)
            vec = index.as_retriever(similarity_top_k=request.similarity_top_k)
            vec_results = vec.retrieve(request.query)
            results.extend(vec_results)
        except Exception:
            # If vector index isn't available, continue with BM25-only
            pass

    # Target penalties explicitly with a second retrieval pass
    penalty_query = (
        f"{request.query} penalty sanction fine contravention violation non-compliance liable revocation suspension"
    )
    try:
        penalty_results = bm25.retrieve(penalty_query)
        results.extend(penalty_results)
    except Exception:
        # If BM25 fails for any reason here, continue with the primary results
        pass

    # Prefer pages with normative cues ("shall", "must", etc.) and regulation-like filenames.
    def _normative_score(text: str, meta: dict) -> float:
        t = (text or "").lower()
        score = 0.0
        # Normative verbs/cues
        cues = [
            " shall ",
            " must ",
            " required to ",
            " are required to ",
            " is required to ",
            " shall not ",
            " must not ",
            " prohibited ",
            " ensure that ",
            " subject to ",
            " at least ",
            " no later than ",
            " within ",
        ]
        for cue in cues:
            if cue in t:
                score += 1.0

        # Penalty/violation cues get extra weight
        penalty_cues = [
            " penalty ",
            " penalties ",
            " sanction ",
            " sanctions ",
            " fine ",
            " fines ",
            " contravention ",
            " violation ",
            " non-compliance ",
            " noncompliance ",
            " liable ",
            " revocation ",
            " suspension ",
            " withdrawal of license ",
            " withdrawal of licence ",
        ]
        for cue in penalty_cues:
            if cue in t:
                score += 2.0

        # Filename heuristics
        fname = (meta.get("filename") or meta.get("file_path") or "").lower()
        if any(k in fname for k in ["guideline", "regulation", "act", "circular", "code_"]):
            score += 2.0
        if "financial statement" in fname or "financial_ statements" in fname or "statements_" in fname:
            score -= 2.0

        # Prefer pages nearer to the start (introductory/definitions often include scope/requirements)
        try:
            page_no = int(meta.get("page_no") or 0)
            if 1 <= page_no <= 5:
                score += 0.25
        except Exception:
            pass

        return score

    # Build sortable list with normative score; dedupe by node id
    seen_ids: set[str] = set()
    scored: List[Tuple[float, object]] = []
    for r in results:
        n = r.node if hasattr(r, "node") else r
        meta = getattr(n, "metadata", None) or {}
        text = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
        nid = getattr(n, "node_id", None) or getattr(n, "id_", None) or getattr(n, "id", None)
        if nid and nid in seen_ids:
            continue
        if nid:
            seen_ids.add(nid)
        base = getattr(r, "score", 0.0) or 0.0
        score = base + _normative_score(text, meta)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    max_return = request.max_return if request.max_return is not None else 30
    picked = [r for _, r in scored[: max_return]]

    context_lines: List[str] = []
    used_sources: List[dict] = []
    context_by_file_page: dict[tuple[str, int], str] = {}
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
        # accumulate raw text per (file basename, page) to refine quotes later
        try:
            fname = os.path.basename((meta.get("file_path") or meta.get("filename") or "").strip())
            page_no = int(meta.get("page_no") or 0)
            if fname and page_no:
                key = (fname, page_no)
                prev = context_by_file_page.get(key, "")
                context_by_file_page[key] = (prev + "\n\n" + text) if prev else text
        except Exception:
            pass

    if not context_lines:
        return ExtractResponse()

    instructions = (
        "You are an extraction assistant. From the context below, extract explicit compliance obligations and return them under the key 'obligations'. "
        "Each obligation must include: regulator, instrument_name (infer from filename if needed), instrument_type (Act, Regulation, Guideline, Circular, Code), citation (section/reg number), "
        "actor (who must comply), obligation_text, trigger (if conditional), deadline (if any), penalty (if any), effective_date (if stated), and source_file/source_page from metadata. "
        "Quote the exact obligation sentence/phrase containing 'shall', 'must', or equivalent where possible; otherwise provide a faithful minimal paraphrase (under 40 words). "
        "Explicitly extract penalties/sanctions (e.g., fine, penalty, sanctions, liable, contravention, violation, revocation, suspension). If the obligation itself states a consequence (e.g., 'transactions exceeding limits will not be executed'), copy that consequence into the penalty field. "
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
    try:
        # Optional LLM reranker over picked nodes before passing to the program
        use_reranker = request.use_llm_reranker
        if use_reranker is None:
            use_reranker = os.getenv("AUTOCOMPLY_USE_LLM_RERANKER", "false").lower() in {"1", "true", "yes"}
        if use_reranker:
            try:
                top_n = request.reranker_top_n or min(len(picked), 12)
                reranker = LLMRerank(choice_batch_size=5, top_n=top_n, llm=Settings.llm)
                # Build NodeWithScore-like list from picked
                reranked = reranker.postprocess_nodes(picked, query_str=request.query)  # type: ignore[arg-type]
                context_lines_r = []
                for rr in reranked:
                    nn = rr.node if hasattr(rr, "node") else rr
                    md = getattr(nn, "metadata", None) or {}
                    tx = nn.get_content() if hasattr(nn, "get_content") else getattr(nn, "text", "")
                    if not tx:
                        continue
                    header = f"[{md.get('regulator', '?')} | {md.get('filename', '?')} | p.{md.get('page_no', '?')}]"
                    context_lines_r.append(header + "\n" + tx)
                if context_lines_r:
                    context_payload = "\n\n".join(context_lines_r)
                else:
                    context_payload = "\n\n".join(context_lines)
            except Exception:
                context_payload = "\n\n".join(context_lines)
        else:
            context_payload = "\n\n".join(context_lines)

        result = program(context=context_payload, instructions=instructions)
        obligations = result.obligations
    except Exception as exc:  # noqa: BLE001
        logger.exception("obligation extraction failed: %s", exc)
        obligations = []

    # Fallback: ask the LLM for strict JSON and parse with Pydantic if nothing extracted
    if not obligations:
        try:
            raw_prompt = (
                "Return ONLY valid JSON matching this schema: {\"obligations\": ["
                "{\"regulator\": str|null, \"instrument_name\": str|null, \"instrument_type\": str|null, "
                "\"citation\": str|null, \"actor\": str|null, \"obligation_text\": str, \"trigger\": str|null, "
                "\"deadline\": str|null, \"penalty\": str|null, \"effective_date\": str|null, \"source_file\": str|null, \"source_page\": int|null}]}\n"
                "Use the context to extract explicit obligations; if none, return {\"obligations\": []}.\n\n"
                f"Context:\n{chr(10).join(context_lines)}"
            )

            resp = Settings.llm.complete(raw_prompt)  # type: ignore[attr-defined]
            txt = getattr(resp, "text", str(resp))
            # Trim common code fences
            txt = txt.strip()
            if txt.startswith("```"):
                txt = txt.strip("`\n ")
                if txt.lower().startswith("json"):
                    txt = txt[4:].lstrip("\n")
            # Try to locate JSON object
            first = txt.find("{")
            last = txt.rfind("}")
            if first != -1 and last != -1:
                txt = txt[first : last + 1]
            data = json.loads(txt)
            obligations = ExtractionResult(**data).obligations
        except Exception as exc:  # noqa: BLE001
            logger.info("fallback JSON parsing failed: %s", exc)
            # Keep obligations as []

    # LLM-based refinement: generic across sectors/regulators
    use_refine = request.refine_with_llm
    if use_refine is None:
        use_refine = os.getenv("AUTOCOMPLY_REFINE", "true").lower() in {"1", "true", "yes"}

    if use_refine and obligations:
        refine_instructions = (
            "Refine the draft obligation to match the context precisely. "
            "Quote the exact sentence(s) for obligation_text where possible. "
            "Populate penalty with the exact sanction or consequence text from the same page (e.g., 'will not be executed', 'liable to a fine …'). "
            "If no penalty is present in the context, set penalty to null. "
            "Do not invent or generalize beyond the provided page. Return only valid JSON matching the Obligation schema."
        )

        refine_template = PromptTemplate(
            "{instructions}\n\n"
            "Draft Obligation (JSON):\n{draft}\n\n"
            "Context (single page):\n{context}"
        )
        refiner = LLMTextCompletionProgram.from_defaults(
            output_cls=Obligation,
            prompt=refine_template,
            llm=Settings.llm,
            temperature=0.0,
            input_key="context",
        )

        new_list: List[Obligation] = []
        for ob in obligations:
            try:
                fname = os.path.basename((getattr(ob, "source_file", None) or "").strip())
                page_no = int(getattr(ob, "source_page", 0) or 0)
            except Exception:
                fname = ""
                page_no = 0
            ctx_text = context_by_file_page.get((fname, page_no), "") if fname and page_no else ""
            if not ctx_text:
                new_list.append(ob)
                continue
            try:
                draft_json = ob.json(exclude_none=True)  # pydantic v1
            except Exception:
                draft_json = json.dumps(getattr(ob, "__dict__", {}))
            try:
                refined = refiner(context=ctx_text, draft=draft_json, instructions=refine_instructions)
                # Ensure we have an Obligation instance
                if isinstance(refined, Obligation):
                    new_list.append(refined)
                else:
                    new_list.append(Obligation(**refined.dict()))
            except Exception as exc:  # noqa: BLE001
                logger.info("obligation refine skipped: %s", exc)
                new_list.append(ob)
        obligations = new_list

    return ExtractResponse(obligations=obligations, used_sources=used_sources[:20])
