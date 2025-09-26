

Autocomply turns unstructured regulatory PDFs into searchable chunks and structured “obligation” objects, with a small FastAPI layer to query and extract results. It uses:

- LlamaIndex for chunking, storage, retrieval, and LLM programs
- Together AI for LLM and embeddings (with efficient batched embeddings)
- A lightweight OCR pipeline to reliably parse PDFs
- A property graph (optional) to explore entities and relations

This doc is a tour of the architecture, setup, and API, with code snippets and tips.

## What's Inside

- Parsing and OCR: Robust page‑level text extraction from PDFs with PyMuPDF and Tesseract.
- Indexing: Per‑regulator vector indices persisted under `storage_autocomply/indices/<industry>__<regulator>` plus a simple registry.
- Retrieval: Fast keyword retrieval (BM25) by default, optional vector/hybrid retrieval and LLM reranking.
- Extraction: A structured “obligation” schema with Pydantic models and an LLM program. Includes a strict JSON fallback and a context‑aware refinement step.
- API: Two endpoints — `/autocomply/chat` (Q&A with sources) and `/autocomply/extract_obligations` (schema extraction with sources).
- Knowledge Graph: Optional property graph built from parsed pages for downstream exploration.
- Centralized Schemas: Shared Pydantic models in `src/app/schemas/autocomply.py` used by the API.
- Efficient Embeddings: A batched Together embeddings client to cut HTTP overhead.

## Quickstart

Prereqs

- Python 3.11
- System Tesseract (for OCR fallback): macOS `brew install tesseract`, Ubuntu `sudo apt-get install tesseract-ocr`
- A Together API key

Setup

- Create a virtual environment
- Install dependencies: `pip install -e .` (or `poetry install` if you use Poetry)
- Copy `src/.env.example` to `src/.env.local` and set at least:
  - `TOGETHER_API_KEY=<your-key>`
  - `TOGETHER_EMBED_MODEL=togethercomputer/m2-bert-80M-32k-retrieval` (or any supported model)
  - `TOGETHER_LLM_MODEL=moonshotai/Kimi-K2-Instruct-0905` (or any supported model)

Build the artifacts

- Parse PDFs into page JSONL files: `python parse_corpus_with_ocr.py`
- Build per‑regulator indices + registry: `python build_llamaindex_graph.py`
- (Optional) Build a property knowledge graph: `python build_knowledge_graph.py`

Run an API

- One‑file server: `uvicorn serve_autocomply:app --reload`
- Or include the router in your FastAPI app:

```python
from fastapi import FastAPI
from src.app.api.v1.autocomply import router as autocomply_router

app = FastAPI()
app.include_router(autocomply_router)
```

## API Overview

Chat

- POST `/autocomply/chat`
- Request: `QueryRequest`
- Response: `QueryResponse`

Example

```bash
curl -X POST http://localhost:8000/autocomply/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What are FX reporting obligations for banks?",
    "industry": "banking_fintech",
    "regulator": "CBN",
    "similarity_top_k": 12,
    "use_llm_reranker": true,
    "reranker_top_n": 5,
    "max_sources": 3,
    "keyword_only": true
  }'
```

Extract Obligations

- POST `/autocomply/extract_obligations`
- Request: `ExtractRequest`
- Response: `ExtractResponse`

```bash
curl -X POST http://localhost:8000/autocomply/extract_obligations \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "settlement timelines for FX transactions",
    "industry": "banking_fintech",
    "regulator": "CBN",
    "similarity_top_k": 12,
    "max_return": 30,
    "hybrid": true,
    "use_llm_reranker": true,
    "reranker_top_n": 8,
    "refine_with_llm": true
  }'
```

Response shape (truncated):

```json
{
  "obligations": [
    {
      "regulator": "CBN",
      "instrument_name": "Guidelines for the Electronic Foreign Exchange Matching System",
      "instrument_type": "Guideline",
      "citation": "s.5(2)",
      "actor": "Authorized dealers",
      "obligation_text": "Authorized dealers shall report ...",
      "trigger": null,
      "deadline": "within T+1",
      "penalty": "... liable to sanctions ...",
      "effective_date": null,
      "source_file": ".../Guidelines_for_the_Electronic_Foreign_Exchange_Matching_System_EFEMS.pdf",
      "source_page": 12
    }
  ],
  "used_sources": [
    { "file": ".../EFEMS.pdf", "page": 12, "industry": "banking_fintech", "regulator": "CBN" }
  ]
}
```

## Schemas

All request/response models live in `src/app/schemas/autocomply.py` and are imported by the API.

```python
# src/app/schemas/autocomply.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Obligation(BaseModel):
    industry: Optional[str] = None
    regulator: Optional[str] = None
    instrument_name: Optional[str] = None
    instrument_type: Optional[str] = None
    citation: Optional[str] = None
    actor: Optional[str] = None
    obligation_text: str
    trigger: Optional[str] = None
    deadline: Optional[str] = None
    penalty: Optional[str] = None
    effective_date: Optional[str] = None
    source_file: Optional[str] = None
    source_page: Optional[int] = None

class ExtractionResult(BaseModel):
    obligations: List[Obligation] = Field(default_factory=list)

class QueryRequest(BaseModel):
    question: str
    similarity_top_k: Optional[int] = 10
    reranker_top_n: Optional[int] = 3
    use_llm_reranker: Optional[bool] = False
    max_sources: Optional[int] = 3
    industry: Optional[str] = None
    regulator: Optional[str] = None
    keyword_only: Optional[bool] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)

class ExtractRequest(BaseModel):
    query: str
    similarity_top_k: int = 12
    max_return: int = 30
    industry: Optional[str] = None
    regulator: Optional[str] = None
    use_llm_reranker: Optional[bool] = None
    reranker_top_n: Optional[int] = None
    hybrid: Optional[bool] = None
    refine_with_llm: Optional[bool] = None

class ExtractResponse(BaseModel):
    obligations: List[Obligation] = Field(default_factory=list)
    used_sources: List[Dict[str, Any]] = Field(default_factory=list)
```

## Retrieval and Index Resolution

We persist indices under a root directory with optional child sub‑indices and select one at request time.

```python
# src/app/api/v1/autocomply.py:49
# Preference order:
#   1) AUTOCOMPLY_CHILD_INDEX
#   2) storage_autocomply/indices/<industry>__<regulator>
#   3) storage_autocomply
```

```python
# src/app/api/v1/autocomply.py:49
def _resolve_persist_dir(industry, regulator) -> Path:
    root_dir = Path(os.getenv("AUTOCOMPLY_STORAGE", "storage_autocomply"))
    child_name = os.getenv("AUTOCOMPLY_CHILD_INDEX")
    candidate_dirs = []
    if child_name:
        candidate_dirs.append(root_dir / "indices" / child_name)
    if industry and regulator:
        candidate_dirs.append(root_dir / "indices" / f"{industry}__{regulator}")
    candidate_dirs.append(root_dir)
    for pd in candidate_dirs:
        if pd.exists():
            return pd
    ...
```

For fast keyword retrieval over all nodes in the selected store we read `docstore.json` directly:

```python
# src/app/api/v1/autocomply.py:90
with (persist_dir / "docstore.json").open("r") as f:
    data = json.load(f)
records = data.get("docstore/data", {})
for nid, wrapper in records.items():
    payload = wrapper.get("__data__", {})
    text = payload.get("text") or ""
    md = payload.get("metadata") or {}
    nodes.append(TextNode(text=text, metadata=md, id_=nid))
```

# BM25 + Optional Reranker

By default we run BM25 over all nodes and optionally LLM‑rerank the results.

```python
# src/app/api/v1/autocomply.py:138
bm25 = BM25Retriever.from_defaults(nodes=nodes)
results = bm25.retrieve(request.question)
if request.use_llm_reranker:
    reranker = LLMRerank(top_n=request.reranker_top_n or 3, llm=Settings.llm)
    results = reranker.postprocess_nodes(results, query_str=request.question)
```

## Obligation Extraction: Scoring, Program, Fallback, Refinement

Scoring (prioritize normative language and penalty cues; de‑prioritize irrelevant pages):

```python
# src/app/api/v1/autocomply.py:249
cues = [" shall ", " must ", " required to ", ...]
penalty_cues = [" penalty ", " sanction ", " fine ", ...]
score = base + _normative_score(text, meta)
```

LLM Program (Pydantic‑typed output):

```python
# src/app/api/v1/autocomply.py:365
program = LLMTextCompletionProgram.from_defaults(
    output_cls=ExtractionResult,
    prompt=PromptTemplate("{instructions}\n\nContext:\n{context}"),
    llm=Settings.llm,
    temperature=0.0,
    input_key="context",
)
result = program(context=context_payload, instructions=instructions)
obligations = result.obligations
```

Strict JSON fallback if nothing extracted:

```python
# src/app/api/v1/autocomply.py:430
resp = Settings.llm.complete(raw_prompt)
txt = getattr(resp, "text", str(resp))
first, last = txt.find("{"), txt.rfind("}")
if first != -1 and last != -1:
    txt = txt[first:last+1]
obligations = ExtractionResult(**json.loads(txt)).obligations
```

Context‑aware refinement of each obligation:

```python
# src/app/api/v1/autocomply.py:466
refiner = LLMTextCompletionProgram.from_defaults(
    output_cls=Obligation,
    prompt=PromptTemplate("{instructions}\n\nDraft Obligation (JSON):\n{draft}\n\nContext (single page):\n{context}"),
    llm=Settings.llm,
    temperature=0.0,
)
refined = refiner(context=ctx_text, draft=ob.json(exclude_none=True), instructions=refine_instructions)
```

## Batched Embeddings

We replace per‑text embedding calls with a batched Together client to reduce latency and cost.

```python
# src/app/core/ai/embeddings.py
class BatchedTogetherEmbedding(TogetherEmbedding):
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        segments = [(i, texts[i:i+self._batch_size]) for i in range(0, n, self._batch_size)]
        resp = self._client.embeddings.create(model=self.model_name, input=segment)
        ...
```

## Property Knowledge Graph (Optional)

We can also extract a lightweight property graph of regulators, instruments, obligations, citations, etc., from parsed pages and persist it to `storage_autocomply/kg/`.

```bash
python build_knowledge_graph.py
```

Under the hood, we use `PropertyGraphIndex` + `SchemaLLMPathExtractor` with a constrained schema for entity and relation types.

## Configuration

Common environment variables (set in `src/.env.local` or the environment):

- `TOGETHER_API_KEY` — required
- `TOGETHER_EMBED_MODEL` — Together embeddings model name
- `TOGETHER_EMBED_BATCH` — server‑side batch size for embeddings
- `TOGETHER_EMBED_CONCURRENCY` — parallel embedding requests
- `TOGETHER_LLM_MODEL` — Together LLM model name
- `AUTOCOMPLY_STORAGE` — index root directory (default `storage_autocomply`)
- `AUTOCOMPLY_CHILD_INDEX` — force a specific child index (`<industry>__<regulator>`)
- `AUTOCOMPLY_KEYWORD_ONLY` — default chat retrieval mode (true = BM25)
- `AUTOCOMPLY_HYBRID` — include vector retrieval in extraction
- `AUTOCOMPLY_REFINE` — run the context‑aware refinement step
- `AUTOCOMPLY_SKIP_TOC` — skip obvious Table of Contents pages when parsing

## Notes and Next Steps

- Centralized schemas: API imports `QueryRequest/QueryResponse/Extract*` from `src/app/schemas/autocomply.py` (no duplication).
- The extraction pipeline is conservative: if uncertain, returns no obligations. Tuning cues and thresholds can shift recall vs. precision.
- Possible extensions: cross‑page coalescing of obligations, richer source metadata, de‑duplication across regulators, and KG‑backed query workflows.

