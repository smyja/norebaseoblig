# LlamaIndex Autocomply Pipeline

This repository now ships with a three step ingestion stack that turns the PDF corpus under `regulations_corpus/` into a hybrid searchable index and exposes a FastAPI microservice for question answering and obligation extraction.

## 1. Prerequisites

- Python 3.10+
- Tesseract OCR installed locally (`brew install tesseract` on macOS, `sudo apt-get install tesseract-ocr` on Debian/Ubuntu)
- Environment variable `TOGETHER_API_KEY` defined (either exported or stored in `.env`)

Install the required libraries (Poetry and pip will pick these up via `pyproject.toml`):

```bash
poetry install
```

If you manage dependencies manually, install:

```bash
pip install pymupdf pytesseract pillow unstructured[local-inference] \
    llama-index==0.11.* llama-index-retrievers-bm25
```

## 2. Parse the corpus with OCR fallback

```bash
python parse_corpus_with_ocr.py
```

The script walks `regulations_corpus/<industry>/<regulator>/pdf`, merges metadata from any sibling `manifest.jsonl`, and writes one JSONL file per regulator under `parsed_pages/`. Image-based pages are rendered and sent through Tesseract automatically.

## 3. Build the persisted LlamaIndex store

```bash
python build_llamaindex.py
```

This loads the parsed pages, splits them into sentence chunks (1,500 characters with 150 overlap), configures Together embeddings + LLM, and persists the resulting index to `storage_autocomply/`.

## 4. Serve hybrid retrieval + obligation extraction

Prefer the built-in API router (mounted under `/api/v1/autocomply`) in this FastAPI app:

```bash
uvicorn src.app.main:app --port 8002 --reload
```

Endpoints (via the app router):

- `POST /api/v1/autocomply/chat` — general Q&A with optional `industry` / `regulator` filters.
- `POST /api/v1/autocomply/extract_obligations` — schema-guided extraction over top-k retrieved chunks.

If you want a minimal standalone server for only this feature, you can still run `serve_autocomply.py`:

```bash
uvicorn serve_autocomply:app --port 8003 --reload
```

### Optional: Per-regulator graph (hierarchical routing)

Build a set of indices per regulator and a registry file:

```bash
python build_llamaindex_graph.py
```

This writes one index per regulator under `storage_autocomply/indices/<industry>__<regulator>/` and a `storage_autocomply/registry.json` manifest. To enable hierarchical fan-out routing in the API, set:

```bash
export AUTOCOMPLY_USE_GRAPH=true
```

By default the router will query a handful of child indices and merge results; provide `industry` / `regulator` in your request to constrain the fan-out.

Example queries:

```bash
curl -s localhost:8002/api/v1/autocomply/chat -X POST -H "Content-Type: application/json" \
    -d '{"question":"what are VASP banking account obligations under CBN"}' | jq

curl -s localhost:8002/api/v1/autocomply/extract_obligations -X POST -H "Content-Type: application/json" \
    -d '{"query":"list obligations, deadlines and penalties on call quality", "industry":"telecoms", "regulator":"NCC"}' | jq
```

## 5. Next steps

- The index currently persists via the default storage layer. If you need shared durability, swap in a `PGVectorStore` or similar and the rest of the stack can remain unchanged.
- To experiment with LlamaIndex graphs, build specialised indices per regulator and combine them via `ComposableGraph` (the existing scripts already produce per-regulator JSONL files, making this straightforward).
