# Evaluations and Tracing with Arize Phoenix on Together

This guide shows how to run the Arize Phoenix evaluation tutorial end‑to‑end using Together models (no OpenAI account required), and how to keep Phoenix/Ragas intact.

Two clean approaches:
- Fastest: keep the OpenAI client but point it at Together’s OpenAI‑compatible base URL.
- Cleaner for Ragas: use Together’s official LangChain integration and pass it through Ragas wrappers.

## Install (as project optional deps)

- With extras: `pip install -e .[eval]`
- Or directly: `pip install -U phoenix ragas langchain-together llama-index-embeddings-langchain`

You’ll also need your Together API key: `export TOGETHER_API_KEY=...`

Optional project scoping

- Phoenix organizes runs under a "project". If you don’t set one, it uses `default`.
- To route traces into a named project, set an env var before launch:
  - `export PHOENIX_PROJECT=autocomply-evals`  (or `PHOENIX_PROJECT_NAME`)
- Our `enable_phoenix` helper passes this to Phoenix (when supported), so runs show up under that project in the UI.

## 1) Fastest path: keep `openai` but route to Together

Together’s API is OpenAI‑compatible. Set the base URL and key, then use Together model ids.

```python
# before: the blog sets OPENAI_API_KEY and uses OpenAI defaults
# after: point OpenAI client to Together

import os
from openai import OpenAI

os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY") or "paste-your-key"
client = OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

# example chat call used anywhere the notebook calls gpt-3.5-turbo
resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "ping"}],
)
print(resp.choices[0].message.content)
```

Notes

- Replace any `"gpt-3.5-turbo"` or `"gpt-4*"` strings with Together models (e.g., `"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"`, `"Qwen/Qwen2.5-72B-Instruct-Turbo"`).
- Embeddings are also OpenAI‑compatible at Together. Use an embedding like `togethercomputer/m2-bert-80M-8k-retrieval` via the same base URL.

LlamaIndex pieces in the blog can also be pointed to Together via the OpenAI wrappers:

```python
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

llm = LlamaOpenAI(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key=os.environ["TOGETHER_API_KEY"],
    api_base="https://api.together.xyz/v1",
)

embed = OpenAIEmbedding(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    api_key=os.environ["TOGETHER_API_KEY"],
    api_base="https://api.together.xyz/v1",
)
```

## 2) Cleaner for Ragas: Together + LangChain + Ragas wrappers

The blog often uses `TestsetGenerator.with_openai()` and Ragas defaults that expect OpenAI. Swap in LangChain’s Together client and wrap it for Ragas.

```python
# pip install -U ragas langchain-together
from ragas.llms import LangchainLLMWrapper
from langchain_together import ChatTogether
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# Use Together for both generator and critic
gen_llm = LangchainLLMWrapper(
    ChatTogether(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", temperature=0.2)
)
critic_llm = LangchainLLMWrapper(
    ChatTogether(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", temperature=0.0)
)

generator = TestsetGenerator(
    generator_llm=gen_llm,
    critic_llm=critic_llm,
)

# Example: generate RAG testset from LlamaIndex Documents (adjust for your inputs)
distribution = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}
# documents: List[llama_index.core.Document]
# testset = generator.generate_with_llamaindex_docs(documents, test_size=25, distributions=distribution)
```

Run Ragas metrics with your Together LLM so scoring also uses Together:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_recall, context_precision

# ragas_eval_dataset: created from your run (predictions, contexts, references)
# gen_llm from above is a BaseRagasLLM via LangchainLLMWrapper
result = evaluate(
    dataset=ragas_eval_dataset,
    metrics=[faithfulness, answer_correctness, context_recall, context_precision],
    llm=gen_llm,
)
print(result)
```

### Together embeddings via LangChain, then into LlamaIndex

```python
# pip install -U langchain-together llama-index-embeddings-langchain
from langchain_together.embeddings import TogetherEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

lc_embed = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
li_embed = LangchainEmbedding(lc_embed)  # usable anywhere LlamaIndex expects an embed model
```

## Phoenix notes

- Phoenix tracing and the notebook structure remain provider‑agnostic. Keep the same Phoenix setup from the blog; only swap LLM/embeddings to Together as above.
- If the blog uses LlamaIndex’s Phoenix integration, continue using it unchanged. Your underlying LLM calls (now Together) will still be traced.

## Quick migration checklist

1. Replace the OpenAI client setup with Together’s OpenAI‑compatible `base_url` and key.
2. Swap model names to Together models.
3. For test set generation/evaluation, use `LangchainLLMWrapper(ChatTogether(...))` instead of `with_openai()` so Ragas uses Together.
4. Use Together embeddings either via OpenAI‑compatible embeddings or via `TogetherEmbeddings`.

That’s it — Phoenix, Ragas, and your notebooks should run on Together models without structural changes.

## Run a quick faithfulness eval on this repo

We include a small helper script that uses the project’s retrieval/LLM stack and computes Ragas faithfulness:

```bash
pip install -e .[eval]
export TOGETHER_API_KEY=...  # required

# Optional UI + tracing
export PHOENIX_ENABLED=true
export PHOENIX_PORT=6006

python scripts/evaluate_faithfulness.py \
  --industry banking_fintech \
  --regulator CBN \
  --top-k 10 \
  --queries examples/queries.txt  # optional
```

The script mirrors the `/autocomply/chat` BM25 path to build contexts and runs `ragas.metrics.faithfulness`.

## Compare models with Phoenix Evals

Use the Phoenix-style script to run queries through your own index, compute retrieval metrics, and evaluate responses (QA correctness + hallucination) with a judge LLM. You can compare multiple answer models side-by-side and log annotations back to Phoenix.

```bash
pip install -e .[eval]
export TOGETHER_API_KEY=...
export PHOENIX_ENABLED=true
export PHOENIX_PORT=6006

python scripts/evaluate_rag_phoenix.py \
  --industry banking_fintech \
  --regulator CBN \
  --queries examples/queries.txt \
  --top-k 10 \
  --eval-model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
  --answer-models meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo,Qwen/Qwen2.5-72B-Instruct-Turbo
```

What you get:
- Retrieval metrics (Precision@2, NDCG@2, hit rate) across queries
- Response metrics by model: QA correctness and hallucination (judge LLM is `--eval-model`)
- Span annotations per query in Phoenix named `Q&A Correctness (<model>)` and `Hallucination (<model>)` for visual comparison
