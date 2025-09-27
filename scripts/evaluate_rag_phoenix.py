#!/usr/bin/env python3
"""Phoenix-style RAG evaluation adapted to this codebase and Together.

This script mirrors the Arize Phoenix tutorial but runs against our
persisted indices and retrieval stack. It:
  - Launches Phoenix UI + instruments LlamaIndex via OpenInference
  - Routes Phoenix Evals (OpenAIModel) to Together via OPENAI_BASE_URL
  - Executes a set of queries through LlamaIndex to produce traces
  - Pulls retriever spans and runs relevance evals
  - Computes simple retrieval metrics (P@2, NDCG@2) and prints aggregates

Install:
  pip install -e .[eval]

Run:
  export TOGETHER_API_KEY=...
  export PHOENIX_ENABLED=true
  export PHOENIX_PORT=6006
  python scripts/evaluate_rag_phoenix.py --industry banking_fintech --regulator CBN \
      --queries examples/queries.txt --top-k 10 \
      --eval-model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv


def _load_env() -> None:
    load_dotenv()
    load_dotenv("src/.env.local", override=False)
    load_dotenv("src/.env", override=False)


def _read_lines(path: Path | None) -> List[str]:
    if not path:
        return [
            "FX reporting obligations for banks",
            "penalties for non-compliance in EFEMS",
            "timelines for settlement of FX trades",
        ]
    if not path.exists():
        raise SystemExit(f"queries file not found: {path}")
    return [s for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _compute_ndcg_at_k(scores: List[float], doc_scores: List[float], k: int = 2) -> float:
    # Simple NDCG@k with binary/continuous relevance, handling short lists
    import math

    k = max(1, k)
    n = min(k, len(scores), len(doc_scores))
    if n == 0:
        return 0.0
    # DCG: use model eval order
    dcg = 0.0
    for i in range(n):
        dcg += (2**scores[i] - 1) / math.log2(i + 2)
    # IDCG: ideal order (sort by doc_scores)
    ideal = sorted(zip(scores, doc_scores), key=lambda x: x[1], reverse=True)[:n]
    idcg = 0.0
    for i, (s, _) in enumerate(ideal):
        idcg += (2**s - 1) / math.log2(i + 2)
    return float(dcg / idcg) if idcg > 0 else 0.0


def main() -> None:
    _load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default=os.getenv("EVAL_INDUSTRY"))
    ap.add_argument("--regulator", default=os.getenv("EVAL_REGULATOR"))
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--eval-model",
        default=os.getenv("EVAL_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
        help="Judge LLM for Phoenix Evals (routed to Together via OPENAI_BASE_URL)",
    )
    ap.add_argument(
        "--answer-models",
        default=os.getenv("ANSWER_MODELS"),
        help="Comma-separated Together model ids to generate answers for comparison; defaults to eval-model",
    )
    args = ap.parse_args()

    # Route OpenAI client usage (inside Phoenix Evals) to Together
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        raise SystemExit("TOGETHER_API_KEY missing; export it or set in src/.env.local")
    os.environ["OPENAI_API_KEY"] = together_key
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.together.xyz/v1")

    try:
        import phoenix as px  # type: ignore
        from phoenix.client import Client  # type: ignore
        from phoenix.client.types.spans import SpanQuery  # type: ignore
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor  # type: ignore
        from phoenix.otel import register  # type: ignore
        from phoenix.evals import (
            RelevanceEvaluator,
            QAEvaluator,
            HallucinationEvaluator,
            run_evals,
            OpenAIModel,
        )  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Missing Phoenix packages. Install with `pip install -e .[eval]` ({exc})"
        )

    # Launch Phoenix and instrument with an OTEL exporter to its endpoint
    port = int(os.getenv("PHOENIX_PORT", "6006"))
    px.launch_app(port=port)
    tracer_provider = register(endpoint=f"http://127.0.0.1:{port}/v1/traces")
    instrumented = True
    try:
        LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)
    except Exception as exc:  # noqa: BLE001
        instrumented = False
        print(f"[warn] LlamaIndex instrumentation disabled (compat issue): {exc}")

    # Run queries against our pipeline to generate traces
    from src.app.api.v1.autocomply import _load_index
    from llama_index.core import Settings
    from llama_index.llms.together import TogetherLLM

    industry = args.industry
    regulator = args.regulator
    top_k = int(args.top_k)
    questions = _read_lines(args.queries)

    index = _load_index(industry, regulator)

    # Build list of answer models (Together model ids)
    answer_models = [m.strip() for m in (args.answer_models.split(",") if args.answer_models else []) if m.strip()]
    if not answer_models:
        answer_models = [args.eval_model]

    # Collect QA rows per model and map to retriever span ids for annotation
    qa_rows: List[Dict[str, Any]] = []
    span_ids: List[str] = []

    def latest_retriever_span_id(client: "Client", text: str) -> str | None:
        try:
            df = client.spans.get_spans_dataframe(
                query=(SpanQuery().where("span_kind == 'RETRIEVER'").select("span_id", "attributes.input.value"))
            )
            sub = df[df["input.value"] == text]
            if sub.empty:
                return None
            return str(sub.iloc[-1]["span_id"])  # type: ignore[index]
        except Exception:
            return None

    for model_id in answer_models:
        # Use a Together LLM for answering; retrieval stays the same
        try:
            answer_llm = TogetherLLM(model=model_id, api_key=together_key)
        except Exception:
            # Fallback to default if TogetherLLM import fails
            answer_llm = Settings.llm
        qe = index.as_query_engine(similarity_top_k=top_k, llm=answer_llm)
        for q in questions:
            resp = qe.query(q)
            # Build reference context from source nodes
            contexts: List[str] = []
            try:
                for sn in getattr(resp, "source_nodes", []) or []:
                    if hasattr(sn, "node"):
                        txt = sn.node.get_content()
                    else:
                        txt = sn.get_content()
                    if txt:
                        contexts.append(str(txt))
            except Exception:
                pass
            reference = "\n\n".join(contexts)[:20000]
            qa_rows.append({
                "input": q,
                "reference": reference,
                "output": str(resp),
                "model": model_id,
            })
            try:
                client_tmp = Client()
                sid = latest_retriever_span_id(client_tmp, q)
            except Exception:
                sid = None
            span_ids.append(sid or "")

    # Pull retriever spans with documents and run relevance evals
    client = Client()
    import pandas as pd
    if instrumented:
        from openinference.semconv.trace import DocumentAttributes, SpanAttributes  # type: ignore

        retrieved_df = client.spans.get_spans_dataframe(
            query=(
                SpanQuery()
                .where("span_kind == 'RETRIEVER'")
                .select("trace_id", SpanAttributes.INPUT_VALUE)
                .explode(
                    SpanAttributes.RETRIEVAL_DOCUMENTS,
                    reference=DocumentAttributes.DOCUMENT_CONTENT,
                    document_score=DocumentAttributes.DOCUMENT_SCORE,
                )
            )
        )
        if retrieved_df.empty:
            print("No retriever spans found. Ensure instrumentation is active and queries ran.")
        else:
            retrieved_df.rename(columns={"input.value": "input"}, inplace=True)
            evaluator = RelevanceEvaluator(OpenAIModel(model=args.eval_model))
            eval_df = run_evals(
                evaluators=[evaluator], dataframe=retrieved_df, provide_explanation=False, concurrency=10
            )[0]
            merged = pd.concat([retrieved_df, eval_df.add_prefix("eval_")], axis=1)

            def precision_at_2(group: "pd.DataFrame") -> float:
                return float(group.eval_score.iloc[:2].sum(skipna=False) / 2)

            def ndcg_at_2(group: "pd.DataFrame") -> float:
                scores = [float(x) if x == x else 0.0 for x in group.eval_score.tolist()[:10]]
                doc_scores = [float(x) if x == x else 0.0 for x in group.document_score.tolist()[:10]]
                return _compute_ndcg_at_k(scores, doc_scores, k=2)

            grouped = merged.groupby("context.span_id")
            p_at_2 = grouped.apply(precision_at_2)
            n_at_2 = grouped.apply(ndcg_at_2)
            hit = grouped.apply(lambda g: bool((g.eval_score.iloc[:2].sum(skipna=False) or 0) > 0))

            print("Retrieval metrics (aggregated):")
            print(f"  Precision@2: {float(p_at_2.mean(skipna=True)):.3f}")
            print(f"  NDCG@2:      {float(n_at_2.mean(skipna=True)):.3f}")
            print(f"  Hit rate:    {float(hit.mean()):.3f}")
    else:
        print("[info] Skipping Phoenix-based retrieval evals (no instrumentation). Running only response evals.")

    # Response evaluation per answer model (optional)
    try:
        import pandas as pd

        qa_df = pd.DataFrame(qa_rows)
        if not qa_df.empty:
            qa_df["context.span_id"] = span_ids

            qa_eval = QAEvaluator(OpenAIModel(model=args.eval_model))
            hall_eval = HallucinationEvaluator(OpenAIModel(model=args.eval_model))
            qa_res_df, hall_res_df = run_evals(
                evaluators=[qa_eval, hall_eval],
                dataframe=qa_df.rename(columns={"input": "question"}).rename(columns={"question": "input"}),
                provide_explanation=False,
                concurrency=10,
            )

            # Aggregate metrics by model
            print("\nResponse metrics by model:")
            for model_id in answer_models:
                mask = qa_df["model"] == model_id
                qam = qa_res_df[mask.values]
                hm = hall_res_df[mask.values]
                qa_mean = float(qam["score"].mean(skipna=True)) if not qam.empty else float("nan")
                hall_mean = float(hm["score"].mean(skipna=True)) if not hm.empty else float("nan")
                print(f"  {model_id}: QA_correctness={qa_mean:.3f}, Hallucination={hall_mean:.3f}")

            # Log span annotations to Phoenix for per-query inspection
            try:
                from phoenix.client.__generated__ import v1  # type: ignore

                ann_qac: List[v1.SpanAnnotationData] = []
                ann_hall: List[v1.SpanAnnotationData] = []
                qa_res_reset = qa_res_df.reset_index(drop=True)
                hall_res_reset = hall_res_df.reset_index(drop=True)

                for i in range(len(qa_df)):
                    span_id = str(qa_df.iloc[i].get("context.span_id", ""))
                    model_id = str(qa_df.iloc[i].get("model", ""))
                    try:
                        score_q = qa_res_reset.iloc[i].get("score")
                        score_q = float(score_q) if score_q == score_q else None
                    except Exception:
                        score_q = None
                    try:
                        score_h = hall_res_reset.iloc[i].get("score")
                        score_h = float(score_h) if score_h == score_h else None
                    except Exception:
                        score_h = None
                    ann_qac.append(
                        v1.SpanAnnotationData(
                            name=f"Q&A Correctness ({model_id})",
                            span_id=span_id,
                            annotator_kind="LLM",
                            result={**({"score": score_q} if score_q is not None else {}), "model": model_id},
                        )
                    )
                    ann_hall.append(
                        v1.SpanAnnotationData(
                            name=f"Hallucination ({model_id})",
                            span_id=span_id,
                            annotator_kind="LLM",
                            result={**({"score": score_h} if score_h is not None else {}), "model": model_id},
                        )
                    )
                # New API: log via client.spans
                try:
                    client.spans.log_span_annotations(span_annotations=ann_qac, sync=False)
                    client.spans.log_span_annotations(span_annotations=ann_hall, sync=False)
                except Exception:
                    # Fallback for older client versions
                    client.annotations.log_span_annotations(span_annotations=ann_qac, sync=False)
                    client.annotations.log_span_annotations(span_annotations=ann_hall, sync=False)
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] response evaluation skipped: {exc}")


if __name__ == "__main__":
    main()
