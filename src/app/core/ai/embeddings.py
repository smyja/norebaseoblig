from __future__ import annotations

from typing import List, Optional
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from together import Together
from llama_index.embeddings.together import TogetherEmbedding


class BatchedTogetherEmbedding(TogetherEmbedding):
    """Together Embedding with server-side batching via Together SDK.

    This overrides TogetherEmbedding's per-text behavior to call the official
    Together embeddings endpoint with a list of inputs, reducing HTTP calls.
    """

    def __init__(
        self,
        model_name: str = "togethercomputer/m2-bert-80M-32k-retrieval",
        api_key: str | None = None,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        super().__init__(model_name=model_name, api_key=api_key)
        self._client = Together(api_key=api_key) if api_key else Together()
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        # progress tracking
        self._processed: int = 0
        self._total: Optional[int] = None
        self._start_ts: float = time.time()
        self._name: str = os.getenv("TOGETHER_EMBED_NAME", "embed")
        self._log_enabled: bool = os.getenv("TOGETHER_EMBED_PROGRESS", "1").lower() in {"1", "true", "yes"}
        # client-level timeout isn't supported via per-call arg in Together SDK; omit for now
        self._timeout: float = float(os.getenv("TOGETHER_EMBED_TIMEOUT", "60"))
        self._concurrency: int = max(1, int(os.getenv("TOGETHER_EMBED_CONCURRENCY", "1")))
        self._lock: Lock = Lock()

    def set_progress_total(self, total: int, name: Optional[str] = None) -> None:
        self._total = total
        self._processed = 0
        self._start_ts = time.time()
        if name:
            self._name = name

    def _log_batch(self, batch_len: int) -> None:
        if not self._log_enabled:
            return
        with self._lock:
            self._processed += batch_len
            processed = self._processed
        elapsed = time.time() - self._start_ts
        rate = processed / elapsed if elapsed > 0 else 0.0
        total = self._total
        if total:
            pct = (processed / total) * 100.0
            remaining = max(total - processed, 0)
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"[embeddings:{self._name}] {processed}/{total} ({pct:.1f}%) | "
                f"rate {rate:.1f}/s | elapsed {elapsed:.1f}s | eta {eta:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[embeddings:{self._name}] processed +{batch_len} | elapsed {elapsed:.1f}s | rate {rate:.1f}/s",
                flush=True,
            )

    # LlamaIndex internals call these underscored methods
    def _get_query_embedding(self, query: str) -> List[float]:  # type: ignore[override]
        for attempt in range(self._max_retries):
            try:
                resp = self._client.embeddings.create(model=self.model_name, input=query)
                emb = list(resp.data[0].embedding)
                self._log_batch(1)
                return emb
            except Exception:  # noqa: BLE001
                if attempt == self._max_retries - 1:
                    raise
                import time

                time.sleep(self._retry_backoff * (attempt + 1))
        return []

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        n = len(texts)
        out: List[Optional[List[float]]] = [None] * n

        # Prepare segments
        segments = [(i, texts[i : i + self._batch_size]) for i in range(0, n, self._batch_size)]

        def run_segment(start: int, segment: List[str]) -> None:
            for attempt in range(self._max_retries):
                try:
                    resp = self._client.embeddings.create(model=self.model_name, input=segment)
                    embs = [list(item.embedding) for item in resp.data]
                    for j, emb in enumerate(embs):
                        out[start + j] = emb
                    self._log_batch(len(embs))
                    return
                except Exception:  # noqa: BLE001
                    if attempt == self._max_retries - 1:
                        raise
                    time.sleep(self._retry_backoff * (attempt + 1))

        if self._concurrency <= 1:
            for start, segment in segments:
                run_segment(start, segment)
        else:
            with ThreadPoolExecutor(max_workers=self._concurrency) as ex:
                futures = [ex.submit(run_segment, start, segment) for start, segment in segments]
                for _ in as_completed(futures):
                    pass

        # Fill any missing (shouldn't happen unless persistent failures)
        return [e if e is not None else [] for e in out]
