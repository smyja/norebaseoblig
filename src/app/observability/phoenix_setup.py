from __future__ import annotations

import os
from typing import Optional


def enable_phoenix(*, host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Optionally launch Arize Phoenix UI and instrument LlamaIndex if available.

    Controlled by env var PHOENIX_ENABLED in {"1","true","yes"}.
    Best‑effort: if packages are missing, this is a no‑op.
    """

    enabled = os.getenv("PHOENIX_ENABLED", "false").lower() in {"1", "true", "yes"}
    if not enabled:
        return

    tracer_provider = None
    try:
        import phoenix as px  # type: ignore
        from phoenix.otel import register  # type: ignore

        kwargs = {}
        if host:
            kwargs["host"] = host
        if port:
            kwargs["port"] = port
        # Launch the Phoenix UI in the background
        px.launch_app(**kwargs)  # type: ignore[arg-type]

        # Configure an OTEL exporter pointed at Phoenix
        endpoint = os.getenv("PHOENIX_OTEL_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
        project = os.getenv("PHOENIX_PROJECT") or os.getenv("PHOENIX_PROJECT_NAME")
        try:
            tracer_provider = register(endpoint=endpoint, project_name=project)  # type: ignore[call-arg]
        except TypeError:
            # Older phoenix.otel.register versions may not accept project_name
            tracer_provider = register(endpoint=endpoint)
    except Exception:
        tracer_provider = None

    # Attempt OpenInference/OTel instrumentation for LlamaIndex (if installed)
    try:
        from openinference.instrumentation.llama_index import (  # type: ignore
            LlamaIndexInstrumentor,
        )

        if tracer_provider is not None:
            LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)
        else:
            LlamaIndexInstrumentor().instrument(skip_dep_check=True)
    except Exception:
        # If OpenInference instrumentation isn't available, skip silently.
        pass


def instrument_fastapi(app) -> None:
    """Optionally instrument a FastAPI app and common HTTP clients.

    Best‑effort and safe to call multiple times. Requires the following optional
    packages if you want HTTP/HTTPX/FastAPI spans:
      - opentelemetry-instrumentation-fastapi
      - opentelemetry-instrumentation-requests
      - opentelemetry-instrumentation-httpx
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore

        FastAPIInstrumentor().instrument_app(app)
    except Exception:
        pass
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor  # type: ignore

        RequestsInstrumentor().instrument()
    except Exception:
        pass
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # type: ignore

        HTTPXClientInstrumentor().instrument()
    except Exception:
        pass
