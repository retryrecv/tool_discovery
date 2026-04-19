"""FastAPI service exposing `POST /route`.

Run with::

    uv run uvicorn tool_index.router.service:app --reload

Env::

    TOOL_INDEX_SNAPSHOTS_ROOT  default: data/snapshots
    TOOL_INDEX_DEFAULT_BEAM    default: 2
    TOOL_INDEX_DEFAULT_K       default: 10

Dependencies (add to project if not present)::

    uv add fastapi uvicorn

The embedder is constructed lazily so importing this module doesn't
require a live Azure OpenAI key (helpful for tests).
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .registry import SnapshotRegistry
from .telemetry import RequestLogger, RouteRecord
from ..retrieval.traverser import retrieve_with_path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "router service requires `fastapi` and `pydantic`. "
        "Install with: uv add fastapi uvicorn"
    ) from e


SNAPSHOTS_ROOT = os.environ.get("TOOL_INDEX_SNAPSHOTS_ROOT", "data/snapshots")
DEFAULT_K = int(os.environ.get("TOOL_INDEX_DEFAULT_K", "10"))
DEFAULT_BEAM = int(os.environ.get("TOOL_INDEX_DEFAULT_BEAM", "2"))


_registry: Optional[SnapshotRegistry] = None
_logger: Optional[RequestLogger] = None
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from ..providers import make_embedding
        _embedder = make_embedding("azure_openai", dim=3072)
    return _embedder


def _get_registry() -> SnapshotRegistry:
    global _registry
    if _registry is None:
        _registry = SnapshotRegistry(SNAPSHOTS_ROOT)
    return _registry


def _get_logger() -> RequestLogger:
    global _logger
    if _logger is None:
        _logger = RequestLogger(SNAPSHOTS_ROOT)
    return _logger


class RouteRequest(BaseModel):
    customer_id: str
    query: str
    k: int | None = None
    beam: int | None = None
    session_id: str | None = None


class RouteResponse(BaseModel):
    request_id: str
    snapshot_version: str
    tool_id: str | None
    top_k: list[str]
    latency_ms: float


app = FastAPI(title="tool_index.router")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    registry = _get_registry()
    try:
        snap = registry.get(req.customer_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    k = req.k or DEFAULT_K
    beam = req.beam or DEFAULT_BEAM

    embedder = _get_embedder()
    start = time.perf_counter()
    q_emb = embedder.embed(req.query)
    top_k, path = retrieve_with_path(snap.tree, q_emb, k=k, beam=beam)
    latency_ms = (time.perf_counter() - start) * 1000.0

    routed = top_k[0] if top_k else None

    rec = RouteRecord.new(
        customer_id=req.customer_id,
        snapshot_version=snap.version,
        query=req.query,
        routed_tool_id=routed,
        path=path,
        node_scores=[],
        top_k_tool_ids=top_k,
        latency_ms=latency_ms,
        session_id=req.session_id,
    )
    _get_logger().log(rec)

    return RouteResponse(
        request_id=rec.request_id,
        snapshot_version=snap.version,
        tool_id=routed,
        top_k=top_k,
        latency_ms=latency_ms,
    )
