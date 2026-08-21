"""HTTP surface of the detection sidecar.

This service is internal: it is not exposed publicly by ``docker-compose``, and the Go API
is its only intended client. It therefore speaks a slightly richer contract than the
challenge's public schema -- per-class probabilities, unsuppressed candidates -- so that the
Go policy layer has something to actually decide with. Collapsing that to the public
``{bbox, is_checked}`` shape is the Go service's job.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engine.classifier import CheckboxClassifier, ModelUnavailableError
from engine.pipeline import FLOOR, DetectionPipeline

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("detector")

# Upload ceiling. Enforced here as well as in the Go gateway because the sidecar is
# independently reachable inside the compose network and must not rely on a caller it does
# not control to bound its own memory.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model once at startup rather than per request.

    A missing artifact is logged and left unloaded instead of crashing the process: the
    container then still answers ``/health`` with ``model_loaded: false``, which is a far
    more diagnosable failure than a container that restart-loops with no readable state.
    """
    try:
        clf = CheckboxClassifier(os.getenv("MODEL_PATH") or None,
                                 threads=int(os.getenv("ORT_THREADS", "1")))
        _state["pipeline"] = DetectionPipeline(clf)
        _state["model_path"] = clf.model_path
        log.info("model loaded from %s", clf.model_path)
    except ModelUnavailableError as exc:
        _state["error"] = str(exc)
        log.error("model unavailable: %s", exc)
    yield
    _state.clear()


app = FastAPI(
    title="Checkbox Detection Engine",
    version="1.0.0",
    description="Stage 1 geometric proposals + Stage 2 CNN classification.",
    lifespan=lifespan,
)


class CandidateOut(BaseModel):
    """One scored candidate. Unsuppressed: the caller applies its own NMS and threshold."""

    bbox: list[int] = Field(..., description="[x1, y1, x2, y2] in source pixels")
    is_checked: bool
    confidence: float = Field(..., description="probability of the winning checkbox class")
    p_negative: float
    p_unchecked: float
    p_checked: float


class DetectOut(BaseModel):
    """Response envelope, including counters used for latency and recall diagnostics."""

    candidates: list[CandidateOut]
    width: int
    height: int
    raw_proposals: int
    scored_proposals: int
    elapsed_ms: float


class HealthOut(BaseModel):
    """Liveness plus model readiness, reported separately on purpose."""

    status: str
    model_loaded: bool
    model_path: str | None = None
    detail: str | None = None


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    """GET /health -- liveness and model readiness. No side effects.

    Returns 200 even when the model is absent, with ``model_loaded: false``. Orchestrators
    should gate traffic on that field rather than on the status code, so that a model-less
    container is observable instead of merely unreachable.
    """
    loaded = "pipeline" in _state
    return HealthOut(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_path=_state.get("model_path"),  # type: ignore[arg-type]
        detail=_state.get("error"),  # type: ignore[arg-type]
    )


@app.post("/v1/detect", response_model=DetectOut)
async def detect(
    file: UploadFile = File(..., description="document image (PNG/JPEG/TIFF/BMP)"),
    floor: float = Query(FLOOR, ge=0.0, le=1.0,
                         description="drop candidates below this checkbox probability"),
) -> DetectOut:
    """POST /v1/detect -- run the two-stage pipeline over one uploaded page.

    Side effects: none. No persistence, no outbound calls; the request is pure compute, which
    is what makes the service safe to scale horizontally and to run as a spot/serverless task.

    Failure modes:
      * 413 when the upload exceeds ``MAX_UPLOAD_BYTES``;
      * 400 when the bytes are not a decodable image;
      * 503 when the model artifact never loaded -- retriable, unlike the other two.
    """
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail=_state.get("error", "model not loaded"))

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")

    started = time.perf_counter()
    try:
        result = pipeline.run(data, floor=floor)  # type: ignore[union-attr]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed = (time.perf_counter() - started) * 1000.0

    log.info(
        "detect filename=%s bytes=%d raw=%d scored=%d kept=%d ms=%.1f",
        file.filename, len(data), result.raw_proposals, result.scored_proposals,
        len(result.candidates), elapsed,
    )
    return DetectOut(
        candidates=[CandidateOut(**c.__dict__) for c in result.candidates],
        width=result.width,
        height=result.height,
        raw_proposals=result.raw_proposals,
        scored_proposals=result.scored_proposals,
        elapsed_ms=round(elapsed, 1),
    )


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception) -> JSONResponse:  # pragma: no cover
    """Convert an unexpected error into a 500 without leaking a stack trace to the caller.

    The trace is logged server-side; the client gets a stable shape. Document pipelines are
    fed by untrusted uploads, so an exception message is exactly the wrong thing to echo back.
    """
    log.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
