"""Stage 2: ONNX inference for the checkbox classifier.

Runtime deliberately depends on onnxruntime rather than torch. The training wheel is close
to a gigabyte and pulls a compiler toolchain's worth of transitive dependencies; the serving
image needs neither, and keeping them apart is what allows the detector image to stay small
enough to be worth deploying per-request rather than as a long-lived GPU box.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - the service refuses to start without it
    ort = None

CLASS_NAMES = ("not_a_checkbox", "unchecked", "checked")
IDX_NEGATIVE, IDX_UNCHECKED, IDX_CHECKED = 0, 1, 2

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "checkbox_cls.onnx"

# Crops are scored in slices rather than as one array. A dense page can nominate tens of
# thousands of proposals, and materialising every crop at once is what turns a 40 MB request
# into a 500 MB one; chunking bounds peak memory independently of page size.
_BATCH = 2048


class ModelUnavailableError(RuntimeError):
    """Raised when the ONNX artifact is missing or unreadable.

    Surfaced as a distinct type so the HTTP layer can answer 503 (the service is
    misconfigured and may recover) instead of 500, and so the health endpoint can report
    model readiness separately from process liveness.
    """


class CheckboxClassifier:
    """Scores proposal crops as not_a_checkbox / unchecked / checked."""

    def __init__(self, model_path: str | os.PathLike | None = None, threads: int = 0) -> None:
        """Load the ONNX graph.

        ``threads`` of 0 leaves onnxruntime's own heuristic in place. It is pinned to 1 in
        the container image instead, because the service already parallelises across
        requests and letting each request fan out over every core causes the two levels of
        parallelism to fight for the same CPUs under load.
        """
        if ort is None:  # pragma: no cover
            raise ModelUnavailableError("onnxruntime is not installed")
        path = Path(model_path or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise ModelUnavailableError(
                f"model artifact not found at {path}; run 'python -m training.train' to build it"
            )
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self.model_path = str(path)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, crops: np.ndarray) -> np.ndarray:
        """Return per-crop class probabilities of shape (N, 3).

        ``crops`` is (N, 40, 40) float32 in [0, 1]. An empty input returns an empty (0, 3)
        array rather than raising, because a page with no geometric proposals is a valid
        result (a photograph with no form on it) and not an error condition.
        """
        if crops.size == 0:
            return np.zeros((0, len(CLASS_NAMES)), np.float32)
        flat = crops.astype(np.float32, copy=False)
        batch = flat.reshape(-1, 1, crops.shape[-2], crops.shape[-1])
        out = np.empty((batch.shape[0], len(CLASS_NAMES)), np.float32)
        for start in range(0, batch.shape[0], _BATCH):
            chunk = batch[start:start + _BATCH]
            logits = self._session.run(None, {self._input: chunk})[0]
            out[start:start + chunk.shape[0]] = self._softmax(logits)
        return out
