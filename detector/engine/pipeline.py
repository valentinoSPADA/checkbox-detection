"""Wires Stage 1 (geometric proposals) to Stage 2 (learned classification).

The pipeline returns *scored candidates without suppression*. Deduplication for final
output and the confidence cut are policy decisions and live in the Go service
(``backend/internal/domain``); what happens here is only the compute-bound work that needs
the pixels. The one exception is the coarse collapse in :func:`_collapse_duplicates`, which
is a memory optimisation rather than a policy: proposals at adjacent sizes and one-pixel
offsets describe the same box, and scoring all of them would multiply inference cost by an
order of magnitude for identical results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine import preprocess
from engine.classifier import IDX_CHECKED, IDX_NEGATIVE, IDX_UNCHECKED, CheckboxClassifier
from engine.proposals import Proposal, propose

# Quantisation used when collapsing near-identical proposals. Centres within 4 px and sides
# within 6 px of each other are treated as the same nomination.
# Proposals cropped and scored per slice. Matched to the classifier's own inference batch so
# the two do not chunk at different granularities and interleave badly; see the sweep behind
# that number in engine/classifier.py.
CROP_BATCH = 256

# Classes the model emits: not_a_checkbox / unchecked / checked.
NUM_CLASSES = 3

_CENTRE_BUCKET = 4
_SIZE_BUCKET = 6

# Candidates below this floor are dropped before leaving the sidecar. This is a transport
# concession, not the detection threshold: a dense page yields thousands of candidates and
# nearly all are confidently negative, so shipping them to the Go service would waste far
# more bandwidth than it preserves optionality. The real threshold is applied in Go and is
# well above this floor.
FLOOR = 0.30


@dataclass(frozen=True)
class Candidate:
    """A classified proposal, in source-image pixel coordinates."""

    bbox: list[int]           # [x1, y1, x2, y2]
    is_checked: bool
    confidence: float         # probability of the winning checkbox class
    p_negative: float
    p_unchecked: float
    p_checked: float


def _collapse_duplicates(proposals: list[Proposal]) -> list[Proposal]:
    """Keep one proposal per (centre bucket, size bucket).

    Hash-based rather than IoU-based because this runs on up to ~80k proposals per page and
    must stay linear; genuine overlap resolution happens later, on the few hundred survivors,
    where an O(n^2) pass is affordable and can be ranked by model confidence.
    """
    seen: dict[tuple[int, int, int], Proposal] = {}
    for p in proposals:
        key = (
            (p.x + p.w // 2) // _CENTRE_BUCKET,
            (p.y + p.h // 2) // _CENTRE_BUCKET,
            max(p.w, p.h) // _SIZE_BUCKET,
        )
        if key not in seen:
            seen[key] = p
    return list(seen.values())


@dataclass
class PipelineResult:
    """Candidates plus the counters the API surfaces for observability."""

    candidates: list[Candidate]
    width: int
    height: int
    raw_proposals: int
    scored_proposals: int


class DetectionPipeline:
    """Stateless per-request pipeline; the loaded model is the only shared resource."""

    def __init__(self, classifier: CheckboxClassifier) -> None:
        self._clf = classifier

    def run(self, image_bytes: bytes, floor: float = FLOOR) -> PipelineResult:
        """Detect and classify checkboxes in an encoded image.

        Raises ValueError if the bytes are not a decodable image. Returns every candidate
        whose best checkbox class exceeds ``floor``, unsuppressed and unordered; callers are
        expected to apply their own suppression and threshold.
        """
        img = preprocess.decode(image_bytes)
        gray = preprocess.to_gray(img)
        ink = preprocess.binarize(gray)

        raw = propose(ink)
        deduped = _collapse_duplicates(raw)

        if not deduped:
            return PipelineResult([], gray.shape[1], gray.shape[0], len(raw), 0)

        # Cropped and scored in slices, never all at once.
        #
        # The classifier already chunks its own inference, and that was defeated here: a
        # dense page nominates ~31k proposals, and one np.stack over all of them is a 200 MB
        # array built from a 200 MB list of pieces. Measured, the request went from a 140 MB
        # process to a 640 MB one and the allocator did not hand it back. Slicing bounds the
        # crop memory at the batch size -- ~13 MB -- regardless of how dense the page is.
        probs = np.empty((len(deduped), NUM_CLASSES), np.float32)
        for start in range(0, len(deduped), CROP_BATCH):
            part = deduped[start:start + CROP_BATCH]
            crops = np.stack([
                preprocess.crop_with_context(gray, p.x, p.y, p.w, p.h) for p in part
            ])
            probs[start:start + len(part)] = self._clf.predict(crops)

        out: list[Candidate] = []
        for p, pr in zip(deduped, probs, strict=False):
            best_checkbox = float(max(pr[IDX_UNCHECKED], pr[IDX_CHECKED]))
            if best_checkbox < floor:
                continue
            out.append(Candidate(
                bbox=p.as_bbox(),
                is_checked=bool(pr[IDX_CHECKED] >= pr[IDX_UNCHECKED]),
                confidence=best_checkbox,
                p_negative=float(pr[IDX_NEGATIVE]),
                p_unchecked=float(pr[IDX_UNCHECKED]),
                p_checked=float(pr[IDX_CHECKED]),
            ))
        return PipelineResult(out, gray.shape[1], gray.shape[0], len(raw), len(deduped))
