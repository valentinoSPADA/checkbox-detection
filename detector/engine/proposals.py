"""Stage 1: geometric region proposals.

This stage answers "where *could* a checkbox be?" and is deliberately tuned for recall,
not precision. Anything it fails to nominate is unrecoverable downstream, whereas the
false positives it emits are exactly what the learned Stage 2 classifier exists to remove.

Why rectangle assembly from straight runs, rather than the two obvious alternatives:

* Morphological line masks cannot work here. The structuring element would have to be
  long enough to exclude a bold-text stroke (~12 px at 300 dpi) yet short enough to keep a
  checkbox rule (~22 px). That window is too narrow, and in practice the mask fills with
  letter counters.
* Connected components cannot work here either. On dense appraisal forms a checkbox very
  often *touches* the table rule beside it, so its component becomes the whole table grid
  and the box disappears. Measured on sample 3, that approach found 5 boxes on a page
  holding roughly ninety.

Assembling rectangles from straight ink runs is immune to both failures: it only ever asks
"does a straight run of at least ``span * side`` start here", which a shared rule satisfies
just as well as a private one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Absolute pixel bounds on the side of a candidate. The range is wide on purpose: page DPI
# is unknown and sample 2 is a zoomed crop whose boxes are several times the size of
# sample 1's, so no single expected size exists. Roughly 100-800 dpi is covered.
MIN_SIDE = 10
MAX_SIDE = 70
SIZE_STEP = 2

# Fraction of a side that must be covered by one straight ink run for that side to count as
# present. Below 1.0 so that a broken scan rule, a rounded corner, or a mark overlapping the
# border does not veto an otherwise valid box.
DEFAULT_SPAN = 0.80

# Rows/columns of slack when matching an edge, absorbing border thickness and slight skew.
DEFAULT_TOL = 2

# Width of the morphological closing applied along each axis before runs are measured.
# Scanned and faxed rules drop pixels, and a run test is unforgiving of that: a four-pixel
# dropout in a thirty-pixel border halves the longest run and vetoes the whole box. Closing
# along each axis separately bridges such gaps without joining strokes that are genuinely
# apart, and crucially without bridging in the perpendicular direction, which is what would
# start fusing neighbouring glyphs into false rules.
# Measured on sample 1: a bridge of 5 produces 320k raw proposals against 78k with no
# bridging, because it fuses the glyphs of a text line into one long run and then treats that
# run as a rule. A bridge of 3 costs 108k and still closes the one- and two-pixel dropouts
# that scanned rules actually have. The wider setting bought tolerance nobody needed at a
# price the classifier had to pay on every page.
BRIDGE = 3

@dataclass(frozen=True)
class Proposal:
    """A candidate checkbox location in source-image pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    def as_bbox(self) -> list[int]:
        """Return [x1, y1, x2, y2] as the challenge's response schema expects."""
        return [self.x, self.y, self.x + self.w, self.y + self.h]


def _run_lengths(ink: np.ndarray, axis: int) -> np.ndarray:
    """Length of the contiguous ink run starting at each pixel, along ``axis``.

    ``axis=1`` measures rightwards, ``axis=0`` downwards. Implemented with a reversed
    running minimum over the index of the next non-ink pixel, which is fully vectorised;
    the naive per-column Python loop this replaces cost ~19 s on a 2550x4200 page and
    dominated total request latency.
    """
    if axis == 0:
        return _run_lengths(ink.T, 1).T

    _h, w = ink.shape
    idx = np.arange(w, dtype=np.int32)
    # For every pixel, the index of the nearest non-ink pixel at or to its right (w if none).
    nxt = np.where(~ink, idx[None, :], np.int32(w))
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    return np.where(ink, nxt - idx[None, :], 0).astype(np.int32)


def propose(ink: np.ndarray,
            span: float = DEFAULT_SPAN,
            tol: int = DEFAULT_TOL,
            min_side: int = MIN_SIDE,
            max_side: int = MAX_SIDE) -> list[Proposal]:
    """Nominate every square region whose four sides each carry a long straight ink run.

    ``ink`` is the boolean mask from :func:`preprocess.binarize`. The sweep is over square
    sides only; slightly non-square checkboxes are still caught because ``span`` < 1 and the
    edge match carries ``tol`` pixels of slack, and the classifier is trained on crops with
    the same aspect jitter.

    Returns proposals in raster order, unfiltered and overlapping — deduplication is
    deliberately deferred until confidences exist, so that suppression can keep the
    best-scoring box rather than an arbitrary geometric pick.
    """
    h, w = ink.shape
    ink_u8 = ink.astype(np.uint8)
    bridged_h = cv2.morphologyEx(
        ink_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (BRIDGE, 1)))
    bridged_v = cv2.morphologyEx(
        ink_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, BRIDGE)))
    runs_r = _run_lengths(bridged_h.astype(bool), axis=1)
    runs_d = _run_lengths(bridged_v.astype(bool), axis=0)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * tol + 1))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tol + 1, 1))

    out: list[Proposal] = []
    for side in range(min_side, max_side + 1, SIZE_STEP):
        if side > min(h, w):
            break
        need = round(span * side)
        # Dilating along the perpendicular axis is what grants the +/- tol slack: a top rule
        # two pixels thick then satisfies the test at any of its rows.
        horiz = cv2.dilate((runs_r >= need).astype(np.uint8), v_kernel).astype(bool)
        vert = cv2.dilate((runs_d >= need).astype(np.uint8), h_kernel).astype(bool)

        top = horiz[: h - side + 1, : w - side + 1]
        bottom = horiz[side - 1:, : w - side + 1]
        left = vert[: h - side + 1, : w - side + 1]
        right = vert[: h - side + 1, side - 1:]

        ys, xs = np.nonzero(top & bottom & left & right)
        out.extend(Proposal(int(x), int(y), side, side) for y, x in zip(ys, xs, strict=False))
    return out
