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

# Size bounds, expressed as a FRACTION OF PAGE WIDTH rather than in pixels.
#
# A checkbox is sized relative to the page it is printed on, not in absolute pixels: it is
# drawn to be legible next to body text, so a scan at twice the resolution has boxes twice as
# wide. Measured over 284 hand-confirmed checkboxes across the four sample pages -- which span
# 1586 to 2550 px wide and 20 to 56 px of box -- the ratio holds inside a single narrow band:
#
#     side / page width:  min 0.0094   median 0.0196   max 0.0220
#
# while the 343 candidates a person rejected sit at a median of 0.0047, mostly letter counters.
# The two populations barely overlap, which an absolute pixel range cannot express: a 10 px
# floor is 0.0039 of sample 1 and 0.0063 of sample 2, so one number means different things on
# different pages, and it has to be loose enough for the smallest of them.
#
# The bounds below are set with the margin stated, not fitted to the data:
#
#     lower 0.0065  ->  31% below the smallest confirmed checkbox; drops 246/343 rejects
#     upper 0.0300  ->  37% above the largest;                     drops 0 real
#
# This is a recall-critical stage, so the asymmetry is deliberate -- both bounds are set well
# outside the observed range, and the classifier still decides everything inside it.
MIN_SIDE_FRAC = 0.0065
MAX_SIDE_FRAC = 0.0300

# Absolute bounds. These are the sweep for any input the fractions cannot sensibly describe --
# a small crop, a thumbnail -- and the clamps on one that would otherwise sweep hundreds of
# sizes. MIN_SIDE and MAX_SIDE_FLOOR together are exactly the range this stage used before the
# fractions existed, which is the point: the relative rule may only ever NARROW the sweep, on a
# page large enough for a fraction of its width to mean something. A 200 px crop holding one
# 30 px checkbox is a legitimate input, and 0.0065 of 200 px is one pixel -- a rule derived
# from full pages must not quietly declare such an image to contain nothing.
MIN_SIDE = 10
MAX_SIDE_FLOOR = 70
MAX_SIDE = 160
SIZE_STEP = 2


def size_range(width: int) -> tuple[int, int]:
    """Pixel size bounds for an image of the given width.

    Returns the sweep the proposal stage should run, in pixels. A function rather than two
    constants because the answer depends on the image, and every caller -- the pipeline, the
    annotator, the tests -- must derive it identically or they will disagree about which
    candidates exist at all.

    On the four sample pages this yields 17-77 for the 2550 px scans and 10-70 for the 1586 px
    crop, cutting raw proposals by 71-78% on the former while losing none of the 284
    hand-confirmed checkboxes.
    """
    lo = max(MIN_SIDE, round(width * MIN_SIDE_FRAC))
    hi = min(MAX_SIDE, max(MAX_SIDE_FLOOR, round(width * MAX_SIDE_FRAC)))
    return lo, max(hi, lo + SIZE_STEP)

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

# Rows per block when measuring run lengths. Large enough that the per-block Python overhead
# is irrelevant against the vectorised work, small enough that the four int32 temporaries the
# calculation needs stay in the low megabytes instead of the low hundreds.
_RUN_BLOCK = 512

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


def _run_lengths(ink: np.ndarray, axis: int, cap: int | None = None) -> np.ndarray:
    """Length of the contiguous ink run starting at each pixel, along ``axis``.

    ``axis=1`` measures rightwards, ``axis=0`` downwards. Implemented with a reversed
    running minimum over the index of the next non-ink pixel, which is fully vectorised;
    the naive per-column Python loop this replaces cost ~19 s on a 2550x4200 page and
    dominated total request latency.

    ``cap`` clamps every length to at most that value and, when it fits in a byte, returns
    uint8 instead of int32. This is not an approximation: the only use of these lengths is
    ``runs >= need`` with ``need`` bounded by ``span * max_side``, so any run longer than the
    largest side under consideration is already indistinguishable from one exactly that long.
    On a 2550x4200 page it takes the two retained arrays from 86 MB to 21 MB, which is the
    difference between fitting a small container and not.
    """
    if axis == 0:
        return _run_lengths(ink.T, 1, cap).T

    h, w = ink.shape
    dtype = np.uint8 if (cap is not None and cap < 256) else np.int32
    out = np.empty((h, w), dtype)
    idx = np.arange(w, dtype=np.int32)

    # Computed in row blocks rather than over the whole page at once.
    #
    # Rows are independent for a rightward run, so any horizontal partition is exact: this
    # changes the memory profile and nothing else, and the proposal count is identical either
    # way. It matters because the vectorised form needs four int32 temporaries the size of its
    # input, and this function is the single largest allocation the service makes.
    #
    # Swept on sample 1 (2550x4200), peak traced allocation for the whole propose() call:
    #
    #     block   128    512   1024   2048   whole page
    #     peak     73     99    133    168      193 MB
    #     secs   4.86   4.23   4.41   4.60     4.44
    #
    # 512 is the knee. Note the timings: blocking is not a speed/memory trade here, it is
    # free -- the working set stays in cache, which pays back the per-block overhead. 128 is
    # the only setting that costs anything, and it buys 26 MB for 15% of the runtime.
    for y0 in range(0, h, _RUN_BLOCK):
        block = ink[y0:y0 + _RUN_BLOCK]
        # For every pixel, the index of the nearest non-ink pixel at or to its right, or w.
        nxt = np.where(~block, idx[None, :], np.int32(w))
        nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
        runs = np.where(block, nxt - idx[None, :], 0)
        if cap is not None:
            np.minimum(runs, cap, out=runs)
        out[y0:y0 + _RUN_BLOCK] = runs
    return out


def propose(ink: np.ndarray,
            span: float = DEFAULT_SPAN,
            tol: int = DEFAULT_TOL,
            min_side: int | None = None,
            max_side: int | None = None) -> list[Proposal]:
    """Nominate every square region whose four sides each carry a long straight ink run.

    ``ink`` is the boolean mask from :func:`preprocess.binarize`. The sweep is over square
    sides only; slightly non-square checkboxes are still caught because ``span`` < 1 and the
    edge match carries ``tol`` pixels of slack, and the classifier is trained on crops with
    the same aspect jitter.

    ``min_side`` and ``max_side`` default to :func:`size_range` of the image width, which is
    what makes the stage scale-adaptive. Passing them explicitly is for tests and experiments
    only; production must not, or a page at an unexpected DPI silently gets the wrong sweep.

    Returns proposals in raster order, unfiltered and overlapping — deduplication is
    deliberately deferred until confidences exist, so that suppression can keep the
    best-scoring box rather than an arbitrary geometric pick.
    """
    h, w = ink.shape
    auto_lo, auto_hi = size_range(w)
    min_side = auto_lo if min_side is None else min_side
    max_side = auto_hi if max_side is None else max_side
    ink_u8 = ink.astype(np.uint8)
    bridged_h = cv2.morphologyEx(
        ink_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (BRIDGE, 1)))
    bridged_v = cv2.morphologyEx(
        ink_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, BRIDGE)))
    # Capped at max_side: see _run_lengths. Nothing in the sweep can ask about a longer run.
    runs_r = _run_lengths(bridged_h.astype(bool), axis=1, cap=max_side)
    runs_d = _run_lengths(bridged_v.astype(bool), axis=0, cap=max_side)
    del bridged_h, bridged_v, ink_u8

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * tol + 1))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tol + 1, 1))

    # Scratch buffers, allocated once and rewritten on every size.
    #
    # The sweep runs ~30 times on a full page, and the naive form allocates seven
    # page-sized temporaries per pass -- a threshold mask, its uint8 cast, the dilation, its
    # bool cast, twice over, plus the chain of ANDs. That is ~115 MB of churn per iteration,
    # which the allocator does not hand back between requests: RSS climbed to 1.2 GB over
    # four pages. Reusing buffers makes the loop allocate nothing but its results.
    mask = np.empty((h, w), np.uint8)
    dil_h = np.empty((h, w), np.uint8)
    dil_v = np.empty((h, w), np.uint8)
    acc = np.empty((h, w), bool)

    out: list[Proposal] = []
    for side in range(min_side, max_side + 1, SIZE_STEP):
        if side > min(h, w):
            break
        need = round(span * side)
        hh, ww = h - side + 1, w - side + 1

        # Dilating along the perpendicular axis is what grants the +/- tol slack: a top rule
        # two pixels thick then satisfies the test at any of its rows.
        np.greater_equal(runs_r, need, out=mask, casting="unsafe")
        cv2.dilate(mask, v_kernel, dst=dil_h)
        np.greater_equal(runs_d, need, out=mask, casting="unsafe")
        cv2.dilate(mask, h_kernel, dst=dil_v)

        # Folded in place: each AND writes over the running result rather than producing a
        # new array, so a four-way test costs one buffer instead of three.
        window = acc[:hh, :ww]
        np.logical_and(dil_h[:hh, :ww], dil_h[side - 1:, :ww], out=window)
        np.logical_and(window, dil_v[:hh, :ww], out=window)
        np.logical_and(window, dil_v[:hh, side - 1:], out=window)

        ys, xs = np.nonzero(window)
        out.extend(Proposal(int(x), int(y), side, side) for y, x in zip(ys, xs, strict=False))
    return out
