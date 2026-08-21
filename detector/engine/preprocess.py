"""Image decoding and binarisation shared by the proposal and classification stages.

Kept in one module because both stages must agree on the binarisation: if the proposal
stage saw different ink than the classifier does, the crops fed to the model would not
match what the geometry nominated, and the model's training distribution would silently
drift away from what it is asked to score at inference time.
"""

from __future__ import annotations

import cv2
import numpy as np

# Adaptive thresholding parameters. A *local* threshold is mandatory rather than a global
# Otsu cut because sample 3 places checkboxes on blue-shaded table rows and sample 4 lays a
# red watermark across the page: under a global threshold, shaded regions either lose their
# rules entirely or flood to solid ink.
_ADAPTIVE_BLOCK = 31  # neighbourhood size, must be odd
_ADAPTIVE_C = 10  # constant subtracted from the local mean


def decode(data: bytes) -> np.ndarray:
    """Decode raw upload bytes into a BGR image.

    Raises ValueError when the bytes are not a decodable image, which is the boundary
    where a malformed or truncated upload is rejected: everything downstream may then
    assume a valid array. Returns colour (not grayscale) because the caller may need
    chroma to reason about shaded backgrounds.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("payload is not a decodable image")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR (or already-gray) input to a single-channel uint8 image."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Produce a boolean ink mask where True means 'dark stroke'.

    Uses a Gaussian adaptive threshold so that a checkbox printed on a shaded row is
    still separated from its own background. The output is inverted relative to the
    source (ink is True) because every downstream operation counts ink, not paper.
    """
    binv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        _ADAPTIVE_BLOCK,
        _ADAPTIVE_C,
    )
    return binv > 0


def crop_with_context(gray: np.ndarray, x: int, y: int, w: int, h: int,
                      context: float = 1.35, out: int = 40) -> np.ndarray:
    """Cut a padded, size-normalised patch around a proposal for the classifier.

    The patch is deliberately *larger* than the proposed box (``context`` > 1) because the
    decisive evidence for rejecting a false positive is usually just outside the candidate:
    a checkbox sits in whitespace, whereas a same-sized table cell is surrounded by more
    rules and a letter counter is surrounded by the rest of its glyph. Cropping tightly
    would throw that evidence away.

    Edges are replicated rather than zero-padded when the context window runs off the page,
    so that a checkbox at the page margin does not acquire an artificial black border that
    the model would read as a fifth rule. Returns a float32 array in [0, 1] of shape
    (out, out).
    """
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * context
    half = side / 2.0
    x0, y0 = round(cx - half), round(cy - half)
    x1, y1 = round(cx + half), round(cy + half)

    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - gray.shape[1]), max(0, y1 - gray.shape[0])
    if pad_l or pad_t or pad_r or pad_b:
        gray = cv2.copyMakeBorder(gray, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0, x1, y0, y1 = x0 + pad_l, x1 + pad_l, y0 + pad_t, y1 + pad_t

    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        patch = np.full((out, out), 255, np.uint8)
    patch = cv2.resize(patch, (out, out), interpolation=cv2.INTER_AREA)
    return patch.astype(np.float32) / 255.0


# Ring colour for the annotator crop, BGR. Pure red: appraisal scans are black on white, so
# no red can be mistaken for ink, and the prompt can name the colour without ambiguity.
_MARKER_BGR = (0, 0, 255)
_MARKER_PX = 2


def mark_candidate(patch: np.ndarray, context: float) -> np.ndarray:
    """Ring the centred candidate inside a wide judging crop, returning a BGR image.

    A crop cut at `context` times a checkbox's own size shows its neighbours too, and a model
    asked "is this checked?" answers about whichever box carries a mark. Measured on sample_1,
    that was the sole cause of every disagreement between Claude's verdicts and the pixels:
    fourteen candidates with exactly 0.0% interior ink labelled "checked", each one sitting
    directly above or below a marked box. The adjudication prompt already said to ignore
    neighbours and that was not enough -- the model cannot reliably tell which box is "the
    centre one" when boxes tile the region. Drawing the referent removes the inference.

    The ring is placed just OUTSIDE the candidate so it never covers the mark under judgement.
    Accepts either float [0,1] or uint8 input, since callers differ.

    Args:
        patch: square crop, grayscale, centred on the candidate.
        context: the multiple the crop was cut at; the candidate occupies 1/context of a side.

    Returns:
        BGR uint8 image of the same size, with the candidate ringed.
    """
    if patch.dtype != np.uint8:
        patch = (np.clip(patch, 0.0, 1.0) * 255).astype(np.uint8)
    out = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    side = out.shape[0]
    half = max(1, round(side / (2 * max(context, 1.0))))
    c = side // 2
    cv2.rectangle(out, (c - half - _MARKER_PX, c - half - _MARKER_PX),
                  (c + half + _MARKER_PX, c + half + _MARKER_PX), _MARKER_BGR, _MARKER_PX)
    return out
