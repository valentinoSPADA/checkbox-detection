"""Label real Stage 1 proposals with Claude, to close the synthetic-to-real gap.

Why this exists. The classifier trained purely on synthetic crops reached 99.7% on synthetic
validation and then reported thousands of filled checkboxes per real page, because the
generator had never produced the thing real forms are mostly made of: a bordered cell with a
number inside. Synthesis can only contain failure modes someone thought of. The four supplied
images contain the real distribution but carry no labels, and hand-labelling tens of
thousands of crops is not a two-day task.

So Claude labels them. This is weak supervision / model distillation: an expensive general
model annotates data once, offline, and a small specialised model learns from it and then
serves every request for free. The expensive model is not in the request path.

    python -m training.annotate --per-image 400 --out data/annotations.npz

Requires ANTHROPIC_API_KEY. Costs real money, roughly proportional to --per-image, and is
never run by CI or by the service.

Two deliberate details:

* Proposals are sampled *stratified by the current model's confidence*, not uniformly. A
  uniform sample of a page's proposals is almost entirely obvious negatives and teaches the
  model nothing it already knows; the crops worth paying to label are the ones it is unsure
  about. This is ordinary active learning and it is what makes a few hundred labels per page
  worth more than a few thousand random ones.
* The prompt is read from the Go service's embedded prompt file rather than restated here.
  Two copies would drift the moment either is tuned, and a drifted annotation prompt teaches
  the local model a different notion of "checkbox" than the runtime enforces.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.classifier import CheckboxClassifier, ModelUnavailableError
from engine.pipeline import _collapse_duplicates
from engine.preprocess import binarize, crop_with_context, decode, mark_candidate, to_gray
from engine.proposals import propose

REPO = Path(__file__).resolve().parent.parent.parent
# Single source of truth for the labelling instructions; see the module docstring.
PROMPT_PATH = REPO / "backend" / "internal" / "detector" / "vlm" / "prompts" / "adjudicate.txt"

LABELS = {"not_a_checkbox": 0, "unchecked": 1, "checked": 2}

# Regions per model call. Large enough that the per-call overhead is amortised, small enough
# that one malformed response costs a manageable slice of the run and that the image payload
# stays well inside the request limit.
BATCH = 20

# Context multiple for the crop handed to the annotator. Wider than the classifier's own 1.35
# because a human -- or a model -- judging in isolation needs to see the neighbourhood to tell
# a checkbox from a small table cell, and this crop is for judging, not for training input.
ANNOT_CONTEXT = 3.0


# Share of the labelling budget per confidence stratum. The three are sampled SEPARATELY, and
# that separation is the whole point: a single "tails" pool sampled uniformly is swallowed by
# the low tail, which outnumbers the high tail by roughly fifty to one on these pages. The
# first run of this tool spent its budget that way and came back with 97 "checked" crops out
# of 1600 -- not enough positives to teach a three-class model anything about positives.
_BAND_SHARE = 0.45   # 0.2-0.8, where the model is genuinely undecided
_HIGH_SHARE = 0.35   # >= 0.8, confident checkbox: the positive examples
_LOW_SHARE = 0.20    # <= 0.2, confident reject: keeps the model calibrated on easy negatives


# --- Independent pixel evidence, used to police the annotator's verdicts ---------------
#
# The annotator is a model, and a model asked "is this checked?" about a region containing no
# checkbox at all will still answer "unchecked" a good fraction of the time. Two prompt
# revisions cut that from roughly 90% of sampled verdicts to roughly 45%; a third revision was
# not attempted, because the failure is not a wording problem. Instead, every verdict is
# checked against a measurement the pixels make for free.
#
# The measurement is interior brightness: a checkbox is a *container*, so its inside is paper
# whether or not anyone ticked it. Measured on the 117 detections of sample 1 that were
# confirmed genuine by direct ink measurement:
#
#     unchecked boxes   interior brightness 1.000 exactly, all 81
#     checked boxes     0.716 - 0.828, all 36  (an X darkens the middle, it does not fill it)
#
# A proposal inside the black section rail measures near 0.0. The margin is wide enough that
# the bounds below reject nothing real: all 117 survive them.
#
# This is NOT the surrounding-ink filter that was measured and rejected for the detector
# itself (see docs/prototype-log.md). That one judged a candidate by the density of the page
# around it, which is a property of the page. This judges the inside of the candidate against
# what the verdict claims is there, and it never decides anything on its own -- it only
# refuses to let a labelled crop into the training set when the pixels contradict the label.
_VOID_MAX = 0.30        # below this the region contains no paper: not a checkbox, no call made
_UNCHECKED_MIN = 0.85   # a blank interior is blank
_CHECKED_LO, _CHECKED_HI = 0.30, 0.98  # a mark darkens the interior without filling or leaving it


def interior_brightness(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Mean brightness strictly inside a proposal, in [0, 1], with its border ring excluded.

    The ring is excluded because it is the box itself: including it would measure how thickly
    the box is printed rather than what is inside it.
    """
    s = max(w, h)
    m = max(1, int(s * 0.25))
    region = gray[y + m:y + h - m, x + m:x + w - m]
    return float(region.mean()) / 255.0 if region.size else 1.0


def verdict_agrees(label: str, brightness: float) -> bool:
    """Whether the pixels permit the annotator's verdict.

    "not_a_checkbox" is always permitted: it is the safe class, and a false negative costs the
    training set one crop out of thousands. The positive classes are policed, because a wrong
    positive is what actually damages a detector -- it teaches the model that page furniture
    is a form control, which is precisely the failure this whole retraining exists to undo.
    """
    if label == "unchecked":
        return brightness >= _UNCHECKED_MIN
    if label == "checked":
        return _CHECKED_LO <= brightness <= _CHECKED_HI
    return True


def stratified_sample(scores: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick indices across three confidence strata, sampled independently.

    Labelling budget is money, so it is spent where it buys the most: on crops the model is
    unsure about, and on enough confident positives that the retrained model still has a
    notion of what a checkbox looks like. The strata are sampled independently rather than
    pooled because they differ in size by two orders of magnitude -- a page yields thousands
    of confident rejects and perhaps a hundred and fifty confident checkboxes, so any pooled
    draw is effectively a draw from the rejects alone.

    Any stratum that cannot fill its share hands the remainder back, so a page with no
    confident detections still spends its whole budget rather than silently under-labelling.
    """
    if len(scores) <= n:
        return np.arange(len(scores))

    strata = [
        (np.flatnonzero((scores > 0.2) & (scores < 0.8)), _BAND_SHARE),
        (np.flatnonzero(scores >= 0.8), _HIGH_SHARE),
        (np.flatnonzero(scores <= 0.2), _LOW_SHARE),
    ]
    picked: list[np.ndarray] = []
    remaining = n
    # Smallest stratum first: it decides how much it cannot use, and that surplus is then
    # available to the larger ones instead of being lost.
    for pool, share in sorted(strata, key=lambda s: len(s[0])):
        want = min(len(pool), max(0, round(n * share)), remaining)
        if want:
            picked.append(rng.choice(pool, want, replace=False))
            remaining -= want
    if remaining:
        already = np.concatenate(picked) if picked else np.array([], int)
        rest = np.setdiff1d(np.arange(len(scores)), already)
        if len(rest):
            picked.append(rng.choice(rest, min(remaining, len(rest)), replace=False))
    return np.concatenate(picked).astype(int) if picked else np.array([], int)


def encode_png(patch: np.ndarray) -> str:
    """Encode a crop as base64 PNG for the model (grayscale or BGR; both encode fine)."""
    ok, buf = cv2.imencode(".png", patch)
    if not ok:
        raise RuntimeError("failed to encode crop")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def verdict_tool() -> dict:
    """Schema mirroring the Go adapter's report_verdicts tool."""
    return {
        "name": "report_verdicts",
        "description": "Report one verdict per numbered region.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "verdict": {"type": "string",
                                        "enum": list(LABELS.keys())},
                            "confidence": {"type": "number"},
                        },
                        "required": ["index", "verdict", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["verdicts"],
            "additionalProperties": False,
        },
    }


def annotate_batch(client, model: str, prompt: str, patches: list[np.ndarray]) -> dict[int, str]:
    """Send one batch of crops and return {index: verdict}.

    A batch whose response cannot be parsed is dropped rather than retried or guessed at:
    fabricating a label would poison the training set in a way no later metric would reveal,
    and the cost of losing twenty crops out of several thousand is negligible.
    """
    content: list[dict] = [{"type": "text", "text": prompt}]
    for i, patch in enumerate(patches):
        content.append({"type": "text", "text": f"Region {i}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": encode_png(patch)},
        })

    message = client.messages.create(
        model=model,
        max_tokens=4000,
        tools=[verdict_tool()],
        tool_choice={"type": "tool", "name": "report_verdicts"},
        messages=[{"role": "user", "content": content}],
    )
    for block in message.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "report_verdicts":
            out: dict[int, str] = {}
            for v in block.input.get("verdicts", []):
                idx, verdict = v.get("index"), v.get("verdict")
                if isinstance(idx, int) and verdict in LABELS:
                    out[idx] = verdict
            return out
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, default=REPO / "samples")
    ap.add_argument("--per-image", type=int, default=400,
                    help="proposals labelled per sample image")
    ap.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations.npz"))
    ap.add_argument("--sheet", type=Path, default=Path("data/annotations_preview.png"),
                    help="contact sheet for human spot-checking of the labels")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; this tool cannot run without it.", file=sys.stderr)
        return 2
    try:
        import anthropic
    except ImportError:
        print("pip install -r requirements-train.txt", file=sys.stderr)
        return 2

    client = anthropic.Anthropic()
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    rng = np.random.default_rng(args.seed)

    try:
        clf: CheckboxClassifier | None = CheckboxClassifier()
    except ModelUnavailableError:
        # First run, before any model exists: fall back to uniform sampling. The stratified
        # path is an optimisation over a model that already has opinions, not a requirement.
        clf = None
        print("no model yet; sampling proposals uniformly", file=sys.stderr)

    crops: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict] = []

    for image_path in sorted(args.samples.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        gray = to_gray(decode(image_path.read_bytes()))
        props = _collapse_duplicates(propose(binarize(gray)))
        if not props:
            continue

        model_input = np.stack([crop_with_context(gray, p.x, p.y, p.w, p.h) for p in props])
        if clf is not None:
            probs = clf.predict(model_input)
            scores = probs[:, 1:].max(axis=1)
        else:
            scores = rng.random(len(props))
        chosen = stratified_sample(scores, args.per_image, rng)

        # Regions with no paper inside them are settled here, for free and with certainty:
        # nothing that measures this dark can be a box a person ticks. They are the black
        # section rail, and they are both the commonest false positive on these pages and the
        # commonest thing the annotator got wrong -- so labelling them locally buys accuracy
        # AND spends the budget on crops that genuinely need judgement.
        bright = np.array([interior_brightness(gray, props[i].x, props[i].y,
                                               props[i].w, props[i].h) for i in chosen])
        void = chosen[bright < _VOID_MAX]
        chosen = chosen[bright >= _VOID_MAX]
        for i in void:
            crops.append(model_input[int(i)])
            labels.append(LABELS["not_a_checkbox"])
            records.append({"image": image_path.name, "bbox": props[int(i)].as_bbox(),
                            "label": "not_a_checkbox", "source": "pixels"})
        free = len(void)

        # Wider crops for judging; the narrow ones above are what the classifier will train on.
        judge_input = [crop_with_context(gray, props[i].x, props[i].y, props[i].w, props[i].h,
                                         context=ANNOT_CONTEXT, out=96) for i in chosen]

        labelled = 0
        contradicted = 0
        for start in range(0, len(chosen), BATCH):
            idx_slice = chosen[start:start + BATCH]
            # Ringed, not bare: see mark_candidate. Labelling a wide crop without marking
            # which box is the subject teaches the model that a blank box beside a marked one
            # is itself marked -- the exact defect this pipeline exists to remove.
            patches = [mark_candidate(judge_input[k], ANNOT_CONTEXT)
                       for k in range(start, min(start + BATCH, len(chosen)))]
            try:
                verdicts = annotate_batch(client, args.model, prompt, patches)
            except Exception as exc:
                print(f"  batch at {start} failed: {exc}", file=sys.stderr)
                continue
            for local_i, verdict in verdicts.items():
                if local_i >= len(idx_slice):
                    continue
                global_i = int(idx_slice[local_i])
                p = props[global_i]
                # Two independent sources must agree before a crop enters the training set.
                # Where they disagree the crop is DISCARDED rather than resolved in favour of
                # either -- guessing which one is right is how label noise gets laundered into
                # something that looks like data.
                if not verdict_agrees(verdict, interior_brightness(gray, p.x, p.y, p.w, p.h)):
                    contradicted += 1
                    continue
                crops.append(model_input[global_i])
                labels.append(LABELS[verdict])
                records.append({"image": image_path.name, "bbox": p.as_bbox(),
                                "label": verdict, "source": "model"})
                labelled += 1
        print(f"{image_path.name}: {len(props)} proposals, {labelled} labelled, "
              f"{free} settled by pixels, {contradicted} discarded as contradicted")

    if not crops:
        print("nothing was labelled", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, crops=np.stack(crops).astype(np.float32),
                        labels=np.array(labels, np.int64))
    args.out.with_suffix(".json").write_text(json.dumps(records, indent=1), encoding="utf-8")

    _write_sheet(args.sheet, crops, labels)
    counts = {name: labels.count(v) for name, v in LABELS.items()}
    print(f"\nwrote {len(crops)} labelled crops to {args.out}")
    print(f"class balance: {counts}")
    print(f"spot-check the labels in {args.sheet} before training on them")
    return 0


def _write_sheet(path: Path, crops: list[np.ndarray], labels: list[int]) -> None:
    """Render a contact sheet so a human can audit what the annotator decided.

    Not optional in spirit: labels produced by a model are training data of unknown quality
    until someone has actually looked at a sample of them, and a systematic labelling error
    would otherwise be discovered only as an unexplained accuracy ceiling.
    """
    names = {v: k[:5] for k, v in LABELS.items()}
    cols, size = 24, 40
    rows = min(20, (len(crops) + cols - 1) // cols)
    sheet = np.full((rows * (size + 12), cols * (size + 3)), 210, np.uint8)
    for k in range(min(len(crops), rows * cols)):
        r, c = divmod(k, cols)
        y, x = r * (size + 12), c * (size + 3)
        sheet[y:y + size, x:x + size] = (crops[k] * 255).astype(np.uint8)
        cv2.putText(sheet, names[labels[k]], (x, y + size + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


if __name__ == "__main__":
    raise SystemExit(main())
