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
from engine.preprocess import binarize, crop_with_context, decode, to_gray
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


def stratified_sample(scores: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick indices spread across the confidence range, weighted toward the uncertain middle.

    Half the budget goes to the 0.2-0.8 band where the model is actually undecided, and the
    remainder is spread over the confident tails so the labelled set still contains examples
    of what confident-correct looks like. Sampling only the middle would produce a training
    set with no easy examples at all and a model that loses its calibration on them.
    """
    if len(scores) <= n:
        return np.arange(len(scores))
    middle = np.flatnonzero((scores > 0.2) & (scores < 0.8))
    tails = np.flatnonzero((scores <= 0.2) | (scores >= 0.8))
    want_mid = min(len(middle), n // 2)
    want_tail = min(len(tails), n - want_mid)
    picked = np.concatenate([
        rng.choice(middle, want_mid, replace=False) if want_mid else np.array([], int),
        rng.choice(tails, want_tail, replace=False) if want_tail else np.array([], int),
    ])
    return picked.astype(int)


def encode_png(patch: np.ndarray) -> str:
    """Encode a grayscale crop as base64 PNG for the model."""
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

        # Wider crops for judging; the narrow ones above are what the classifier will train on.
        judge_input = [crop_with_context(gray, props[i].x, props[i].y, props[i].w, props[i].h,
                                         context=ANNOT_CONTEXT, out=96) for i in chosen]

        labelled = 0
        for start in range(0, len(chosen), BATCH):
            idx_slice = chosen[start:start + BATCH]
            patches = [(judge_input[k] * 255).astype(np.uint8)
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
                crops.append(model_input[global_i])
                labels.append(LABELS[verdict])
                p = props[global_i]
                records.append({"image": image_path.name, "bbox": p.as_bbox(), "label": verdict})
                labelled += 1
        print(f"{image_path.name}: {len(props)} proposals, {labelled} labelled")

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
