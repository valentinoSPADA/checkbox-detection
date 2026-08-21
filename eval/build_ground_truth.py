"""Build a ground-truth file by having Claude adjudicate the detector's own candidates.

    python eval/build_ground_truth.py --api http://localhost:8080 --threshold 0.80

The challenge supplies no annotations, and hand-labelling four dense appraisal pages is not a
two-day task. This produces a usable substitute: run the detector at a deliberately permissive
threshold to get a high-recall candidate pool, crop each candidate with surrounding context,
and ask Claude to judge each one as checked / unchecked / not_a_checkbox. Judging a small crop
is exactly the task a compact vision model is good at -- unlike localising a hundred 22 px
boxes on a full page, which it is not.

**The limitation this carries, stated plainly.** Ground truth derived from the detector's own
candidates cannot contain a checkbox the detector never proposed. Recall measured against it
is therefore recall *relative to the Stage 1 proposal pool at this threshold*, not absolute
recall. That number is still worth having -- it isolates how much of what the geometry found
the classifier and policy then keep -- but it flatters the system compared with ground truth
drawn independently, and the writeup must say so rather than quoting the figure bare.

The prompt is read from the Go service's embedded prompt file, so the annotator and the
runtime adjudicator cannot drift apart.
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
import requests

REPO = Path(__file__).resolve().parent.parent
# The candidate marker is shared with the training annotator rather than reimplemented here:
# two copies of "which box am I asking about" would drift, and a drifted marker silently
# relabels data. See engine.preprocess.mark_candidate.
sys.path.insert(0, str(REPO / "detector"))

from engine.preprocess import mark_candidate

PROMPT_PATH = REPO / "backend" / "internal" / "detector" / "vlm" / "prompts" / "adjudicate.txt"

BATCH = 20
CONTEXT = 3.0     # crop this many times the candidate's own size
CROP_PX = 96      # rendered size handed to the model


def crop_around(gray: np.ndarray, bbox: list[int]) -> np.ndarray:
    """Cut a padded, size-normalised crop centred on a candidate, with the candidate ringed.

    The ring is not decoration. Without it, a crop at CONTEXT=3.0 on a dense form shows two or
    three checkboxes and the annotator judges whichever one carries a mark. That produced every
    error in the first ground-truth file: on sample_1, fourteen candidates with exactly 0.0%
    interior ink were labelled "checked", each directly above or below a marked box, and the
    detector was scored wrong for being right. See engine.preprocess.mark_candidate.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(12, int(max(x2 - x1, y2 - y1) * CONTEXT / 2))
    px0, py0 = max(0, half - cx), max(0, half - cy)
    px1 = max(0, cx + half - gray.shape[1])
    py1 = max(0, cy + half - gray.shape[0])
    padded = gray
    if px0 or py0 or px1 or py1:
        padded = cv2.copyMakeBorder(gray, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
        cx, cy = cx + px0, cy + py0
    patch = padded[cy - half:cy + half, cx - half:cx + half]
    if patch.size == 0:
        patch = np.full((CROP_PX, CROP_PX), 255, np.uint8)
    patch = cv2.resize(patch, (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA)
    return mark_candidate(patch, CONTEXT)


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
                                        "enum": ["checked", "unchecked", "not_a_checkbox"]},
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


def adjudicate(client, model: str, prompt: str, patches: list[np.ndarray]) -> dict[int, str]:
    """Send one batch of crops; return {index: verdict}. Unparseable batches yield nothing."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for i, patch in enumerate(patches):
        ok, buf = cv2.imencode(".png", patch)
        if not ok:
            continue
        content.append({"type": "text", "text": f"Region {i}:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(buf.tobytes()).decode("ascii")}})

    message = client.messages.create(
        model=model, max_tokens=4000, tools=[verdict_tool()],
        tool_choice={"type": "tool", "name": "report_verdicts"},
        messages=[{"role": "user", "content": content}])

    for block in message.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "report_verdicts":
            return {v["index"]: v["verdict"] for v in block.input.get("verdicts", [])
                    if isinstance(v.get("index"), int)}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="permissive floor used to build the candidate pool")
    ap.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"))
    ap.add_argument("--samples", type=Path, default=REPO / "samples")
    ap.add_argument("--out", type=Path, default=REPO / "eval" / "ground_truth.json")
    ap.add_argument("--sheet", type=Path, default=REPO / "docs" / "ground_truth_preview.png")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    import anthropic

    client = anthropic.Anthropic()
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    samples_out, audit = [], []

    for image_path in sorted(args.samples.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with image_path.open("rb") as fh:
            resp = requests.post(f"{args.api.rstrip('/')}/detect",
                                 params={"min_confidence": args.threshold, "verbose": "true"},
                                 files={"file": (image_path.name, fh)}, timeout=300)
        resp.raise_for_status()
        boxes = resp.json()["boxes"]
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        patches = [crop_around(gray, b["bbox"]) for b in boxes]

        truth = []
        for start in range(0, len(patches), BATCH):
            chunk = patches[start:start + BATCH]
            try:
                verdicts = adjudicate(client, args.model, prompt, chunk)
            except Exception as exc:  # noqa: BLE001 - one bad batch must not end the run
                print(f"  batch {start}: {exc}", file=sys.stderr)
                continue
            for local_i, verdict in verdicts.items():
                if local_i >= len(chunk):
                    continue
                gi = start + local_i
                audit.append((patches[gi], verdict))
                if verdict in {"checked", "unchecked"}:
                    truth.append({"bbox": boxes[gi]["bbox"],
                                  "is_checked": verdict == "checked"})
        samples_out.append({"image": image_path.name, "boxes": truth})
        print(f"{image_path.name}: {len(boxes)} candidates -> {len(truth)} confirmed checkboxes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "note": ("Adjudicated by Claude from the detector's own candidate pool. Recall measured "
                 "against this file is relative to that pool, not absolute -- see the module "
                 "docstring in eval/build_ground_truth.py."),
        "threshold": args.threshold,
        "model": args.model,
        "samples": samples_out,
    }, indent=1), encoding="utf-8")

    _write_sheet(args.sheet, audit)
    total = sum(len(s["boxes"]) for s in samples_out)
    print(f"\nwrote {total} ground-truth boxes to {args.out}")
    print(f"spot-check the adjudications in {args.sheet}")
    return 0


def _write_sheet(path: Path, audit: list[tuple[np.ndarray, str]]) -> None:
    """Contact sheet so a human can audit what the model decided.

    Model-produced labels are of unknown quality until someone has looked at a sample; a
    systematic labelling error would otherwise surface only as an unexplained metric.
    """
    if not audit:
        return
    short = {"checked": "CHK", "unchecked": "UNC", "not_a_checkbox": "neg"}
    cols, size = 20, 56
    rows = min(18, (len(audit) + cols - 1) // cols)
    sheet = np.full((rows * (size + 13), cols * (size + 3)), 205, np.uint8)
    step = max(1, len(audit) // (rows * cols))  # spread the sample across the whole run
    for k in range(min(len(audit) // step, rows * cols)):
        patch, verdict = audit[k * step]
        r, c = divmod(k, cols)
        y, x = r * (size + 13), c * (size + 3)
        sheet[y:y + size, x:x + size] = cv2.resize(patch, (size, size))
        cv2.putText(sheet, short.get(verdict, "?"), (x, y + size + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


if __name__ == "__main__":
    raise SystemExit(main())
