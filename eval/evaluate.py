"""Measure detection quality against a ground-truth file.

Run against a live API so that what is measured is the system a reviewer would actually
call -- policy, suppression, thresholds and all -- rather than an internal function that
happens to be convenient to import:

    python eval/evaluate.py --api http://localhost:8080 --engine local
    python eval/evaluate.py --api http://localhost:8080 --engine assisted --iou 0.5

Ground truth lives in eval/ground_truth.json:

    {"samples": [{"image": "sample_1_urar_1004.png",
                  "boxes": [{"bbox": [x1,y1,x2,y2], "is_checked": true}, ...]}]}

Matching is greedy by descending IoU, one detection to at most one ground-truth box, which is
the standard object-detection convention: without the one-to-one constraint a detector could
inflate recall by emitting many overlapping boxes on the same checkbox.

Three numbers are reported, and the split matters. Localisation precision/recall says whether
the box was *found*; classification accuracy, computed only over matched pairs, says whether
filled/unfilled was called correctly. A single blended score would let a detector that finds
everything and mislabels half of it look identical to one that finds half and labels it
perfectly, and those are very different systems to operate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    is_checked: bool

    @staticmethod
    def from_json(d: dict) -> Box:
        x1, y1, x2, y2 = d["bbox"]
        return Box(int(x1), int(y1), int(x2), int(y2), bool(d.get("is_checked", False)))

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: Box) -> float:
        iw = min(self.x2, other.x2) - max(self.x1, other.x1)
        ih = min(self.y2, other.y2) - max(self.y1, other.y1)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    class_correct: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def class_accuracy(self) -> float:
        """Filled/unfilled accuracy over matched pairs only.

        Undefined when nothing matched; reported as 0.0 rather than raising, so a totally
        failed run still produces a row in the table instead of aborting the whole sweep.
        """
        return self.class_correct / self.tp if self.tp else 0.0


def match(truth: list[Box], pred: list[Box], iou_threshold: float) -> Score:
    """Greedily pair predictions to ground truth by descending IoU."""
    pairs = sorted(
        (
            (t.iou(p), ti, pi)
            for ti, t in enumerate(truth)
            for pi, p in enumerate(pred)
            if t.iou(p) >= iou_threshold
        ),
        key=lambda x: -x[0],
    )
    used_t: set[int] = set()
    used_p: set[int] = set()
    score = Score()
    for _, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        score.tp += 1
        if truth[ti].is_checked == pred[pi].is_checked:
            score.class_correct += 1
    score.fp = len(pred) - len(used_p)
    score.fn = len(truth) - len(used_t)
    return score


def detect(api: str, image: Path, engine: str, min_confidence: float | None,
           timeout: float) -> list[Box]:
    """Call the live API for one image."""
    params = {"engine": engine, "verbose": "true"}
    if min_confidence is not None:
        params["min_confidence"] = str(min_confidence)
    with image.open("rb") as fh:
        response = requests.post(
            f"{api.rstrip('/')}/detect", params=params,
            files={"file": (image.name, fh, "application/octet-stream")}, timeout=timeout)
    response.raise_for_status()
    return [Box.from_json(b) for b in response.json()["boxes"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--engine", default="local", choices=["local", "vlm", "assisted"])
    ap.add_argument("--iou", type=float, default=0.4,
                    help="IoU at which a detection counts as a match (default 0.4; box edges "
                         "are only a couple of pixels thick, so a stricter threshold measures "
                         "border alignment more than detection)")
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--truth", type=Path, default=REPO / "eval" / "ground_truth.json")
    ap.add_argument("--samples", type=Path, default=REPO / "samples")
    args = ap.parse_args()

    if not args.truth.exists():
        print(f"no ground truth at {args.truth}\n"
              f"build one with: python eval/annotate.py --help", file=sys.stderr)
        return 2

    truth_doc = json.loads(args.truth.read_text(encoding="utf-8"))
    total = Score()
    rows: list[tuple[str, Score, int]] = []

    for entry in truth_doc["samples"]:
        image = args.samples / entry["image"]
        if not image.exists():
            print(f"skipping missing sample {image}", file=sys.stderr)
            continue
        truth = [Box.from_json(b) for b in entry["boxes"]]
        pred = detect(args.api, image, args.engine, args.min_confidence, args.timeout)
        score = match(truth, pred, args.iou)
        rows.append((entry["image"], score, len(pred)))
        total.tp += score.tp
        total.fp += score.fp
        total.fn += score.fn
        total.class_correct += score.class_correct

    header = f"{'sample':34s}{'GT':>5s}{'pred':>6s}{'TP':>5s}{'FP':>5s}{'FN':>5s}" \
             f"{'prec':>8s}{'rec':>8s}{'F1':>8s}{'class':>8s}"
    print(f"\nengine={args.engine}  IoU>={args.iou}\n")
    print(header)
    print("-" * len(header))
    for name, s, npred in rows:
        print(f"{name:34s}{s.tp + s.fn:>5d}{npred:>6d}{s.tp:>5d}{s.fp:>5d}{s.fn:>5d}"
              f"{s.precision:>8.3f}{s.recall:>8.3f}{s.f1:>8.3f}{s.class_accuracy:>8.3f}")
    print("-" * len(header))
    print(f"{'TOTAL':34s}{total.tp + total.fn:>5d}{'':>6s}{total.tp:>5d}{total.fp:>5d}"
          f"{total.fn:>5d}{total.precision:>8.3f}{total.recall:>8.3f}{total.f1:>8.3f}"
          f"{total.class_accuracy:>8.3f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
