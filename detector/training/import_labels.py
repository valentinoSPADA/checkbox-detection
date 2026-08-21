"""Turn hand-made labels into a training set, overriding the model-made ones.

    python -m training.import_labels --labels ~/Downloads/labels.json \
        --merge data/annotations.npz --out data/merged.npz

Why a merge rather than a replacement. The Claude-labelled set is ~1832 crops and is mostly
right; its value is volume, especially on `not_a_checkbox`, which is the class a page has
thousands of and a person has no reason to enumerate by hand. The hand-labelled set is a few
hundred crops and is *authoritative*: it is the only source in this repository that defines
what a checkbox is without a model in the loop. So both are kept, and where they describe the
same box the human wins.

Three details that decide whether this helps or quietly hurts:

* **Precedence is by overlap, not by exact coordinates.** The two sets come from the same
  proposal pool, but a box can shift a pixel between runs, and an exact-match rule would
  silently leave a contradicting model label in the set right next to the human correction --
  the model's mistake would survive precisely where it was caught.
* **Human labels are weighted, not merely included.** A few hundred crops among tens of
  thousands of synthetic ones contribute almost nothing to the gradient at equal weight. They
  are repeated, and by more than the Claude labels are, because they are more trustworthy.
* **Nothing is inferred for what was skipped.** A crop the labeller passed on is a crop nobody
  was sure about, and the honest handling of "I don't know" is absence, not a guess.

Crops are re-cut from the sample images rather than carried in the labels file, so a labels
file is a few hundred KB of decisions that stays readable and diffable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from engine.preprocess import crop_with_context, decode, to_gray

REPO = Path(__file__).resolve().parent.parent.parent

LABELS = {"not_a_checkbox": 0, "unchecked": 1, "checked": 2}
NAMES = {v: k for k, v in LABELS.items()}

# Overlap above which two entries are taken to describe the same checkbox. Deliberately
# generous: the cost of treating two boxes as one is losing a redundant crop, while the cost of
# treating one box as two is keeping a model label that a person already overruled.
SAME_BOX_IOU = 0.35


def iou(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def cut(samples: Path, records: list[dict]) -> np.ndarray:
    """Re-cut the classifier's own input crop for each record.

    Uses `crop_with_context` at its defaults, which is the same call the pipeline makes at
    inference. Cutting them any other way here would train the model on crops it never sees.
    """
    cache: dict[str, np.ndarray] = {}
    out = []
    for r in records:
        name = r["image"]
        if name not in cache:
            cache[name] = to_gray(decode((samples / name).read_bytes()))
        x1, y1, x2, y2 = r["bbox"]
        out.append(crop_with_context(cache[name], x1, y1, x2 - x1, y2 - y1))
    return np.stack(out).astype(np.float32) if out else np.zeros((0, 40, 40), np.float32)


def report_disagreements(records: list[dict]) -> None:
    """Print where the human overruled the model, because that is the point of the exercise.

    A hand-labelling pass that agrees with the model everywhere taught the model nothing and
    should be visible as such, rather than being reported as several hundred new labels.
    """
    rows = [r for r in records if r.get("model_says") in {"checked", "unchecked", "rejected"}]
    if not rows:
        return
    agree = sum(1 for r in rows if r["model_says"] == r["label"])
    print(f"\nagreement with the current model: {agree}/{len(rows)} ({agree / len(rows):.1%})")

    pairs = Counter((r["model_says"], r["label"]) for r in rows if r["model_says"] != r["label"])
    if not pairs:
        print("  no disagreements -- this pass confirms the model rather than correcting it")
        return
    print("  corrections, most common first:")
    for (said, truth), n in pairs.most_common(10):
        print(f"    model said {said:<14s} -> actually {truth:<16s} {n:4d}")

    # Confidently wrong is the interesting subset: an error the threshold cannot filter out.
    confident = [r for r in rows
                 if r["model_says"] != r["label"] and r.get("confidence", 0) >= 0.90]
    if confident:
        print(f"  of those, {len(confident)} were held at confidence >= 0.90 -- errors no "
              f"threshold can remove")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True, help="labels.json from label.html")
    ap.add_argument("--merge", type=Path, default=Path("data/annotations.npz"),
                    help="model-labelled set to merge under the hand labels; '' to skip")
    ap.add_argument("--samples", type=Path, default=REPO / "samples")
    ap.add_argument("--out", type=Path, default=Path("data/merged.npz"))
    ap.add_argument("--human-weight", type=int, default=4,
                    help="times each hand label is repeated relative to a model label")
    args = ap.parse_args()

    records = json.loads(args.labels.read_text(encoding="utf-8"))
    records = [r for r in records if r.get("label") in LABELS]
    if not records:
        print("no usable labels in that file", file=sys.stderr)
        return 1
    counts = Counter(r["label"] for r in records)
    print(f"{len(records)} hand labels: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    report_disagreements(records)

    x_human = cut(args.samples, records)
    y_human = np.array([LABELS[r["label"]] for r in records], np.int64)

    xs = [np.repeat(x_human, args.human_weight, axis=0)]
    ys = [np.repeat(y_human, args.human_weight, axis=0)]
    provenance = ["human"] * len(ys[0])

    if args.merge and str(args.merge) and args.merge.exists():
        prior = json.loads(args.merge.with_suffix(".json").read_text(encoding="utf-8"))
        d = np.load(args.merge)
        xm, ym = d["crops"].astype(np.float32), d["labels"].astype(np.int64)

        # Index the human boxes per image so the override check does not become quadratic
        # across the whole set; per page it is a few hundred against a few hundred.
        by_image: dict[str, list[list[int]]] = {}
        for r in records:
            by_image.setdefault(r["image"], []).append(r["bbox"])

        keep = [k for k, rec in enumerate(prior)
                if all(iou(rec["bbox"], b) < SAME_BOX_IOU
                       for b in by_image.get(rec["image"], []))]
        dropped = len(prior) - len(keep)
        xs.append(xm[keep])
        ys.append(ym[keep])
        provenance += ["model"] * len(keep)
        print(f"\nmerged {len(keep)} model labels ({dropped} superseded by a hand label)")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, crops=x, labels=y)
    balance = {NAMES[k]: int((y == k).sum()) for k in sorted(NAMES)}
    print(f"\nwrote {len(y)} training rows to {args.out}")
    print(f"class balance: {balance}")
    print(f"provenance: {Counter(provenance)}")
    print(f"\nnext: python -m training.train --annotations {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
