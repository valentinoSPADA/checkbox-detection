"""Train the Stage 2 classifier on synthetic crops and export it to ONNX.

Run from the ``detector`` directory:

    python -m training.train --train 80000 --val 10000 --epochs 8

Why ONNX rather than shipping a .pt and importing torch at runtime: PyTorch is roughly a
gigabyte of wheel that the serving image would otherwise carry for no benefit. onnxruntime
is ~15 MB, starts faster, and holds the inference contract stable regardless of what the
training environment upgrades to. Training dependencies therefore live in
``requirements-train.txt`` and are absent from the runtime image entirely.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.model import CLASS_NAMES, INPUT_SIZE, NUM_CLASSES, CheckboxNet
from training.synth import GenConfig, SyntheticGenerator

DEFAULT_OUT = Path("models/checkbox_cls.onnx")


def build_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset split. Validation uses a different seed so the val set is drawn
    from the same distribution but shares no individual sample with training."""
    gen = SyntheticGenerator(GenConfig(seed=seed))
    return gen.dataset(n)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, np.ndarray]:
    """Return overall accuracy and a (3,3) confusion matrix indexed [true, predicted].

    The confusion matrix, not accuracy, is the number that matters here: confusing
    unchecked with checked is a wrong answer on a real field, while confusing either with
    not_a_checkbox merely drops a detection. Those two errors have different costs and the
    scalar hides that.
    """
    model.eval()
    correct = total = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum())
            total += int(yb.numel())
            for t, p in zip(yb.cpu().numpy(), pred.cpu().numpy(), strict=False):
                cm[t, p] += 1
    return correct / max(1, total), cm


def export_onnx(model: nn.Module, out: Path) -> None:
    """Write the ONNX graph with a dynamic batch axis.

    The batch axis must be dynamic because the number of proposals varies per page -- a
    fixed batch would force padding on every request and waste compute on blank crops.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["crops"],
        output_names=["logits"],
        dynamic_axes={"crops": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    _consolidate(out)


def _consolidate(out: Path) -> None:
    """Rewrite the graph with its weights inline, as a single file.

    The exporter may place tensors in a sibling ``.onnx.data`` file. That is right for
    multi-gigabyte models and wrong for this one: a two-file artifact is one more thing to
    forget when copying into an image, and forgetting it fails at load time with a message
    that does not mention the missing sidecar file. At ~300 KB there is no reason to split.
    """
    external = out.with_suffix(out.suffix + ".data")
    if not external.exists():
        return
    import onnx  # imported lazily: only the training environment has it

    model = onnx.load(str(out))  # resolves the external tensors from disk
    onnx.save_model(model, str(out), save_as_external_data=False)
    external.unlink()


def _force_utf8_stdout() -> None:
    """Make stdout able to carry non-ASCII, on any platform.

    Not cosmetic. torch's ONNX exporter writes progress lines containing emoji, and a Windows
    console defaults to cp1252, which cannot encode them -- so the export raises
    UnicodeEncodeError and eight minutes of training are lost at the last step, with a
    traceback that points at a print statement rather than at anything to do with the model.
    Errors are replaced rather than raised: a mangled progress character is not worth failing
    a training run over.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=80_000, help="synthetic training samples")
    ap.add_argument("--val", type=int, default=10_000, help="synthetic validation samples")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--annotations", type=Path, default=None,
                    help="npz from training.annotate: real crops labelled by Claude")
    ap.add_argument("--real-weight", type=int, default=12,
                    help="times each real crop is repeated in the training set")
    ap.add_argument("--real-holdout", type=float, default=0.25,
                    help="fraction of real crops kept out of training, for honest validation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    xtr, ytr = build_split(args.train, args.seed)
    xva, yva = build_split(args.val, args.seed + 9_999)
    print(f"generated {args.train}+{args.val} crops in {time.time() - t0:.1f}s")

    real_va = None
    if args.annotations:
        # Real crops from the four supplied pages, labelled by Claude and filtered against the
        # pixels (see training/annotate.py). They are the point of this whole exercise: the
        # synthetic generator can only contain failure modes someone thought of, and the first
        # model trained purely on it scored 0.9972 on synthetic validation while reporting ten
        # thousand filled checkboxes per real page.
        d = np.load(args.annotations)
        xr, yr = d["crops"].astype(np.float32), d["labels"].astype(np.int64)
        if xr.ndim == 3:
            xr = xr[:, None]
        # Per-row weight, when the file carries one: hand labels are more trustworthy than
        # model labels and are repeated more, but the *file* must hold one row per crop so
        # that the split below can be honest. See training/import_labels.py.
        wr = d["weights"].astype(np.int64) if "weights" in d.files else np.ones(len(yr), np.int64)

        # Split BEFORE oversampling. Repeating first and splitting after puts copies of the
        # same crop on both sides and turns validation into a memorisation check -- which is
        # not hypothetical: it happened, and read 0.9990 on held-out "real" crops.
        rs = np.random.default_rng(args.seed).permutation(len(xr))
        cut = int(len(xr) * args.real_holdout)
        hold, keep = rs[:cut], rs[cut:]

        # Oversampled because a couple of thousand real crops against 80k synthetic would
        # otherwise contribute almost nothing to the gradient. Repetition rather than a class
        # weight, so the augmentation already baked into the synthetic pipeline is not applied
        # to them a second time.
        reps = np.repeat(keep, wr[keep] * args.real_weight)
        xr_tr = xr[reps]
        yr_tr = yr[reps]
        xtr = np.concatenate([xtr, xr_tr])
        ytr = np.concatenate([ytr, yr_tr])
        real_va = DataLoader(TensorDataset(torch.from_numpy(xr[hold]),
                                           torch.from_numpy(yr[hold])), batch_size=512)
        # Both numbers, because they answer different questions: how much real data exists,
        # and how loudly it speaks in the loss. A per-row weight multiplies --real-weight, so
        # hand labels at weight 4 end up repeated 4 x real_weight times.
        print(f"mixed in {len(keep)} distinct real crops -> {len(reps)} training rows "
              f"(base x{args.real_weight}, per-row weight {wr[keep].min()}-{wr[keep].max()}), "
              f"holding out {len(hold)} distinct crops for real validation")

    tr = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                    batch_size=args.batch, shuffle=True, drop_last=True)
    va = DataLoader(TensorDataset(torch.from_numpy(xva), torch.from_numpy(yva)),
                    batch_size=512)

    model = CheckboxNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(tr))
    # Label smoothing: the synthetic labels are certain but the *concept* boundary is not
    # (a box with a speck of dust is genuinely ambiguous), and over-confident logits would
    # make the downstream confidence threshold meaningless.
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
        acc, cm = evaluate(model, va, device)
        line = f"epoch {ep}/{args.epochs}  loss={run / len(tr):.4f}  synth_val={acc:.4f}"
        if real_va is not None:
            # The number that matters. Synthetic validation saturates near 0.997 whatever the
            # model has actually learned, so it is reported but not believed.
            racc, _ = evaluate(model, real_va, device)
            line += f"  REAL_val={racc:.4f}"
        print(line)

    acc, cm = evaluate(model, va, device)
    print("\nconfusion matrix [true rows x predicted cols]:")
    print(f"{'':>16s}" + "".join(f"{c:>16s}" for c in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>16s}" + "".join(f"{v:>16d}" for v in cm[i]))
    print(f"\nfinal synthetic val accuracy {acc:.4f}")
    if real_va is not None:
        # Reported last and read first. Synthetic validation saturates near 0.997 whatever the
        # model has actually learned about real pages, so this matrix is the one that says
        # whether the retraining worked.
        racc, rcm = evaluate(model, real_va, device)
        print("\nheld-out REAL crops [true rows x predicted cols]:")
        print(f"{'':>16s}" + "".join(f"{c:>16s}" for c in CLASS_NAMES))
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:>16s}" + "".join(f"{v:>16d}" for v in rcm[i]))
        print(f"\nfinal REAL val accuracy {racc:.4f}")

    export_onnx(model, args.out)
    print(f"exported {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
