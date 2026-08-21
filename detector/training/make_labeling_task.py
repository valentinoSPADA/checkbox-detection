"""Build a self-contained HTML page for labelling checkbox candidates by hand.

    python -m training.make_labeling_task --out data/label.html

Why this exists. Every label in this repository so far was produced by a model, and models
have a stubborn, systematic idea of what a checkbox is: two prompt revisions and a pixel gate
took the annotator from roughly 10% to roughly 90% accuracy on the positive classes, and the
residue is not random noise -- it is letter counters (`O`, `0`, `p`) confidently called empty
boxes. Weak supervision cannot repair that, because the errors are exactly where the model's
concept differs from a person's, and averaging more of the same model does not move a concept.

A few hundred human labels do. This tool produces the page to make them on: every candidate
the detector surfaces, rendered large, with the candidate under judgement ringed, three keys
to answer with, and an export at the end.

Two decisions worth stating:

* **Labelling is blind.** The model's own verdict and confidence are not shown. They are in
  the export for later analysis, but showing them would anchor the labeller to the very
  opinion this exercise exists to overrule -- and a human agreeing with a model they were
  just shown is not an independent label.
* **The pool is the surfaced candidates plus a slice of rejects.** The surfaced ones fix
  precision and the filled/unfilled call. The rejects are the only way a *miss* can be
  recorded at all: a checkbox the classifier scored at 0.01 is invisible to any process that
  only looks at what the detector returned.

The page writes progress to localStorage after every answer, so closing the tab loses nothing
and the work can be done in several sittings. Nothing is uploaded anywhere; it is a local file
that runs with no network at all.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.classifier import CheckboxClassifier
from engine.pipeline import FLOOR, DetectionPipeline
from engine.preprocess import decode, to_gray

REPO = Path(__file__).resolve().parent.parent.parent

# Two views per candidate. One is not enough: tight tells you whether the interior carries a
# mark, wide tells you whether the thing is a checkbox or a table cell, and those are the two
# questions being asked. Rendering both removes the zooming that makes hand-labelling slow.
TIGHT_CONTEXT, WIDE_CONTEXT = 1.8, 5.0
TIGHT_PX, WIDE_PX = 168, 168

# "Surfaced" means what the service actually hands to the policy layer -- the pipeline's own
# candidate floor. Going below it does not widen the decision surface, it just adds the noise
# the pipeline already declines to return: at 0.02 this pool is 2059 candidates on sample 1
# against 143 at the floor, and none of the extra 1900 is a checkbox anybody would argue about.
SURFACED_FLOOR = FLOOR

# Matches domain.DefaultPolicy's IoUThreshold, so the labeller sees one entry per checkbox for
# the same reason the API returns one.
SUPPRESS_IOU = 0.30


def crop(gray: np.ndarray, bbox: list[int], context: float, out: int, ring: bool) -> np.ndarray:
    """Cut a square view centred on bbox, optionally ringing the candidate.

    Edges are replicated rather than zero-padded, for the same reason as everywhere else in
    this codebase: a candidate near the page margin must not acquire a black border that reads
    as a rule that is not there.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(8, int(max(x2 - x1, y2 - y1) * context / 2))
    pad = half + 4
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    patch = padded[cy - half + pad:cy + half + pad, cx - half + pad:cx + half + pad]
    if patch.size == 0:
        patch = np.full((out, out), 255, np.uint8)
    scale = out / patch.shape[0]
    # INTER_NEAREST on upscale: these crops are 30-300 px of scanned line art, and smoothing
    # a 2 px rule into a grey smear is exactly the information the labeller needs kept.
    view = cv2.resize(patch, (out, out), interpolation=cv2.INTER_NEAREST)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    if ring:
        h = int((x2 - x1) * scale / 2), int((y2 - y1) * scale / 2)
        c = out // 2
        cv2.rectangle(view, (c - h[0] - 3, c - h[1] - 3), (c + h[0] + 3, c + h[1] + 3),
                      (0, 0, 255), 2)
    return view


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to encode crop")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def suppress(items: list[dict], iou_min: float = SUPPRESS_IOU) -> list[dict]:
    """Greedy overlap suppression, highest confidence first.

    Mirrors what the Go policy does at runtime. It is repeated here rather than imported
    because this tool must run without the API up, and because a labeller shown the same
    checkbox three times at three sizes will (rightly) stop trusting the tool.
    """
    out: list[dict] = []
    for item in sorted(items, key=lambda i: -i["confidence"]):
        if all(_iou(item["bbox"], k["bbox"]) < iou_min for k in out):
            out.append(item)
    return out


def _iou(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def collect(pipeline: DetectionPipeline, image_path: Path, rejects: int,
            rng: np.random.Generator) -> list[dict]:
    """Every surfaced candidate on one page, plus a sample of what the classifier rejected."""
    data = image_path.read_bytes()
    gray = to_gray(decode(data))
    # floor=0.0: the classifier's own confidence decides nothing here. Its opinion is recorded
    # for later analysis but must not silently pre-filter the pool a person is asked to judge.
    result = pipeline.run(data, floor=0.0)

    surfaced, low = [], []
    for c in result.candidates:
        item = {"image": image_path.name, "bbox": list(c.bbox),
                "confidence": round(float(c.confidence), 4),
                "model_says": "checked" if c.is_checked else "unchecked",
                "p_negative": round(float(c.p_negative), 4)}
        (surfaced if c.confidence >= SURFACED_FLOOR else low).append(item)

    keep = suppress(surfaced)
    extra: list[dict] = []
    if rejects and low:
        # Sampled from the rejects nearest the floor. A uniform draw over all rejects is
        # essentially all obvious paper and would spend the labeller's attention on crops whose
        # answer is never in doubt; the ones just below the floor are where a real miss hides.
        low.sort(key=lambda i: -i["confidence"])
        low = suppress(low)
        pool = low[:max(rejects * 6, rejects)]
        picked = [pool[i] for i in rng.choice(len(pool), min(rejects, len(pool)), replace=False)]
        for item in picked:
            item["model_says"] = "rejected"
            if all(_iou(item["bbox"], k["bbox"]) < SUPPRESS_IOU for k in keep):
                extra.append(item)

    items = keep + extra
    for item in items:
        view = crop(gray, item["bbox"], TIGHT_CONTEXT, TIGHT_PX, ring=True)
        wide = crop(gray, item["bbox"], WIDE_CONTEXT, WIDE_PX, ring=True)
        item["tight"] = png_b64(view)
        item["wide"] = png_b64(wide)
    print(f"{image_path.name}: {len(keep)} surfaced + {len(extra)} rejects = {len(items)}")
    return items


def build_html(items: list[dict], title: str) -> str:
    """Render the labelling page.

    Self-contained by necessity, not by preference: it has to open from the filesystem with no
    server and no network, so the crops are inlined as data URIs and there is no dependency to
    fail to load.
    """
    # The crops are stripped from the payload the page exports, so an exported file is a few
    # hundred KB of decisions rather than a copy of the images.
    payload = json.dumps(items, separators=(",", ":"))
    return _TEMPLATE.replace("__TITLE__", title).replace("__ITEMS__", payload)


_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg:#f8fafc; --card:#fff; --ink:#18181b; --muted:#71717a; --line:#e4e4e7;
    --accent:#4f46e5; --ok:#0d9488; --no:#b91c1c;
    --shadow: 0 1px 2px rgba(16,24,40,.06), 0 4px 12px rgba(16,24,40,.06);
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#18181b; --card:#27272a; --ink:#fafafa; --muted:#a1a1aa; --line:#3f3f46; }
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 Inter,system-ui,-apple-system,Segoe UI,sans-serif; }
  header { display:flex; align-items:center; justify-content:space-between; gap:16px;
           padding:12px 20px; border-bottom:1px solid var(--line); background:var(--card); }
  h1 { font-size:15px; margin:0; letter-spacing:-.01em }
  .bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; flex:1;
         max-width:420px }
  .bar i { display:block; height:100%; background:var(--accent); width:0 }
  main { max-width:820px; margin:0 auto; padding:22px 20px 60px }
  .views { display:flex; gap:16px; justify-content:center; flex-wrap:wrap }
  figure { margin:0; text-align:center }
  figure img { display:block; border:1px solid var(--line); border-radius:8px;
               image-rendering:pixelated; background:#fff; width:336px; height:336px }
  figcaption { font-size:12px; color:var(--muted); margin-top:6px }
  .keys { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:22px }
  button { font:inherit; font-weight:600; padding:16px 12px; border-radius:8px;
           border:1px solid var(--line); background:var(--card); color:var(--ink);
           cursor:pointer; box-shadow:var(--shadow) }
  button:hover { border-color:var(--accent) }
  button b { display:block; font-size:20px; margin-bottom:2px }
  .k1 b { color:var(--no) } .k2 b { color:var(--muted) } .k3 b { color:var(--ok) }
  .row { display:flex; gap:10px; margin-top:14px; align-items:center; flex-wrap:wrap }
  .ghost { padding:9px 14px; font-weight:500; box-shadow:none }
  .muted { color:var(--muted); font-size:13px }
  .done { text-align:center; padding:40px 0 }
  textarea { width:100%; height:180px; margin-top:12px; font-family:ui-monospace,Menlo,monospace;
             font-size:11px; border:1px solid var(--line); border-radius:8px; padding:10px;
             background:var(--card); color:var(--ink) }
  .tally { display:flex; gap:14px; font-size:13px; color:var(--muted) }
  kbd { font:12px ui-monospace,monospace; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:4px; padding:1px 5px; color:var(--muted) }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="bar"><i id="fill"></i></div>
  <div class="tally">
    <span id="count">0 / 0</span>
    <span id="tally"></span>
  </div>
</header>

<main>
  <div id="task">
    <div class="views">
      <figure><img id="tight" alt="candidato, vista cercana">
        <figcaption>cerca &mdash; &iquest;tiene marca adentro?</figcaption></figure>
      <figure><img id="wide" alt="candidato, vista amplia">
        <figcaption>lejos &mdash; &iquest;es una casilla o una celda?</figcaption></figure>
    </div>

    <p class="muted" style="text-align:center;margin:16px 0 0">
      Juzg&aacute; <strong>solo lo que est&aacute; dentro del recuadro rojo</strong>.
      Las marcas en las casillas vecinas no cuentan.
    </p>

    <div class="keys">
      <button class="k1" onclick="answer('not_a_checkbox')"><b>1</b>No es checkbox</button>
      <button class="k2" onclick="answer('unchecked')"><b>2</b>Casilla vac&iacute;a</button>
      <button class="k3" onclick="answer('checked')"><b>3</b>Casilla marcada</button>
    </div>

    <div class="row">
      <button class="ghost" onclick="undo()">&larr; Deshacer <kbd>Z</kbd></button>
      <button class="ghost" onclick="answer('skip')">Saltar <kbd>S</kbd></button>
      <button class="ghost" onclick="save()">Descargar avance</button>
      <span class="muted" id="hint">Se guarda solo en este navegador,
        despu&eacute;s de cada respuesta.</span>
    </div>
  </div>

  <div id="finished" class="done" hidden>
    <h2>Listo &mdash; <span id="total"></span> etiquetas</h2>
    <p class="muted">Descarg&aacute; el archivo y pas&aacute;melo. Si la descarga no funciona,
       copi&aacute; el texto de abajo.</p>
    <button onclick="save()">Descargar labels.json</button>
    <button class="ghost" onclick="restart()">Volver a empezar</button>
    <textarea id="out" readonly></textarea>
  </div>
</main>

<script>
const ITEMS = __ITEMS__;
const KEY = "checkbox-labels-v1";
let answers = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;

// Resume where the last sitting stopped rather than at zero: this is a few hundred decisions
// and being sent back to the start after closing a tab is how a labelling task gets abandoned.
function firstUnanswered() {
  for (let k = 0; k < ITEMS.length; k++) if (!(k in answers)) return k;
  return ITEMS.length;
}

function render() {
  const done = Object.keys(answers).length;
  document.getElementById("fill").style.width = (100 * done / ITEMS.length) + "%";
  document.getElementById("count").textContent = done + " / " + ITEMS.length;
  const t = {checked:0, unchecked:0, not_a_checkbox:0, skip:0};
  for (const k in answers) t[answers[k]] = (t[answers[k]] || 0) + 1;
  document.getElementById("tally").textContent =
    t.checked + " marcadas &middot; " + t.unchecked + " vacias &middot; " + t.not_a_checkbox + " no"
    + (t.skip ? " &middot; " + t.skip + " saltadas" : "");
  document.getElementById("tally").innerHTML = document.getElementById("tally").textContent;

  if (i >= ITEMS.length) {
    document.getElementById("task").hidden = true;
    document.getElementById("finished").hidden = false;
    document.getElementById("total").textContent = done;
    document.getElementById("out").value = exportJSON();
    return;
  }
  document.getElementById("task").hidden = false;
  document.getElementById("finished").hidden = true;
  document.getElementById("tight").src = "data:image/png;base64," + ITEMS[i].tight;
  document.getElementById("wide").src  = "data:image/png;base64," + ITEMS[i].wide;
}

function answer(label) {
  if (i >= ITEMS.length) return;
  answers[i] = label;
  localStorage.setItem(KEY, JSON.stringify(answers));
  i++;
  render();
}

function undo() {
  if (i === 0) return;
  i--;
  delete answers[i];
  localStorage.setItem(KEY, JSON.stringify(answers));
  render();
}

function exportJSON() {
  // The crops are deliberately dropped here. They came from the sample images and putting
  // them back into the export would turn a list of decisions into a copy of the dataset.
  const out = [];
  for (const k in answers) {
    if (answers[k] === "skip") continue;
    const it = ITEMS[k];
    out.push({image: it.image, bbox: it.bbox, label: answers[k],
              model_says: it.model_says, confidence: it.confidence});
  }
  return JSON.stringify(out, null, 1);
}

function save() {
  const blob = new Blob([exportJSON()], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function restart() {
  if (!confirm("Se borran todas las respuestas. Seguro?")) return;
  answers = {}; i = 0;
  localStorage.removeItem(KEY);
  render();
}

addEventListener("keydown", e => {
  if (e.key === "1") answer("not_a_checkbox");
  else if (e.key === "2") answer("unchecked");
  else if (e.key === "3") answer("checked");
  else if (e.key.toLowerCase() === "s") answer("skip");
  else if (e.key.toLowerCase() === "z") undo();
});

i = firstUnanswered();
render();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, default=REPO / "samples")
    ap.add_argument("--rejects", type=int, default=45,
                    help="rejected proposals sampled per page, so misses can be recorded")
    ap.add_argument("--out", type=Path, default=Path("data/label.html"))
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pipeline = DetectionPipeline(CheckboxClassifier())
    items: list[dict] = []
    for image_path in sorted(args.samples.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        items.extend(collect(pipeline, image_path, args.rejects, rng))

    if not items:
        print("no candidates found", file=sys.stderr)
        return 1

    # Interleaved across pages rather than page by page. Labelling one page end to end invites
    # a rhythm -- the same layout, the same answer -- and a rhythm is how a labeller stops
    # looking. Shuffling also spreads any drift in judgement evenly over all four pages
    # instead of concentrating it in whichever one was labelled last.
    order = rng.permutation(len(items))
    items = [items[k] for k in order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(items, "Etiquetado de checkboxes"), encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {len(items)} candidates to {args.out} ({size_mb:.1f} MB)")
    print("open it in a browser, label, then export labels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
