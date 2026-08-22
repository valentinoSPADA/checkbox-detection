# Checkbox Detection

**A take-home challenge submission for HomeVision.** Not a HomeVision product, not affiliated
with or endorsed by them; their wordmark appears in the UI header and here only to say who the
submission is for. The interface deliberately borrows the visual language of homevision.co —
the palette, the radii, the shadow — because a tool for their reviewers should look like it
belongs next to their product.

Detects and classifies checkboxes in appraisal document images, behind the `POST /detect`
endpoint the challenge specifies.

Detection is a **two-stage local pipeline**: geometry proposes every square region that could
be a checkbox, and a small trained CNN decides which ones are. It runs offline, needs no
credentials of any kind, and costs nothing per page. The Go service has **zero external
dependencies**.

Two model-backed engines existed alongside it and were removed after measurement — a language
model returns plausible box sizes at approximate positions rather than measured ones, which is
a worse answer than the geometry gives for free. `DESIGN.md` keeps the numbers as a rejected
alternative; a paid model is still used **offline**, once, to label training data.

- `DESIGN.md` — every architectural decision with its reason, its cost, and the condition
  that would reverse it, plus the measured experiments that drove them.
- `ARCHITECTURE.md` — folder-by-folder map from the tree to the architecture it implements.

---

## Quick start

```bash
docker compose up --build
```

Then:

- API — <http://localhost:8080>
- Overlay viewer — <http://localhost:5173>

```bash
curl -X POST -F "file=@samples/sample_1_urar_1004.png" http://localhost:8080/detect
```

```json
{ "boxes": [ { "bbox": [1077, 1399, 1129, 1451], "is_checked": false }, … ] }
```

Nothing needs configuring. `.env.example` documents the tunables — the confidence floor and
the upload limits — and every one has a measured default.

**Upload limits**, at the two layers that can each enforce something the other cannot:

| Limit | Where | Default | Why there |
|---|---|---|---|
| Byte size | Go gateway | 25 MB | Bounds the body before it is buffered, using `MaxBytesReader` rather than the caller's `Content-Length`, which is a claim |
| Content type | Go gateway | PNG, JPEG, WEBP, TIFF | **Sniffed from the bytes**, not read from the multipart header — anything at all can arrive labelled `image/png` |
| Pixel count | Python sidecar | 40 MP | Compression ratio is attacker-chosen, so a byte limit cannot bound what an image *costs*; enforced where the decode happens |
| Minimum side | Python sidecar | 32 px | Rejects degenerate inputs before they reach code that assumes a page |

40 MP admits a 600 dpi scan of US Letter (33.7 MP) while refusing the decompression bombs a
byte limit alone lets through. The sample pages are 10.7 MP. The browser mirrors the first two
so an oversized file fails instantly instead of after a long upload — that copy is a courtesy,
not a boundary.

### Running without Docker

```bash
# Detection engine
cd detector && pip install -r requirements.txt
uvicorn app.main:app --port 8000

# API (separate shell)
cd backend && DETECTOR_URL=http://localhost:8000 go run ./cmd/api

# UI (separate shell)
cd frontend && npm install && npm run dev
```

### Tests

```bash
cd backend  && go test ./...   # 87% overall, 98% on the domain package
cd detector && pytest          # 97% on engine/
cd frontend && npm test        # API client
```

CI enforces both Go floors (80% overall, 90% domain) and the detector's (85%). `cmd/api` is
the one package left untested and reports 0%: it is the composition root, and a test asserting
that `main` calls the constructors is a tautology rather than a safeguard.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/detect` | Detect and classify checkboxes in an uploaded image |
| `GET` | `/engines` | Which engines this instance can actually run |
| `GET` | `/health` | Process liveness (does not touch the sidecar) |
| `GET` | `/ready` | Readiness, including sidecar and model state |
| `GET` | `/version` | Build version |

`POST /detect` takes `multipart/form-data` with the image in a field named `file` (`image`,
`document` and `upload` are also accepted). Optional query parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `engine` | `local` | `local`. `GET /engines` reports what a build registers; anything else is a 400, never a silent substitution |
| `min_confidence` | `0.95` | Confidence floor for this request (see below) |
| `verbose` | `false` | Add confidence, engine attribution and pipeline counters |

The `min_confidence` default of 0.95 looks severe and is not: the classifier separates true
checkboxes (0.95-0.98) from noise (0.72-0.82) cleanly, so the floor sits in the gap between
two populations rather than partway up one. Sweeping it on sample 1 (≈120 true checkboxes)
gives 1528 boxes at 0.60, 244 at 0.90, **125 at 0.95**, 65 at 0.97. Label smoothing during
training caps the softmax near 0.98, so 0.99 returns nothing at all.

The default response contains exactly the specified keys and nothing else — a contract test
asserts this, because adding keys to a specified schema is the easiest way to break whatever
parses it. `verbose=true` adds fields without renaming or reordering the specified ones.

---

## How it works

### Stage 1 — geometric proposals (recall)

Binarise with a *local* adaptive threshold (sample 3 puts checkboxes on blue-shaded rows and
sample 4 lays a red watermark across the page; a global threshold destroys both), then
assemble every square region whose four sides each carry a straight ink run of at least 80%
of the side length, sweeping sides from 10 to 70 pixels.

The sweep is what makes it scale-adaptive: page DPI is unknown, and sample 2 is a zoomed crop
whose boxes are several times the size of sample 1's, so no single expected size exists.

This stage is tuned for recall, not precision. It nominates roughly 100k raw regions per
page, collapsed to ~20k distinct candidates. That number is deliberate — a checkbox Stage 1
never nominates is unrecoverable, whereas its false positives are exactly what Stage 2 exists
to remove.

### Stage 2 — learned classification (precision)

A small CNN (~60k parameters, 313 KB as ONNX) scores each candidate crop as
`not_a_checkbox` / `unchecked` / `checked`. Folding both questions into one softmax gives the
policy layer a single comparable confidence to threshold and to rank during suppression.

Training data is **synthetic**, because the challenge supplies four images and no
annotations. The generator deliberately produces every failure mode observed on the real
samples: boxes sharing a border with a table rule, boxes on shaded rows, freehand check marks
and scribbles that overflow the border, and — critically — bordered cells containing text.

### Policy — in Go, not in the engine

Thresholding, suppression and capping live in `backend/internal/domain`, applied
identically to every engine's output. That is what makes an accuracy comparison between
engines meaningful; it also keeps the Go service from degrading into a proxy with no logic
worth testing.

---

## Results

Measured at IoU >= 0.4 by `eval/evaluate.py` calling the **live API**, so what is scored is
the system as shipped, policy and thresholds included.

Against **`eval/ground_truth_human.json`** — 284 checkboxes judged by a person, one crop at a
time, not by a model:

| Sample | GT boxes | Precision | Recall | F1 | Filled/unfilled |
|---|---|---|---|---|---|
| 1 — URAR 1004 | 118 | 1.000 | 1.000 | **1.000** | 1.000 |
| 2 — zoomed crop | 39 | 0.951 | 1.000 | **0.975** | 1.000 |
| 3 — shaded rows | 48 | 0.941 | 1.000 | **0.970** | 1.000 |
| 4 — watermarked | 79 | 1.000 | 1.000 | **1.000** | 1.000 |
| **Total** | **284** | **0.983** | **1.000** | **0.991** | **1.000** |

**These numbers are contaminated and must not be read as generalisation.** 470 of the 627 hand
labels went into training, and the 284 boxes above are the positive half of those same labels.
The table says the detector fits the pages it was fitted on. It is reported because it is the
honest description of *these four documents*, and because the earlier model could not reach it
even on data it had seen.

The number that does measure generalisation is a **leave-one-page-out** run: a model trained
only on samples 1-3 — neither the hand labels nor the Claude labels from sample 4 — scored
against sample 4, the hardest page of the four:

| | GT | Precision | Recall | F1 | Filled/unfilled |
|---|---|---|---|---|---|
| **Sample 4, page never seen in training** | 79 | **1.000** | **1.000** | **1.000** | **1.000** |

Held at 0.80, 0.90 and 0.95 alike; at 0.70 it takes one false positive and at 0.50 three.
Independently, held-out **crop**-level accuracy on 495 real crops the classifier never saw is
**0.996**, with zero checked/unchecked confusion.

Two limits on even the leave-one-page-out figure, stated because they are not obvious:

* **Stage 1 saw the page.** Its size sweep was calibrated on all 284 hand-confirmed boxes,
  sample 4's included. So this measures an unseen page against an unseen *classifier*, not an
  unseen system. Four pages is not enough data to separate the two and still have any left.
* **Recall is relative to the candidate pool.** The 79 boxes are the ones a person confirmed
  out of what Stage 1 nominated; a checkbox never proposed cannot appear. Stage 1's own recall
  was measured separately and is 284/284 on the confirmed set.

The previous, model-generated ground truth (`eval/ground_truth.json`, 400 boxes) is kept for
comparison and is measurably worse: on sample 4 it holds **44 boxes that are not checkboxes**,
every one of them 10 px, all of them letter counters the human rejected. Scoring against it
put sample 4's recall at 0.650 — most of the "misses" were things that should never have been
found.

Reading these honestly:

- **Precision is where the work went.** An earlier build scored 0.622 precision / 0.684 F1.
  Three changes moved it to 0.930 / 0.801, and none touched the architecture: calibrating the
  confidence floor (§7 of `DESIGN.md`), adding the page's black section rail to the training
  generator as a negative class (§10), and retraining on real labelled crops instead of purely
  imagined ones. All three were bugs in what the model had been *taught* and *thresholded at*,
  not in how the system is built.
- **Recall is now the weak point, and it is concentrated in sample 2.** 0.494 there against
  0.823 on sample 1. Sample 2 is a zoomed crop, so its checkboxes are far larger than the rest
  and sit near the top of the 10–70 px range Stage 1 sweeps. The honest reading is that the
  size range is tuned for full pages and thins out at the end where this sample lives — a
  fixable proposal-stage limit rather than anything the classifier decides.
- **A threshold belongs to a model, not to a problem.** This was learned the expensive way:
  the retrained model first measured *worse* (F1 0.801 against 0.819) purely because it
  inherited a floor calibrated for its predecessor. `DefaultPolicy` now carries the sweep and
  the instruction to redo it after any retrain.
- **Ground truth is model-generated, and it is measurably imperfect.** It is rebuilt with the
  candidate marked in the crop (see below), which halved the labels the pixels contradict —
  from 14.8% to 8.2% on "checked", and 14.7% to 8.8% on "unchecked". The residue is real: some
  of what still counts as a miss is not one. Every figure in the table is therefore a floor.

  The defect worth reading about, because it was found by looking rather than by a metric: all
  14 filled/unfilled disagreements on sample 1 ran one way, and all 14 boxes contained
  *exactly 0.0% ink inside their own border*, each sitting directly above or below a box
  carrying an X. The adjudicator saw a crop at 3.0× the candidate's size — two or three
  checkboxes on a form this dense — and answered about the marked neighbour. The prompt
  already said to ignore the edges; that could not carry it, because "the centre one" is not
  decidable when boxes tile the region. The fix draws the referent instead of describing it:
  `engine.preprocess.mark_candidate` rings the candidate in red before the crop is sent.

### Two caveats that matter more than the numbers

**The ground truth is model-generated.** It was built by running the detector at a
permissive threshold and having Claude adjudicate each candidate
(`eval/build_ground_truth.py`), because the challenge ships four images and no annotations.
Two consequences follow, and neither is small:

1. *Recall is relative, not absolute.* Ground truth derived from the detector's own candidate
   pool cannot contain a checkbox the detector never proposed. The recall column measures how
   much of what Stage 1 found survives to the response — a real and useful quantity, but a
   more flattering one than recall against independent annotation.
2. *It is measurably wrong in places.* On sample 4 it holds 44 boxes that are not checkboxes,
   every one exactly 10 px — letter counters a person rejected on sight. This is why
   `eval/ground_truth_human.json` exists and is the reference the results table uses.

A contact sheet of the adjudications is written to `docs/ground_truth_preview.png` and was
reviewed by eye before the labels were used, because model-produced labels are of unknown
quality until somebody actually looks at them.

## Why it is built this way

The full reasoning, with tradeoffs and reversal conditions, is in `DESIGN.md`. The short
version of the two decisions most worth questioning:

**Why not a single end-to-end detector (YOLO-class)?** No labelled data. With real
annotations an end-to-end detector would very likely beat this two-stage pipeline and would
remove Stage 1's recall ceiling. Without them, a two-stage design lets the geometry — which
needs no training — carry recall, and confines the learned part to a problem small enough to
be trained on synthesis.

**Why is a machine-learning stage here at all?** Because three purely geometric detector
generations were built and measured first, and each traded recall against precision without
winning both (the log is in `DESIGN.md` §0 and `docs/prototype-log.md`). Geometry cannot
separate "checkbox" from "checkbox-sized table cell", and an ink-density threshold classifies
a freehand check mark badly. The ML stage answers a measured failure rather than a
preference.

---

## Known limitations

Stated plainly, because the challenge asks for them and because the honest ones are more
useful than a flattering summary.

- **The synthetic-to-real gap is narrowed, not closed.** The classifier now trains on 1832
  real crops as well as synthetic ones, which is what took precision to 0.930 and made the
  operating point stable across 0.70–0.90 instead of collapsing 35 points over a 0.05 move.
  But 1832 crops come from *four pages of one document family*, and held-out real accuracy is
  0.9651 — the remaining errors are letter counters (`O`, `0`, `p`) read as empty boxes. A
  different form family would need its own labelling pass.
- **The labels themselves are ~90% accurate**, audited by hand at 32 crops per positive class.
  That is weak supervision working as intended, not a clean dataset, and it puts a ceiling on
  what the model can learn. It is also why every verdict is checked against a pixel
  measurement before it enters training: the failure mode of a noisy annotator is confident
  wrongness, and only an independent source catches that.
- **Stage 1 caps recall.** A checkbox whose border is broken by more than two pixels, or
  which is smaller than 10 px or larger than 70 px, is never nominated and cannot be
  recovered.
- **Only square-ish candidates are swept.** Markedly rectangular checkboxes rely on the
  span tolerance and the classifier's aspect jitter rather than on being searched for.
- **No page-level consistency constraint.** Every candidate is judged independently, yet on a
  real form every checkbox is nearly the same size and they align into rows. That structure
  is currently unused.
- **Three checkbox-shaped table cells survive on sample 3**, all at 0.977 confidence: a cell
  in a repeated column, immediately left of a real checkbox. Both stages are behaving
  correctly — it is a 45 px square with four inked sides, and inside the classifier's window a
  blank cell and a blank checkbox are the same drawing. The evidence that separates them is
  page-level (the same X position repeating down three rows is a *column*), which is exactly
  the structure listed above as unused. No threshold removes them.
- **Four pages from one document family** cannot establish how this generalises, whatever the
  numbers say.

---

## What the challenge required (and is implemented)

- `POST /detect` accepting a document image as a file upload.
- Detection of checkbox locations, filled and unfilled, as pixel bounding boxes.
- Classification of each as filled or unfilled.
- The exact specified response shape, `{"boxes": [{"bbox": [x1,y1,x2,y2], "is_checked": …}]}`,
  enforced by a contract test.
- Build and run instructions (above) — one command for the whole stack.
- This writeup: approach, tradeoffs (`DESIGN.md`), and known limitations (above).
- `TODO(production)` comments marking work deliberately left out of the timebox.

## Extras added beyond the requirements

Each tied to a signal in the job description rather than added for its own sake.

- **A Detector port with the engine chosen per request** — directly answers *"Bridge the AI
  Gap: integrate AI services into our core SaaS platform"*. It carried three real
  implementations and now carries one; **removing two of them touched the composition root
  and nothing else**, which is a better demonstration that the seam is real than having three
  was.
- **A trained model rather than tuned thresholds**, with the training pipeline, the data
  generator and the exported artifact all in the repository and reproducible.
- **Claude as an annotator, run rather than proposed** (`detector/training/annotate.py`) —
  1832 real crops labelled, every verdict gated against an independent pixel measurement. The
  expensive model is offline and never in the request path, and the dataset is committed so
  the model retrains with no credentials.
- **A hand-labelling tool** (`detector/training/make_labeling_task.py`) — 627 candidates
  labelled by a person, which is what found the real defect: the classifier sees a fixed
  40×40 crop and is therefore blind to absolute size, so a 10 px letter counter and a 50 px
  checkbox arrive identical. No amount of model-generated labelling finds that, because the
  models share the blind spot.
- **Upload guards at the layer that can enforce each one** — byte size and sniffed content
  type at the gateway, pixel bounds where the decode happens.
- **React + TypeScript overlay viewer** — for a detector, the overlay *is* the demo, and it
  answers the role's opening line about interfaces that let humans act on AI decisions.
- **Structured JSON logging, `/health` and `/ready` split correctly**, graceful shutdown,
  panic recovery, upload bounds, and a request-scoped timeout — the JD's reliability and
  observability line.
- **Docker Compose bringing up all three services**, CI running lint, typecheck, tests,
  coverage gates, SonarCloud duplication analysis, image builds, and a composed-stack smoke
  test that actually calls `/detect`.
- **An evaluation harness plus a ground truth to run it against** (`eval/evaluate.py`,
  `eval/build_ground_truth.py`) — it scores the live API, so what is measured is the system as
  shipped rather than a convenient internal function, and the annotation gap is closed by
  Claude rather than left as "we had no labels".

## What I'd add with more time / in production

- **More real labels, from more document families.** The pass that is done covers four pages
  of one form family; the residual errors are letter counters, which more labels directly
  address. This is still the highest-value remaining change.
- **Widen or adapt the Stage 1 size sweep.** Recall on sample 2 is 0.494 against 0.823 on
  sample 1, and sample 2 is a zoomed crop whose boxes sit at the top of the 10–70 px range.
  Estimating page scale first and sweeping around it would cost less than the current fixed
  range and cover more.
- **Then replace the two-stage pipeline with a fine-tuned end-to-end detector.** Once real
  labels exist, the Stage 1 recall ceiling stops being worth paying for.
- **Exploit page-level structure**: checkboxes on one form share a size and align into rows
  and columns. A consistency pass over the candidate set would remove false positives that no
  per-crop classifier can, because the evidence is not in the crop.
- **AWS production path.** The JD names Lambda, SQS and S3. The natural shape: S3 upload
  triggers an SQS message; the Go API runs on ECS Fargate behind an ALB; the detector runs as
  a separately-scaled Fargate service (CPU-bound, different scaling curve) or as a Lambda
  container image for spiky loads; results land in DynamoDB keyed by document hash, which also
  gives idempotency for retried pages. Terraform modules per service, one workspace per
  environment. Not built here: it would consume the timebox on infrastructure that is not what
  the challenge grades.
- **Real observability**: OpenTelemetry traces spanning gateway → sidecar → inference, and
  RED metrics per engine. Stage 1's proposal count deserves a metric of its own: it is the
  input to everything downstream, and a page where it collapses or explodes is the earliest
  signal that a new document family has arrived.
- **Caching by content hash.** Appraisal pages are re-submitted often, and detection is
  seconds of CPU; a hash-keyed cache removes that for every repeat.
- **Batch endpoint and back-pressure.** A whole loan file is dozens of pages; a batch API with
  a bounded queue would beat dozens of independent requests.
