# Checkbox Detection

Detects and classifies checkboxes in appraisal document images, behind the `POST /detect`
endpoint the challenge specifies.

Three detection engines share one interface: a **local** two-stage pipeline (geometric
proposals plus a trained CNN), **Claude vision** reading tiled page images directly, and an
**assisted** mode that runs the local pipeline and escalates only its uncertain candidates to
Claude. The local engine is the default and needs no credentials, so the system runs fully on
a clean clone.

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
{ "boxes": [ { "bbox": [612, 1310, 656, 1354], "is_checked": true }, … ] }
```

To enable the Claude-backed engines, copy `.env.example` to `.env` and set
`ANTHROPIC_API_KEY`. Without it, `GET /engines` reports only `local` and asking for another
engine returns 400 rather than silently running a different one.

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
cd backend  && go test ./...          # domain, policy, HTTP contract
cd detector && pytest                 # Stage 1 proposals, preprocessing
cd frontend && npm test               # API client
```

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
| `engine` | `local` | `local`, `vlm`, or `assisted` |
| `min_confidence` | `0.60` | Confidence floor for this request |
| `verbose` | `false` | Add confidence, engine attribution and pipeline counters |

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

Suppression, thresholding, escalation and merging live in `backend/internal/domain`, applied
identically to every engine's output. That is what makes an accuracy comparison between
engines meaningful; it also keeps the Go service from degrading into a proxy with no logic
worth testing.

---

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

- **The synthetic-to-real gap is the dominant error source.** The classifier reaches 99.7% on
  held-out *synthetic* validation and is substantially worse on the real samples: it
  over-predicts `checked`, because a bordered region containing ink is exactly what both a
  filled checkbox and a populated table cell look like. Synthesis can only contain failure
  modes someone thought of. `detector/training/annotate.py` addresses this by having Claude
  label real Stage 1 proposals for retraining — see *What I'd do next*.
- **Stage 1 caps recall.** A checkbox whose border is broken by more than two pixels, or
  which is smaller than 10 px or larger than 70 px, is never nominated and cannot be
  recovered.
- **Only square-ish candidates are swept.** Markedly rectangular checkboxes rely on the
  span tolerance and the classifier's aspect jitter rather than on being searched for.
- **No page-level consistency constraint.** Every candidate is judged independently, yet on a
  real form every checkbox is nearly the same size and they align into rows. That structure
  is currently unused.
- **The `vlm` engine is slow and paid**: a page is split into eight tiles and each is a model
  call. It is a strategy to select deliberately, not a default.
- **Evaluation rests on a small, semi-automatically annotated ground truth** derived from the
  four supplied images. Four pages from one document family cannot establish how this
  generalises.

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

- **Three interchangeable engines behind one port**, selectable per request — directly
  answers *"Bridge the AI Gap: integrate AI services into our core SaaS platform"*. Two-plus
  real implementations are also what make the hexagonal boundary genuine rather than
  decorative.
- **A trained model rather than tuned thresholds**, with the training pipeline, the data
  generator and the exported artifact all in the repository and reproducible.
- **Claude as an annotator** (`detector/training/annotate.py`) — weak supervision to close the
  synthetic-to-real gap, with the expensive model offline and out of the request path.
- **React + TypeScript overlay viewer** — for a detector, the overlay *is* the demo, and it
  answers the role's opening line about interfaces that let humans act on AI decisions.
- **Structured JSON logging, `/health` and `/ready` split correctly**, graceful shutdown,
  panic recovery, upload bounds, and a request-scoped timeout — the JD's reliability and
  observability line.
- **Docker Compose bringing up all three services**, CI running lint, typecheck, tests,
  coverage gates, SonarCloud duplication analysis, image builds, and a composed-stack smoke
  test that actually calls `/detect`.
- **An evaluation harness** (`eval/evaluate.py`) that scores the live API, so what is measured
  is the system as shipped, policy included.

## What I'd add with more time / in production

- **Close the domain gap with real labels.** Run `annotate.py` over several thousand real
  proposals, retrain on the mix, and iterate. This is the single highest-value remaining
  change and everything below is worth less than it.
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
- **Real observability**: OpenTelemetry traces spanning gateway → sidecar → model call, and
  RED metrics per engine. Escalation spend in particular deserves a metric, since it is the
  one cost that scales with traffic.
- **Caching by content hash.** Appraisal pages are re-submitted often; a hash-keyed result
  cache removes both latency and model spend for repeats.
- **Batch endpoint and back-pressure.** A whole loan file is dozens of pages; a batch API with
  a bounded queue would beat dozens of independent requests.
