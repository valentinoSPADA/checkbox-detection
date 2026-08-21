# DESIGN — Checkbox Detection

> Written incrementally **while** the decisions were made, not reconstructed at the end.
> Every entry states what was chosen, why it ties to a specific line of the challenge or the
> job description, what the choice costs, and the condition that would reverse it.

## 0. What the problem actually is

The brief reads "detect and classify checkboxes". Looking at the four supplied appraisal
documents, the hard part is **not** finding squares — it is *rejecting* them. These forms are
dense grids: a single URAR page contains thousands of rectangular table cells, and the
checkboxes are a small fraction of them. Concretely, measured on the samples:

| Sample | Document | Pixels | The specific difficulty it contributes |
|---|---|---|---|
| 1 | URAR Fannie Mae 1004 | 2550x4200 | Digital-native, hairline rules, boxes ~22 px, `X` marks |
| 2 | Zoomed crop of a 1004 | 1586x846 | **Different scale**, heavy stroke weight, freehand pen scribbles crossing the form, a check mark instead of `X`, one hand-drawn box |
| 3 | 1004MC Market Conditions | 2550x4200 | **Blue-shaded table rows** — checkboxes sit on colour, not on white |
| 4 | Manufactured Home 1004C | 2550x3301 | **Diagonal red watermark**, blue ink signature, weaker scanned rules |

So the system must be scale-adaptive, background-adaptive, and above all **precise**, since
recall is comparatively easy to buy.

### Empirical finding that drove the architecture

Three purely geometric detector generations were prototyped and measured against the samples
before any code was committed (see `docs/prototype-log.md`):

| Generation | Approach | Outcome |
|---|---|---|
| v1 | Morphological line masks, then enclosed-cell segmentation | Failed: the opening kernel cannot separate a 22 px checkbox rule from a 12 px bold-text stroke. Detections landed on letter counters (`o`, `a`, `8`). |
| v2/v3 | Connected components plus border-band / straight-run verification | Precision improved, but recall collapsed on sample 3 (5 detections): on dense forms a checkbox frequently **touches** the adjacent table rule, so its connected component becomes the entire table grid. |
| v4 | Multi-scale rectangle assembly from straight ink runs | Recall solved (immune to touching rules), but precision poor: **2198** surviving candidates on sample 1 against a true count near 120. |

Each generation traded recall against precision without winning both. That ceiling is a
property of the problem, not of tuning: geometry alone cannot distinguish "checkbox" from
"checkbox-sized table cell", and an ink-density threshold classifies a freehand check mark
poorly.

**This is the finding that justifies the machine-learning stage.** The ML component is not
decoration — it is the answer to a measured failure of the geometric approach.

---

## 1. Two-stage detection (proposal plus learned classification)

**Decision**: Stage 1 generates high-recall geometric *proposals*; Stage 2 is a small learned
CNN that classifies each proposal crop into `not_a_checkbox` / `unchecked` / `checked`.

**Why**: It converts the measured v4 failure (recall good, precision bad) into the exact shape
of problem a classifier is good at, and it collapses the challenge's two requirements —
"detects the location" and "classifies each as filled or unfilled" — into one learned decision
instead of two hand-tuned thresholds. It is also the standard production document-AI topology,
which matters because the challenge states the review meeting will probe *why* this approach
over alternatives.

**Tradeoff**: Gives up the single-model elegance of an end-to-end detector, and caps recall at
whatever Stage 1 proposes — a checkbox the geometry never nominates can never be recovered by
Stage 2. Adds a training pipeline and a model artifact to the repository.

**Would reconsider if**: A labelled dataset of real appraisal pages existed. With real labels,
fine-tuning an end-to-end detector (YOLO-class) would likely beat a two-stage pipeline and
remove the Stage 1 recall ceiling.

## 2. Synthetic training data

**Decision**: Train Stage 2 on procedurally generated crops — randomised line weight, box size,
background shade, adjacent text, rotation, blur, JPEG artefacts, and mark styles (`X`, check,
scribble, dot, partial) — plus hard negatives mined from Stage 1 proposals on the four real
samples.

**Why**: The challenge supplies four images and no annotations. Synthesis is the only route to
a training set of meaningful size, and it lets the *known* failure modes catalogued in section 0
be generated deliberately rather than hoped for.

**Tradeoff**: Introduces a synthetic-to-real domain gap; the model can only be as good as the
generator's imagination, and a real-world artefact never simulated will be misread. Hard
negatives from the real samples narrow but do not close the gap.

**Would reconsider if**: Even a couple hundred annotated real pages were available — at that
point fine-tuning on real data dominates synthesis, and synthesis drops to an augmentation role.

## 3. Go API plus Python detection sidecar

**Decision**: A Go service owns the public HTTP API and all orchestration/policy; a Python
service owns pixels and the model. They communicate over HTTP inside the compose network.

**Why**: The JD names Go first ("Go (preferred)") and asks for concurrency, low latency and
"millions of documents", while separately asking to "Bridge the AI Gap: integrate AI services
into our core SaaS platform, ensuring seamless handoffs between our decisioning models and the
user application". That is a literal description of this split, and it is how the ML ecosystem
actually forces the boundary — OpenCV, PyTorch and ONNX Runtime live in Python.

**Tradeoff**: Two runtimes, two images, an extra network hop per request, and the pixel work —
arguably the most interesting code — lands outside Go. A single Python service would be
simpler; a pure-Go pipeline (no cgo) would be a stronger Go signal but would forfeit the
learned stage entirely.

**Would reconsider if**: The model were exportable to a format with mature pure-Go inference
and the CV preprocessing were simple enough to reimplement, at which point one Go binary would
beat two services on both deploy simplicity and latency.

## 4. Detector as a port with three real adapters

**Decision**: `Detector` is an interface in the Go domain. Three implementations satisfy it:
`local` (the sidecar), `vlm` (Anthropic Claude vision), and `assisted` (local proposals with
low-confidence candidates escalated to Claude).

**Why**: Two-plus real implementations are what make a hexagonal boundary genuine rather than
aspirational — a port with one adapter is just indirection. It also lets the submission
demonstrate generative-AI integration as a *runtime-selectable strategy* with a documented cost
and latency profile, instead of as an untested claim.

**Tradeoff**: More surface area to build, test and document inside a 2-3 day budget, and the
`vlm` path adds network dependency, per-request cost and latency that the local path does not
have.

**Would reconsider if**: The timebox tightened — the `assisted` adapter is the first thing to
cut, since `local` and `vlm` alone already prove the port.

## 5. Detection policy lives in Go, not in the sidecar

**Decision**: Cross-engine non-maximum suppression, confidence thresholding, the escalation
rule for `assisted`, and coordinate normalisation live in `backend/internal/domain`, with zero
HTTP or image imports.

**Why**: Without this, the Go service degrades into a proxy and there is no unit-testable Go
business logic — which would undercut both the "Backend Fluency" JD line and this repo's
testing pillar. Escalation policy (which candidates are worth a paid model call) is genuine
business logic with real cost consequences, so it belongs in the core.

**Tradeoff**: Some geometry is expressed twice in two languages (Python for proposals, Go for
policy). Shared box geometry is kept minimal and the split is documented to keep it honest.

**Would reconsider if**: The policy stayed trivial enough to be a single threshold, in which
case pushing it into the sidecar and letting Go be a thin gateway would be the simpler design.
