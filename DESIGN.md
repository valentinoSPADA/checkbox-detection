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

## 6. Claude Haiku 4.5 as the default model

**Decision**: Both Claude-backed paths -- the `vlm` engine and the offline annotator -- default
to `claude-haiku-4-5` rather than a frontier model, overridable via `ANTHROPIC_MODEL`.

**Why**: Measured against the two jobs actually being asked of the model. Annotation sends
96x96 crops, roughly 12 image tokens each, so labelling 1600 real proposals costs about $0.17
on Haiku against $0.86 on Opus 5 -- both negligible. The `vlm` engine is where spend actually
lives: eight ~1400 px tiles per page is about $0.12 a page on Haiku against $0.58 on Opus 5.
Since that engine is a demonstration of AI integration rather than the primary path, paying
five times more for it buys nothing this submission needs.

**Tradeoff**: Haiku is materially weaker at the harder of the two jobs. Localising ~100
checkboxes of ~22 px each is a precision task where a frontier model's advantage is real, so
the `vlm` engine's numbers will understate what that architecture can do. Annotation -- a
three-way classification of a small crop -- is well within Haiku's range, which is the job
that matters, because annotation quality sets the ceiling for the local model.

**Would reconsider if**: The annotation spot-check showed systematically bad labels. That is
the one failure that must not be accepted on cost grounds: a mislabelled training set produces
a model that is confidently wrong with no metric revealing it, so `annotate.py` writes a
contact sheet and a small batch is reviewed by eye before the full run. If Haiku's labels were
poor there, annotation would move to a stronger model and the `vlm` engine would stay on Haiku
-- the two choices are independent and only one of them is cost-sensitive.

## 7. The confidence floor is the whole ballgame (measured)

**Decision**: `MinConfidence` defaults to 0.95, and confidence floors are configured **per
producing engine** rather than shared.

**Why**: This started as a bug hunt, not a design choice. The system was returning 1528 boxes
on a page holding roughly 120, and the obvious diagnosis -- "the classifier is bad" -- was
wrong. Breaking the detections down by size showed two clean populations:

| size cluster | count | mean confidence |
|---|---|---|
| 10 px | 940 | 0.725 |
| 12 px | 302 | 0.803 |
| **52 px** | **57** | **0.971** |
| **54 px** | **42** | **0.972** |

The true checkboxes are the 52-54 px cluster, and the model is *already* separating them --
by confidence, cleanly. The default floor of 0.60 simply sat underneath the noise. Sweeping
it:

| floor | 0.60 | 0.90 | 0.95 | 0.97 | 0.99 |
|---|---|---|---|---|---|
| boxes on sample 1 | 1528 | 244 | **125** | 65 | 0 |

125 against a true count near 120, with no change to the model. The cliff at 0.99 is
explained rather than mysterious: label smoothing of 0.05 during training caps the softmax
near 0.98, so anything above ~0.97 is off the usable scale.

The per-source part came out of the same investigation. With one shared floor of 0.95, every
candidate escalated to Claude was discarded on return, because the model reports its own
certainty around 0.90 even when sure. The assisted engine was paying for model calls that
*could not change the answer by construction*, and it looked like it was working. Confidences
from a synthetic-trained softmax and from a language model's self-assessment are not the same
quantity and must not share a threshold.

**Tradeoff**: A floor this high is tuned against four pages of one document family, and a
scanned page with weaker contrast would produce lower confidences across the board and lose
recall silently. It is also a *calibration* fix rather than a *capability* fix -- the
classifier's real weakness on the domain gap is still there, hidden below the floor rather
than removed.

**Would reconsider if**: The classifier were retrained on real labelled data. Better
calibration would move the useful floor back toward the middle of the range, where it is far
less brittle. A per-page adaptive threshold -- picking the floor from the confidence histogram
of each page rather than fixing it globally -- is the more robust version of this and is the
right answer if pages vary more than these four do.

## 8. What Claude Haiku 4.5 could and could not do (measured)

Both Claude paths were run end to end against the real API, and the two results differ sharply
in a way worth recording rather than smoothing over.

**Adjudication (small crops): works.** This is the assisted engine and the ground-truth
builder. Judging a 96 px crop as checked / unchecked / not-a-checkbox is well inside Haiku's
range, and it is the job that matters most, because it is the one that produces training
labels and corrections.

**Localisation (whole tiles): does not.** The `vlm` engine returns roughly the right *number*
of boxes in roughly the right *rows*, but the coordinates are systematically offset -- many
land in empty space (`docs/overlays/_vlm_sample2.png`). This was predicted in decision 6 and
is confirmed: asking a compact model for precise pixel coordinates of a hundred 22 px objects
is asking for the thing it is worst at.

The honest conclusion is that the `vlm` engine's numbers measure Haiku, not the architecture.
Pointing `ANTHROPIC_MODEL` at a frontier model is a one-variable change and is the right move
if that path ever needs to be good rather than merely demonstrated.
