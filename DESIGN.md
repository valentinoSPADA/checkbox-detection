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

**Update after measurement**: the gap was real and was closed the way this entry anticipated —
`training/annotate.py` labelled 1832 real crops with Claude and the mixed model gained +3.9 F1
over synthetic-only. The tradeoff below stands, narrowed rather than removed: 1832 crops from
four pages of one form family is not a dataset, it is a patch on one distribution.

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

> **SUPERSEDED by §12.** Two of the three adapters were removed. The entry is kept as written
> because the "would reconsider if" below names the condition that actually fired, and a
> decision log that quietly edits itself after the fact is worth nothing.

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

## 8. What the vision models could and could not do (measured)

Both Claude paths were run end to end against the real API, and the result corrected an
assumption made in decision 6 rather than confirming it.

**Adjudication (small crops): works.** This is the assisted engine and the ground-truth
builder. Judging a 96 px crop as checked / unchecked / not-a-checkbox is well inside Haiku's
range, and it is the job that matters most, because it is the one that produces training
labels and corrections.

**Localisation (whole page or tile): does not — and not because the model is small.**
Decision 6 predicted that Haiku would localise poorly and implied a frontier model would fix
it. The first half was right; the second was wrong, and the probe that tested it is worth
recording:

A single un-tiled page was sent directly to the Messages API with the same prompt and tool
schema the adapter uses, so no code of this repository was involved in the result.

| model | boxes returned (≈48 real) | box widths returned | placement |
|---|---|---|---|
| Claude Haiku 4.5 | 45 | every one exactly 24 px | systematically offset |
| Claude Opus 5 | 43 | 19-20 px | systematically offset |

Both find approximately the right *number* of checkboxes and the right *rows*. Neither places
them accurately, and the constant box width is the tell: the model is not measuring an object,
it is emitting a plausible size at an approximate position. Overlays for both are in
`docs/overlays/_probe_*.png`.

**What this actually means for the architecture.** Precise pixel localisation of a hundred
20 px objects is not a task current vision models do well, at any tier. That is not a reason
to drop them — it is a reason to use them for the half of the problem they *are* good at.
Which is exactly the split this system already has: classical geometry localises, because
geometry is exact and free; the model judges what geometry found, because judgement is what
it is better at than an ink-density threshold. The `assisted` engine is therefore not a
compromise between the two engines, it is the only one of the three that asks each component
for the thing it does well.

**Would reconsider if**: A model exposed a segmentation or grounding head rather than
free-form coordinates, or the task moved to a handful of large boxes per page instead of a
hundred small ones -- at which point coordinate estimation stops being the bottleneck.

## 9. Detection is an explicit action, never a side effect

**Decision**: The UI runs detection only when the user presses **Detect**. Choosing a file
loads it for display; changing the engine marks the on-screen result stale and says so.
Neither sends a request.

**Why**: Two of the three engines cost money per run. The first version detected on file
selection *and* on every engine change, so opening one page and comparing two engines spent
three calls, two of which nobody asked for. Selecting what to run and deciding to run it are
different decisions, and an interface that fuses them spends the user's budget on their
behalf. The stale-result banner exists for the same reason: silently relabelling a `local`
result as `assisted` would corrupt the exact comparison the UI is built to support.

**Tradeoff**: One more click on the free engine, where auto-running was genuinely convenient.
That is a real cost and it is worth paying, because the failure it prevents is silent and the
inconvenience it adds is visible.

**Would reconsider if**: The engine were free and local-only, at which point the cost argument
disappears and immediate feedback wins.

## 10. The black section rail: a recognition problem, not a geometry one

**Decision**: The false positives along each page's black section rail -- the vertical
SUBJECT / CONTRACT / NEIGHBORHOOD band -- are fixed by adding that region to the synthetic
generator as a negative class, not by a geometric filter.

**Why**: Inside a solid dark block every pixel is ink, so Stage 1 finds long runs everywhere
and nominates rectangles by the hundred; the classifier accepted them because no training crop
had ever been mostly dark. The obvious filter is to reject candidates surrounded by too much
ink, and it was measured before being rejected:

| surrounding-ink threshold | sample 1 (clean) | sample 4 (watermarked) |
|---|---|---|
| < 0.22 | removes 51 of 53 rail false positives, keeps **100%** of body detections | keeps only **54%** of body detections |

The threshold that is free on one page destroys another, because absolute ink density is a
property of the page rather than of the object. A page-dependent constant is exactly the kind
of tuning that survives four samples and fails on the fifth. Teaching the model what the
inside of a black bar looks like has no such coupling.

**Measured effect**: rail false positives on sample 1 fell from 53 to 35, and end-to-end
precision across all four samples rose from 0.622 to 0.868 (F1 0.684 to 0.819; later 0.930 /
0.801 once the classifier was retrained on real labels). The rail is
reduced rather than eliminated, which is the honest result: a generator can only approximate
the real thing.

**Tradeoff**: Requires retraining rather than a one-line filter, and the fix only holds for
dark regions resembling the generated ones -- an inverted-video scan would still confuse it.
It also made the classifier more conservative overall, which cost recall on the watermarked
page (0.656 to 0.595) while nearly doubling its precision (0.475 to 0.812).

**Would reconsider if**: A page-level segmentation step existed (identifying rails, headers and
body regions before detection), which would make this a routing decision rather than a
classification one and would generalise better than either approach.

## 11. The escalation budget must scale with uncertainty, not be a constant

> **SUPERSEDED by §12.** Escalation is gone with the engine that used it. The measurement below
> stands and is the reason the reasoning is kept: a flat budget allocated help in inverse
> proportion to how much a page needed it, which is a mistake that generalises well beyond this.


**Decision**: `MaxEscalations` raised from 40 to 120, and adjudication chunked into batches of
20 crops per model call.

**Why**: A reviewer noticed that `assisted` returned nearly the same boxes as `local` and asked
whether that was intended. It was not, and the cause was arithmetic. The number of candidates
inside the uncertainty band, measured per sample:

| sample | in band | escalated at cap 40 | coverage |
|---|---|---|---|
| 1 — clean URAR | 57 | 40 | 70% |
| 2 — zoomed crop | 136 | 40 | 29% |
| 3 — shaded rows | 146 | 40 | 27% |
| 4 — watermarked | **448** | 40 | **9%** |

A flat cap allocates help in inverse proportion to how much a page needs it. Sample 4 has the
worst recall of the four and received the least adjudication of the four; with 9% coverage,
`assisted` matching `local` is not a surprising result, it is the expected one.

Chunking is what makes a higher cap usable at all: adjudication previously sent every
candidate in one message, and a single request asking for 448 verdicts overruns `max_tokens`
and returns a truncated, unparseable tool call — losing the whole page's adjudication rather
than one batch of it.

**Measured effect**: assisted went from +0.7 F1 over local to **+1.9** (0.819 → 0.838), and
the gain is in recall (0.776 → 0.814) rather than precision, which is the direction that
matters for the pages that were failing. Recall on sample 4 rose from 0.595 to 0.634.

**Tradeoff**: Three times the model spend per hard page — roughly $0.01 against $0.004 on
Haiku, still trivial per page but not per million. Precision also dips slightly (0.872 to
0.863): more second opinions means more chances to be talked into a wrong one.

**Would reconsider if**: Spend mattered more than accuracy, in which case the cap belongs in
per-tenant configuration rather than in a default. The genuinely better design is a budget
proportional to the band's population with a hard ceiling, but "how much may this page cost"
is a policy decision belonging to whoever pays the bill, so it is exposed as
`MAX_ESCALATIONS` rather than decided here.

---

## 12. Removing the vision engines

**Decision**: `vlm` and `assisted` are deleted. One engine ships. The Go service now has zero
external dependencies and the system needs no credential of any kind.

**Why**: They made the product worse, and the measurement in §8 says why. A language model
asked to locate checkboxes on a full page returns **plausible sizes at approximate positions**
rather than measured ones — Haiku returned 45 boxes every one of which was exactly 24 px wide,
and Opus 5 returned 43 at 19-20 px, both systematically offset. The constant width is the tell:
it is emitting a reasonable answer, not measuring one. Geometry produces exact coordinates for
free, so the paid path was buying a worse answer.

`assisted` was the more defensible of the two, and it went for a different reason: with the
classifier retrained on real labels (§13 of `docs/prototype-log.md`) the uncertainty band it
fed on is nearly empty. Its premise — that the local model is often unsure — stopped being
true, and a mechanism whose premise has expired is cost with no upside.

**Tradeoff**: The port now has one adapter, and a port with one adapter is, on its face, just
indirection. The submission also loses a live demonstration of third-party AI integration,
which the JD explicitly asks about.

Both are accepted, for reasons that are not consolation:

* The removal is itself the stronger evidence about the seam. Deleting two of three engines
  touched `cmd/api` and the engines map — no handler, no policy, no domain type changed
  shape, and `TestDomainHasNoOutboundDependencies` never moved. An architecture is proved by
  a change, and this was a bigger one than adding a third adapter would have been.
* The AI integration did not go away, it moved off the request path. Claude labels training
  data offline in `training/annotate.py`, and the argument for putting it there rather than in
  the hot path is the measurement above. "We used a model where it was better than the
  alternative, and stopped where it was worse" is a stronger answer in a review than a paid
  call that runs on every page because it was impressive to have.

**What was kept**: `Policy.SourceMinConfidence` and the `Source` field on every `Detection`,
both of which are dead weight with one engine. They stay because the reason they exist is
structural rather than incidental — confidences from different producers are not on the same
scale — and because that was learned expensively: under one shared floor, every verdict from
the second engine was silently discarded, so it paid for calls that could not change the answer
by construction.

**Would reconsider if**: A document family arrives where the geometry cannot propose the boxes
at all — handwritten forms, photographs at an angle, anything where "four straight inked sides"
stops describing a checkbox. That is a Stage 1 failure, and no classifier retraining fixes it;
a vision model that reads the page as a page would then be the right tool rather than a more
expensive way to do worse.

## 13. Upload limits at two layers

**Decision**: Byte size and content type are enforced in the Go gateway; pixel count and
minimum side are enforced in the Python sidecar.

**Why**: Neither layer can do the other's job.

The gateway can bound the *body* cheaply, before it is buffered, using `MaxBytesReader` rather
than the caller-supplied `Content-Length` — which is a claim, and which a chunked upload omits
entirely. It can also sniff the format from the leading bytes, because the multipart header's
Content-Type is likewise a claim and anything at all can arrive labelled `image/png`.

What the gateway cannot bound is what the image *costs*, because compression ratio is chosen
by whoever sends the file: a 25 MB PNG of flat colour decodes to hundreds of megapixels, and
Stage 1 then allocates several full-resolution int32 arrays over it. That limit has to sit
where the decode happens, and the decode happens once, in `engine.preprocess.decode`.

**Tradeoff**: The rule "check input at the edge" is broken deliberately — one class of check
lives two hops in. A reader looking only at the gateway will not see the whole story, which is
why both `.env.example` and the README state the split as a table rather than a list.

**Would reconsider if**: Go started decoding the image for its own reasons. It does not today
and that is not an accident: the gateway owns policy, the sidecar owns pixels, and moving the
pixel check outward would mean decoding every page twice to enforce it.
