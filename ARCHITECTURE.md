# ARCHITECTURE

The point of this document is that the architecture can be **verified by reading the tree**,
not just by trusting the prose in `DESIGN.md`. Every top-level folder below is mapped to the
concept it implements, and the two claims most worth doubting — that the domain is really
isolated, and that the port really has more than one implementation — are each backed by
something executable rather than by assertion.

## Request path

```
browser / curl
      │  multipart POST /detect?engine=…
      ▼
┌─────────────────────────────────────────────────────────────┐
│ backend  (Go)                     no external dependencies  │
│                                                             │
│  httpapi ──────────► domain.Detector (PORT) ◄────────────┐  │
│  inbound adapter          │                              │  │
│  parse, validate,         └── localengine  (adapter) ────┼──┼──► detector (Python)
│  sniff, delegate                   │                        │
│                            domain.Policy                    │
│                    threshold · suppression · cap            │
└─────────────────────────────────────────────────────────────┘
```

**One adapter ships, and the port is still load-bearing.** Two others existed here — `vlm`,
which sent the page to Claude, and `assisted`, a composite holding `localengine` and
escalating only its uncertain candidates — and both were removed after measurement: a language
model returns plausible box sizes at approximate positions rather than measured ones, which is
a worse answer than the geometry produces for free, and it costs money per page. `DESIGN.md`
keeps the numbers as a rejected alternative.

What that removal demonstrates is the point of the seam, from the other direction: deleting
two of three engines touched `cmd/api` and the engines map. No handler, no policy, no domain
type changed shape. An architecture is only proved by a change, and this was one.

It also left the Go service with **zero external dependencies** — `go.mod` names none — which
is what a supply chain looks like when the only thing crossing the network is one HTTP call to
a service in the same compose file.

## Folder → concept

```
/backend                     The public API. Owns orchestration and policy; owns no pixels.
  /cmd/api                   → Composition root. The ONLY place that names concrete adapters.
                               Everything below depends on the port, which is what lets an
                               engine be chosen by query parameter at runtime.
  /internal/domain           → Core. Boxes, detections, and detection policy. Zero imports of
                               net/http, image, or any adapter — enforced by
                               TestDomainHasNoOutboundDependencies, which parses this
                               package's own imports and fails on a violation. Without that
                               test the isolation claim would be a convention, and the first
                               `import "net/http"` would pass review unnoticed.
  /internal/detector         → Outbound adapters. One sub-package per Detector implementation:
    /localengine               HTTP client for the Python sidecar. Currently the only one.
  /internal/httpapi          → Inbound adapter. Routes, middleware, DTOs, and the upload
                               guards: size, and content sniffed from the bytes rather than
                               trusted from the caller's header. Contains no detection logic;
                               if a handler ever branches on a confidence value, that logic
                               has leaked out of the domain.
  /internal/config           → Environment loading and eager validation.
  /internal/observability    → Structured logging setup.

/detector                    The detection engine. Owns pixels and the model; owns no policy.
  /engine                    → The two-stage pipeline:
    preprocess.py              decode + adaptive binarisation, shared by both stages so the
                               crops the model scores are the ones the geometry nominated.
    proposals.py               STAGE 1 — geometric region proposals, tuned for recall.
    classifier.py              STAGE 2 — ONNX inference. Runtime depends on onnxruntime, not
                               torch, which is what keeps the serving image small.
    pipeline.py                wires the two stages; returns candidates UNSUPPRESSED, because
                               suppression is policy and policy lives in Go.
  /training                  → Offline only; never installed into the runtime image, and not
                               on any request path. This is where a paid model is still used:
                               once, to label training data, with the result committed.
    synth.py                   procedural training-data generator.
    model.py / train.py        architecture, training loop, ONNX export.
    annotate.py                Claude labels real Stage 1 proposals; every verdict is checked
                               against a pixel measurement before it is kept.
    make_labeling_task.py      builds a self-contained HTML page for labelling by hand.
    import_labels.py           merges hand labels over model labels; hand labels win.
    /prompts                   Prompt text as data files rather than string constants, so the
                               annotator and the evaluation harness read the same bytes.
  /models                    → The committed model artifact. Versioned deliberately: a
                               reviewer must be able to clone and run without training.
  /tests                     → Stage 1 and preprocessing tests, on synthetic pages whose
                               checkbox positions are known exactly.
  /app                       → FastAPI surface. Internal service, not published by compose.

/frontend                    Overlay viewer. Secondary by design (see DESIGN.md §3).
  /src/lib/api.ts            → The one module that knows the wire format.
  /src/components            → Presentation only.

/eval                        → Measurement harness. Runs against the LIVE API rather than
                               importing internals, so what is scored is the system a
                               reviewer would actually call, policy included.
/samples                     → The four images from the challenge PDF, extracted.
/docs                        → Prototype log and generated diagnostic sheets.
```

## The two claims, and how to check them

**"The domain is isolated."**
`go test ./internal/domain/...` includes a test that parses every non-test file in the
package and fails if it imports `net/http`, `image`, `database/sql`, an adapter package, or
the Anthropic SDK. The rest of the domain suite runs with no sidecar, no model file and no
network, which is the practical consequence being claimed.

**"The port has real implementations, not one plus indirection."**
Three types satisfy `domain.Detector`, and `GET /engines` reports which of them this instance
actually registered — the vision engines are absent without an API key, and asking for one
then returns 400 rather than silently running a different engine.

## Where the language boundary falls, and its cost

Python owns everything that touches pixels; Go owns everything that decides. The honest cost
of that split is that box geometry exists in both languages: `engine/proposals.py` computes
candidate rectangles and `internal/domain/box.go` computes IoU over them. This is duplication
of a *concept*, not of code — neither implementation could call the other without a network
hop per box — but it is exactly the kind of thing that spreads, so SonarCloud's duplication
check runs in CI as the automated backstop (see `sonar-project.properties`).
