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
│ backend  (Go)                                               │
│                                                             │
│  httpapi ──────────► domain.Detector (PORT) ◄────────────┐  │
│  inbound adapter          │                              │  │
│  parse, validate,         ├── localengine  (adapter) ────┼──┼──► detector (Python)
│  delegate, shape          ├── vlm          (adapter) ────┼──┼──► Anthropic API
│                           └── assisted     (adapter) ────┘  │
│                                    │                        │
│                            domain.Policy                    │
│              suppression · threshold · escalation · merge   │
└─────────────────────────────────────────────────────────────┘
```

The `assisted` adapter is itself a composite: it holds `localengine` and calls the vision
model only for the candidates `domain.Policy.SelectForEscalation` hands it. That it can be
built out of the other two without either knowing is the practical test that the port is a
real seam.

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
    /localengine               HTTP client for the Python sidecar.
    /vlm                       Anthropic Claude vision, with page tiling and concurrency.
      /prompts                 Prompt text as data files, embedded via go:embed. Kept here
                               rather than as Go constants because the offline annotation
                               tool reads these same bytes — one source of truth for what
                               "checkbox" means to a model.
    /assisted                  Composite: local pipeline + escalation of the uncertain band.
  /internal/httpapi          → Inbound adapter. Routes, middleware, DTOs. Contains no
                               detection logic; if a handler ever branches on a confidence
                               value, that logic has leaked out of the domain.
  /internal/imaging          → Adapter-side pixel utilities (decode, resize, tile, crop).
                               Deliberately NOT in the domain: only the vlm adapter needs it.
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
  /training                  → Offline only; never installed into the runtime image.
    synth.py                   procedural training-data generator.
    model.py / train.py        architecture, training loop, ONNX export.
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
