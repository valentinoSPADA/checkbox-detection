# Production image: the whole stack in one container.
#
# Compose runs three services, and that is the honest architecture -- a static bundle, an API,
# and a Python detection sidecar, each independently scalable. This image collapses them for
# deployment, and the reason is specific rather than a shortcut:
#
#   * On a free/small tier the three would be three separate machines with three cold starts,
#     and the API would happily answer requests while the sidecar it depends on was still
#     asleep. That failure looks like an intermittent 503 and is miserable to diagnose.
#   * The API talks to the sidecar over HTTP either way. In one container that call goes over
#     loopback, which removes a network hop and its failure modes without changing a line of
#     code -- the Detector port still speaks HTTP to something that could be anywhere.
#   * nginx disappears entirely: the Go binary serves the bundle from its own filesystem
#     (internal/webui), so there is no third process whose only job is twelve static files.
#
# What it costs: the two processes scale together and share a CPU quota, so a page under
# detection makes the API less responsive. At this scale that is the correct trade; at real
# volume the compose topology is what to deploy, unchanged.

# ── Frontend ────────────────────────────────────────────────────────────────────────────────
FROM node:22-alpine AS ui

WORKDIR /ui

# Lockfile first, for layer caching. `npm ci` rather than `npm install` so the bundle is built
# from the exact tree the tests ran against, not from whatever satisfies the ranges today.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./

# The bundle is served by the same origin that answers /detect, so the API base is relative.
# The literal is a sentinel rather than an empty string -- see SAME_ORIGIN in lib/api.ts for
# why "empty means same origin" is a rule this codebase deliberately does not have.
ENV VITE_API_BASE_URL=same-origin
RUN npm run build

# ── API ─────────────────────────────────────────────────────────────────────────────────────
FROM golang:1.25-alpine AS api

WORKDIR /src

COPY backend/go.mod backend/go.sum* ./
RUN go mod download

COPY backend/ ./

# The built bundle replaces the committed placeholder before compiling, so go:embed picks up
# the real UI. Doing it here rather than at runtime is what makes the binary self-contained:
# there is no deployment in which a stale bundle talks to a newer API.
COPY --from=ui /ui/dist ./internal/webui/dist

ARG VERSION=dev

# Vet and test during the image build, so a broken build cannot be pushed even if someone
# bypasses CI. This has already earned its place once: it caught a test suite that had started
# asserting against a new 415 instead of against the handler, and refused to produce an image.
RUN go vet ./... && go test ./...

# CGO off for a static binary; trimpath and stripped symbols because neither a build path nor
# a symbol table belongs in a shipped artifact.
RUN CGO_ENABLED=0 go build \
    -trimpath -ldflags "-s -w -X main.version=${VERSION}" \
    -o /out/api ./cmd/api

# ── Runtime ─────────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgl and libglib are OpenCV's runtime shared libraries. opencv-python-headless still links
# against them, and their absence surfaces as an ImportError at first request rather than at
# build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY detector/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only what the request path needs. `training/` is deliberately absent: it pulls in torch,
# which is hundreds of megabytes and is not used to serve a single request.
COPY detector/app ./app
COPY detector/engine ./engine
COPY detector/models ./models

COPY --from=api /out/api /usr/local/bin/api
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Runs as a non-root user with no home and no shell. Nothing in the request path writes to
# disk, so there is nothing for it to own.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin app
USER app

ENV PORT=8080 \
    DETECTOR_URL=http://127.0.0.1:8000 \
    ADDR=:8080 \
    # One thread each: the two processes share a small CPU quota, and letting onnxruntime fan
    # out across every core makes it contend with the very API that has to answer the request.
    ORT_THREADS=1 \
    OMP_NUM_THREADS=1 \
    # glibc returns freed pages reluctantly; without these a process that peaks at ~290 MB per
    # page settles near the high-water mark of every page it has ever seen.
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=131072 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# Probes /ready rather than /health: readiness reports whether the model actually loaded,
# and a container that is alive but cannot detect anything should not be taking traffic.
HEALTHCHECK --interval=20s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=4).status==200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
