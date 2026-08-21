# Public API gateway.
#
# Multi-stage: the toolchain image is ~800 MB and the resulting binary is a few megabytes, so
# the split is worth it here in a way it is not for the Python sidecar.

FROM golang:1.25-alpine AS build

WORKDIR /src

# Module files first, so `go mod download` is cached independently of source edits.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

ARG VERSION=dev
# CGO is disabled to produce a genuinely static binary, which is what allows the final stage
# to be `scratch` rather than a distro image. -trimpath keeps build-host paths out of the
# binary, so the artifact is reproducible across machines.
RUN CGO_ENABLED=0 GOOS=linux go build \
    -trimpath \
    -ldflags="-s -w -X main.version=${VERSION}" \
    -o /out/api ./cmd/api

# Test and vet during the image build, so a broken build cannot be pushed even if someone
# bypasses CI. Cheap here: the packages are already compiled by the step above.
RUN go vet ./... && go test ./...

FROM scratch AS runtime

# CA roots are required for the outbound HTTPS call to the Anthropic API. Without them the
# vlm engine fails with an opaque x509 error that looks like a code bug rather than a
# missing file, which is a genuinely expensive hour to lose.
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=build /out/api /api

# Numeric rather than named, because scratch has no /etc/passwd to resolve a name against.
USER 10001:10001

EXPOSE 8080

ENTRYPOINT ["/api"]
