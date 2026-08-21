// Package localengine adapts the Python detection sidecar to the domain.Detector port.
//
// This is an outbound adapter: it owns the wire format, the multipart encoding and the
// failure taxonomy of one specific downstream service, and exports none of that. The domain
// never learns that Stage 1 and Stage 2 happen over HTTP in another language.
package localengine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
	"time"

	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

// ErrUnavailable reports that the sidecar could not be reached or is not ready.
//
// Distinguished from a malformed-input error because the two need opposite handling: this
// one is retriable and maps to 503, while a decode failure is the caller's fault and maps to
// 400. Collapsing them into one error type is what turns a transient outage into a stream of
// misleading 400s in the caller's dashboards.
var ErrUnavailable = errors.New("detection engine unavailable")

// ErrBadImage reports that the sidecar could not decode the uploaded bytes.
var ErrBadImage = errors.New("image could not be decoded")

// Client calls the detection sidecar over HTTP.
type Client struct {
	baseURL string
	http    *http.Client
	floor   float64
}

// Option customises a Client.
type Option func(*Client)

// WithHTTPClient injects a preconfigured *http.Client, which is how tests point the adapter
// at an httptest server and how production supplies connection-pool tuning.
func WithHTTPClient(c *http.Client) Option {
	return func(cl *Client) { cl.http = c }
}

// WithFloor sets the candidate probability floor requested from the sidecar.
//
// This is a transport concession rather than the detection threshold: it exists to stop a
// dense page from shipping thousands of confidently-negative candidates over the wire. It is
// kept well below domain.Policy.MinConfidence so that policy, not transport, decides what is
// returned -- and in particular so escalation can still see candidates the floor would drop.
func WithFloor(f float64) Option {
	return func(cl *Client) { cl.floor = f }
}

// New builds a Client for the sidecar at baseURL.
func New(baseURL string, timeout time.Duration, opts ...Option) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: timeout},
		floor:   0.30,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Name identifies this adapter in responses and metrics.
func (c *Client) Name() domain.EngineName { return domain.EngineLocal }

// wireResponse mirrors the sidecar's JSON contract. Kept unexported and separate from the
// domain types so a change on the Python side is absorbed here rather than rippling inward.
type wireResponse struct {
	Candidates []struct {
		BBox       []int   `json:"bbox"`
		IsChecked  bool    `json:"is_checked"`
		Confidence float64 `json:"confidence"`
	} `json:"candidates"`
	Width           int `json:"width"`
	Height          int `json:"height"`
	RawProposals    int `json:"raw_proposals"`
	ScoredProposals int `json:"scored_proposals"`
}

// Detect runs the two-stage pipeline on one page.
//
// Returns candidates *unfiltered*: suppression and thresholding belong to domain.Policy, and
// applying them here would hide from the policy layer exactly the low-confidence candidates
// the assisted engine needs in order to decide what to escalate.
//
// Failure modes: ErrUnavailable when the sidecar is unreachable, times out, reports 503, or
// returns a 5xx; ErrBadImage on 400 or 415; a wrapped error on any other status. Context
// cancellation propagates, so an abandoned inbound request tears the outbound call down with
// it instead of leaving it to finish into a dead writer.
func (c *Client) Detect(ctx context.Context, img domain.Image) (domain.Result, error) {
	body, contentType, err := encodeMultipart(img)
	if err != nil {
		return domain.Result{}, fmt.Errorf("encoding upload: %w", err)
	}

	url := fmt.Sprintf("%s/v1/detect?floor=%.3f", c.baseURL, c.floor)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return domain.Result{}, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return domain.Result{}, ctx.Err()
		}
		return domain.Result{}, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer func() {
		// Drain before closing so the connection returns to the pool instead of being torn
		// down; at a few thousand pages an hour, leaking connections here would show up as
		// ephemeral-port exhaustion long before it showed up as latency.
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
		_ = resp.Body.Close()
	}()

	switch {
	case resp.StatusCode == http.StatusOK:
	case resp.StatusCode == http.StatusBadRequest, resp.StatusCode == http.StatusUnsupportedMediaType:
		return domain.Result{}, fmt.Errorf("%w: sidecar rejected the upload", ErrBadImage)
	case resp.StatusCode >= 500:
		return domain.Result{}, fmt.Errorf("%w: sidecar returned %d", ErrUnavailable, resp.StatusCode)
	default:
		return domain.Result{}, fmt.Errorf("sidecar returned unexpected status %d", resp.StatusCode)
	}

	var wire wireResponse
	if err := json.NewDecoder(resp.Body).Decode(&wire); err != nil {
		return domain.Result{}, fmt.Errorf("decoding sidecar response: %w", err)
	}

	dets := make([]domain.Detection, 0, len(wire.Candidates))
	for _, c := range wire.Candidates {
		if len(c.BBox) != 4 {
			// A malformed entry is skipped rather than failing the page: one bad box should
			// not discard the other hundred valid detections on the same document.
			continue
		}
		dets = append(dets, domain.Detection{
			Box:        domain.NewBox(c.BBox[0], c.BBox[1], c.BBox[2], c.BBox[3]),
			IsChecked:  c.IsChecked,
			Confidence: c.Confidence,
			Source:     domain.EngineLocal,
		})
	}

	return domain.Result{
		Detections: dets,
		Width:      wire.Width,
		Height:     wire.Height,
		Engine:     domain.EngineLocal,
		Stats: domain.Stats{
			RawProposals:    wire.RawProposals,
			ScoredProposals: wire.ScoredProposals,
			Candidates:      len(dets),
		},
	}, nil
}

// Health reports whether the sidecar is reachable and has its model loaded.
//
// Used by the API's readiness endpoint. Model readiness is reported separately from
// reachability because the two demand different operator responses: unreachable means check
// the network or the container, model-not-loaded means the artifact was never built.
func (c *Client) Health(ctx context.Context) (ready bool, detail string, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return false, "", err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return false, "", fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer func() { _ = resp.Body.Close() }()

	var payload struct {
		Status      string `json:"status"`
		ModelLoaded bool   `json:"model_loaded"`
		Detail      string `json:"detail"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return false, "", fmt.Errorf("decoding health response: %w", err)
	}
	return payload.ModelLoaded, payload.Detail, nil
}

// encodeMultipart builds the multipart body the sidecar expects.
//
// The filename and content type are forwarded from the original upload because OpenCV's
// decoder is content-sniffing but the sidecar logs the filename, and losing it makes a
// failing document impossible to identify in the logs. A missing content type falls back to
// application/octet-stream rather than being omitted, since some HTTP stacks reject a part
// with no Content-Type header at all.
func encodeMultipart(img domain.Image) (io.Reader, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	name := img.Filename
	if name == "" {
		name = "upload"
	}
	ct := img.ContentType
	if ct == "" {
		ct = "application/octet-stream"
	}

	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename=%q`, name))
	h.Set("Content-Type", ct)
	part, err := w.CreatePart(h)
	if err != nil {
		return nil, "", err
	}
	if _, err := part.Write(img.Data); err != nil {
		return nil, "", err
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return &buf, w.FormDataContentType(), nil
}
