package localengine

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

const okBody = `{
  "candidates": [
    {"bbox": [10, 20, 32, 42], "is_checked": true,  "confidence": 0.97},
    {"bbox": [60, 20, 82, 42], "is_checked": false, "confidence": 0.91}
  ],
  "width": 800, "height": 600, "raw_proposals": 1234, "scored_proposals": 567
}`

func newTestClient(t *testing.T, handler http.HandlerFunc) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return New(srv.URL, 5*time.Second), srv
}

func testImage() domain.Image {
	return domain.Image{Data: []byte("pretend-png"), Filename: "page.png", ContentType: "image/png"}
}

func TestDetectParsesCandidates(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, okBody)
	})

	res, err := client.Detect(context.Background(), testImage())
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(res.Detections) != 2 {
		t.Fatalf("got %d detections, want 2", len(res.Detections))
	}
	if res.Width != 800 || res.Height != 600 {
		t.Errorf("page size = %dx%d, want 800x600", res.Width, res.Height)
	}
	if res.Engine != domain.EngineLocal {
		t.Errorf("engine = %q, want local", res.Engine)
	}
	// Every detection must be attributed, or the assisted engine's merged response becomes
	// impossible to debug.
	for _, d := range res.Detections {
		if d.Source != domain.EngineLocal {
			t.Errorf("detection missing source attribution: %+v", d)
		}
	}
	if res.Stats.RawProposals != 1234 || res.Stats.ScoredProposals != 567 {
		t.Errorf("pipeline counters lost: %+v", res.Stats)
	}
}

// TestDetectDoesNotFilter is the contract that makes the assisted engine possible: if this
// adapter applied the confidence floor itself, the policy layer would never see the
// low-confidence candidates that escalation exists to rescue.
func TestDetectDoesNotFilter(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"candidates":[
			{"bbox":[0,0,10,10],"is_checked":false,"confidence":0.31}],
			"width":10,"height":10}`)
	})
	res, err := client.Detect(context.Background(), testImage())
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("a 0.31-confidence candidate was filtered by the adapter: %+v", res.Detections)
	}
}

func TestDetectSendsMultipartWithFilename(t *testing.T) {
	var gotName, gotContentType string
	var gotBody []byte
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotContentType = r.Header.Get("Content-Type")
		if err := r.ParseMultipartForm(1 << 20); err == nil {
			if f, h, err := r.FormFile("file"); err == nil {
				gotName = h.Filename
				gotBody, _ = io.ReadAll(f)
				_ = f.Close()
			}
		}
		_, _ = io.WriteString(w, okBody)
	})

	if _, err := client.Detect(context.Background(), testImage()); err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if !strings.HasPrefix(gotContentType, "multipart/form-data") {
		t.Errorf("Content-Type = %q", gotContentType)
	}
	// The filename is forwarded because the sidecar logs it; losing it makes a failing
	// document impossible to identify afterwards.
	if gotName != "page.png" {
		t.Errorf("filename = %q, want page.png", gotName)
	}
	if string(gotBody) != "pretend-png" {
		t.Errorf("payload altered in transit: %q", gotBody)
	}
}

func TestDetectSendsTheConfiguredFloor(t *testing.T) {
	var gotFloor string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotFloor = r.URL.Query().Get("floor")
		_, _ = io.WriteString(w, okBody)
	}))
	defer srv.Close()

	client := New(srv.URL, 5*time.Second, WithFloor(0.125))
	if _, err := client.Detect(context.Background(), testImage()); err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if gotFloor != "0.125" {
		t.Errorf("floor query = %q, want 0.125", gotFloor)
	}
}

func TestDetectStatusMapping(t *testing.T) {
	cases := []struct {
		name   string
		status int
		want   error
	}{
		// A retriable outage and a bad upload need opposite handling by the caller, so they
		// must not collapse into one error type.
		{"bad request", http.StatusBadRequest, ErrBadImage},
		{"unsupported media", http.StatusUnsupportedMediaType, ErrBadImage},
		{"server error", http.StatusInternalServerError, ErrUnavailable},
		{"service unavailable", http.StatusServiceUnavailable, ErrUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
			})
			_, err := client.Detect(context.Background(), testImage())
			if !errors.Is(err, tc.want) {
				t.Fatalf("status %d gave %v, want %v", tc.status, err, tc.want)
			}
		})
	}
}

func TestDetectUnexpectedStatusIsNotSwallowed(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	})
	_, err := client.Detect(context.Background(), testImage())
	if err == nil {
		t.Fatal("an unexpected status must not be treated as success")
	}
}

func TestDetectMalformedJSONIsAnError(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "{not json")
	})
	if _, err := client.Detect(context.Background(), testImage()); err == nil {
		t.Fatal("malformed JSON was accepted")
	}
}

// TestDetectSkipsMalformedBoxesWithoutFailingThePage: one bad entry must not discard the
// other hundred valid detections on the same document.
func TestDetectSkipsMalformedBoxesWithoutFailingThePage(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"candidates":[
			{"bbox":[1,2,3],"is_checked":false,"confidence":0.9},
			{"bbox":[10,20,32,42],"is_checked":true,"confidence":0.97}],
			"width":100,"height":100}`)
	})
	res, err := client.Detect(context.Background(), testImage())
	if err != nil {
		t.Fatalf("a malformed box failed the whole page: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("got %d detections, want the 1 well-formed one", len(res.Detections))
	}
}

func TestDetectUnreachableSidecar(t *testing.T) {
	// Port 0 on the loopback is never listening, so this exercises the dial-failure path.
	client := New("http://127.0.0.1:0", time.Second)
	_, err := client.Detect(context.Background(), testImage())
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v, want ErrUnavailable", err)
	}
}

func TestDetectHonoursContextCancellation(t *testing.T) {
	// An abandoned inbound request must tear the outbound call down rather than leaving a
	// goroutine writing into a dead connection.
	// The handler must be slow, never infinite. Blocking purely on r.Context().Done() looks
	// right but deadlocks the suite: httptest.Server.Close waits for outstanding requests,
	// and the server-side context is not guaranteed to fire promptly once the client gives
	// up. The timer is the escape hatch that keeps a cancellation test from hanging CI.
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Detect(ctx, testImage())
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestHealthReportsModelState(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("health probe hit %q", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"status":"degraded","model_loaded":false,"detail":"no artifact"}`)
	})
	ready, detail, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	if ready {
		t.Error("ready is true despite model_loaded false")
	}
	if detail != "no artifact" {
		t.Errorf("detail = %q", detail)
	}
}

func TestHealthOnUnreachableSidecar(t *testing.T) {
	client := New("http://127.0.0.1:0", time.Second)
	if _, _, err := client.Health(context.Background()); err == nil {
		t.Fatal("expected an error from an unreachable sidecar")
	}
}

func TestNewTrimsTrailingSlash(t *testing.T) {
	// Otherwise every request URL contains a double slash, which some proxies reject.
	if got := New("http://engine:8000/", time.Second).baseURL; got != "http://engine:8000" {
		t.Fatalf("baseURL = %q", got)
	}
}

func TestNameIsStable(t *testing.T) {
	if got := New("http://x", time.Second).Name(); got != domain.EngineLocal {
		t.Fatalf("Name() = %q", got)
	}
}

func TestEncodeMultipartDefaultsMissingMetadata(t *testing.T) {
	// A part with no Content-Type at all is rejected by some HTTP stacks, so the encoder
	// substitutes a generic one rather than omitting the header.
	body, contentType, err := encodeMultipart(domain.Image{Data: []byte("x")})
	if err != nil {
		t.Fatalf("encodeMultipart: %v", err)
	}
	if !strings.HasPrefix(contentType, "multipart/form-data") {
		t.Errorf("contentType = %q", contentType)
	}
	raw, _ := io.ReadAll(body)
	if !strings.Contains(string(raw), "application/octet-stream") {
		t.Error("missing content type was not defaulted")
	}
	if !strings.Contains(string(raw), `filename="upload"`) {
		t.Error("missing filename was not defaulted")
	}
}
