package webui

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func get(t *testing.T, target string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, target, nil))
	return rec
}

func TestServesTheShellAtRoot(t *testing.T) {
	rec := get(t, "/")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("Content-Type = %q, want text/html", ct)
	}
}

// TestUnknownRouteServesTheShell covers the reason the fallback exists: a single-page app's
// client routes are URLs the user can refresh, bookmark and share, and 404ing them breaks a
// link they obtained by using the app normally.
func TestUnknownRouteServesTheShell(t *testing.T) {
	rec := get(t, "/some/client/route")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want the shell", rec.Code)
	}
}

// TestMissingAssetIs404 is the other half, and the more important one. Answering a missing
// .js with HTML does not fail -- it half-works, and surfaces as an unexpected-token error in
// the console several steps removed from the broken build that caused it.
func TestMissingAssetIs404(t *testing.T) {
	for _, target := range []string{"/assets/index-deadbeef.js", "/favicon.svg", "/x/y.css"} {
		if rec := get(t, target); rec.Code != http.StatusNotFound {
			t.Errorf("GET %s = %d, want 404", target, rec.Code)
		}
	}
}

// TestCacheHeaders pins the split hashed-asset bundling depends on. Getting it backwards is
// silently expensive one way and silently broken the other: a cached index.html asks for
// asset URLs that no longer exist after a deploy.
func TestCacheHeaders(t *testing.T) {
	if got := get(t, "/").Header().Get("Cache-Control"); got != "no-cache" {
		t.Errorf("index Cache-Control = %q, want no-cache", got)
	}
	// The placeholder bundle has no assets/ directory, so this asserts the rule rather than a
	// live response; the production bundle exercises the same branch.
	rec := httptest.NewRecorder()
	setCacheHeaders(rec, "assets/index-abc123.js")
	if got := rec.Header().Get("Cache-Control"); !strings.Contains(got, "immutable") {
		t.Errorf("asset Cache-Control = %q, want an immutable long cache", got)
	}
}

// TestPathTraversalIsContained guards the one genuinely dangerous class of bug in a static
// handler. The paths below are what an attacker sends to read files outside the served tree.
func TestPathTraversalIsContained(t *testing.T) {
	for _, target := range []string{"/../webui.go", "/../../go.mod", "/assets/../../webui.go"} {
		rec := get(t, target)
		if strings.Contains(rec.Body.String(), "package webui") {
			t.Fatalf("GET %s escaped the embedded filesystem", target)
		}
	}
}
