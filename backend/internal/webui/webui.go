// Package webui serves the compiled overlay viewer from inside the API binary.
//
// Why the Go service serves the UI at all. In development the three services run separately
// under compose, which is the honest shape: a static bundle behind nginx, an API, and a
// Python sidecar. In production they collapse into one container, and at that point a third
// process serving twelve static files is a liability rather than a separation -- another
// thing to supervise, another thing that can be up while the API is down, and a whole nginx
// image for a job the Go standard library already does.
//
// Embedding rather than mounting a volume is what makes the artifact self-contained: the
// binary and the UI it was built against cannot be separated, so there is no deployment in
// which a stale bundle talks to a newer API.
package webui

import (
	"embed"
	"io/fs"
	"net/http"
	"path"
	"strings"
)

// dist holds the built frontend. The `all:` prefix keeps files whose names begin with `_` or
// `.`, which bundlers routinely emit and which the default embed rules would silently drop.
//
// A placeholder index.html is committed so that `go build` and `go test` work on a clean
// checkout: go:embed refuses to compile against a directory with no matching files, and
// making the test suite depend on `npm run build` having been run first would be a poor
// trade for a file that says what it is.
//
//go:embed all:dist
var dist embed.FS

// Handler serves the bundle, falling back to index.html for unknown paths.
//
// The fallback is what makes a single-page app survive a refresh on a client-side route: the
// browser asks the server for a path only the bundle knows about, and answering 404 would
// break a URL the user got by using the app normally. It is deliberately limited to paths
// that look like routes rather than assets -- a missing .js or .css must 404, because
// answering it with HTML turns a broken build into a console parse error several steps
// removed from its cause.
func Handler() http.Handler {
	sub, err := fs.Sub(dist, "dist")
	if err != nil {
		// Unreachable: the directory is embedded at compile time, so a failure here means
		// the binary is malformed rather than misconfigured.
		panic("webui: embedded dist is unreadable: " + err.Error())
	}
	files := http.FileServerFS(sub)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(path.Clean(r.URL.Path), "/")
		if name == "" {
			name = "index.html"
		}

		if _, statErr := fs.Stat(sub, name); statErr != nil {
			if path.Ext(name) != "" {
				http.NotFound(w, r)
				return
			}
			serveIndex(w, r, sub)
			return
		}

		setCacheHeaders(w, name)
		files.ServeHTTP(w, r)
	})
}

// serveIndex writes the SPA shell for a route the bundle owns.
func serveIndex(w http.ResponseWriter, r *http.Request, sub fs.FS) {
	body, err := fs.ReadFile(sub, "index.html")
	if err != nil {
		http.NotFound(w, r)
		return
	}
	setCacheHeaders(w, "index.html")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(body)
}

// setCacheHeaders applies the split every hashed-asset bundler assumes.
//
// Asset filenames carry a content hash, so a given URL's contents can never change and it is
// safe to cache them for a year -- which is what makes a repeat visit free. index.html is the
// opposite: its URL is stable and its contents change with every deploy, so caching it is how
// a browser ends up requesting assets that no longer exist.
func setCacheHeaders(w http.ResponseWriter, name string) {
	if name == "index.html" {
		w.Header().Set("Cache-Control", "no-cache")
		return
	}
	if strings.HasPrefix(name, "assets/") {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	}
}
