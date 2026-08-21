package httpapi

import (
	"log/slog"
	"net/http"
	"runtime/debug"
	"strings"
	"time"
)

// statusRecorder captures the status code so the logging middleware can report it.
//
// Necessary because http.ResponseWriter is write-only with respect to status: once
// WriteHeader is called the value is gone, and a log line without it cannot distinguish a
// served page from a rejected upload.
type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	n, err := r.ResponseWriter.Write(b)
	r.bytes += n
	return n, err
}

// loggingMiddleware emits one structured line per request.
//
// Structured rather than formatted because these logs are meant to be queried, not read:
// "which engine is slowest at p99" is a filter over fields, and a printf-formatted line makes
// that a regex problem. Health and readiness probes are dropped at debug level, since at a
// typical probe interval they would otherwise be the overwhelming majority of all log volume
// and would push real requests out of any retention window.
func loggingMiddleware(log *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			started := time.Now()
			rec := &statusRecorder{ResponseWriter: w}
			next.ServeHTTP(rec, r)

			level := slog.LevelInfo
			if r.URL.Path == "/health" || r.URL.Path == "/ready" {
				level = slog.LevelDebug
			} else if rec.status >= 500 {
				level = slog.LevelError
			}
			log.Log(r.Context(), level, "request",
				slog.String("method", r.Method),
				slog.String("path", r.URL.Path),
				slog.Int("status", rec.status),
				slog.Int("bytes", rec.bytes),
				slog.Int64("ms", time.Since(started).Milliseconds()))
		})
	}
}

// recoverMiddleware converts a panic into a 500 instead of killing the process.
//
// Worth having specifically because this service decodes untrusted images through third-party
// code paths: a malformed file that trips an index-out-of-range somewhere deep should cost one
// request, not every in-flight request on the instance. The stack is logged, never returned.
func recoverMiddleware(log *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					log.Error("panic recovered",
						slog.Any("panic", rec),
						slog.String("path", r.URL.Path),
						slog.String("stack", string(debug.Stack())))
					w.Header().Set("Content-Type", "application/json; charset=utf-8")
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error":"internal error"}`))
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// corsMiddleware allows the browser UI to call the API cross-origin.
//
// Credentials are never allowed and no cookie is ever read, so a permissive default origin is
// not the risk it would be on a session-bearing API: there is no ambient authority for a
// hostile page to borrow. An explicit origin list is still supported and is what a real
// deployment should set.
func corsMiddleware(origins []string) func(http.Handler) http.Handler {
	allowAll := len(origins) == 0
	for _, o := range origins {
		if o == "*" {
			allowAll = true
		}
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")
			switch {
			case allowAll:
				w.Header().Set("Access-Control-Allow-Origin", "*")
			case origin != "" && contains(origins, origin):
				w.Header().Set("Access-Control-Allow-Origin", origin)
				// Vary matters once the value is origin-dependent: without it a shared
				// cache can serve one origin's allow header to a different origin.
				w.Header().Add("Vary", "Origin")
			}
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.Header().Set("Access-Control-Max-Age", "600")

			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func contains(list []string, v string) bool {
	for _, item := range list {
		if strings.EqualFold(item, v) {
			return true
		}
	}
	return false
}
