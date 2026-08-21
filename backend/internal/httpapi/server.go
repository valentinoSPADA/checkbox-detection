// Package httpapi is the inbound adapter: it parses HTTP, delegates to a domain.Detector,
// applies domain.Policy to the result, and shapes a response.
//
// It holds no detection logic. If a handler in this package ever branches on a confidence
// value or compares two boxes, that logic has leaked out of the domain and belongs back in
// domain.Policy -- which is where it can be unit-tested without a server.
package httpapi

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/vspada/checkbox-detection/backend/internal/config"
	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

// ReadinessProbe reports whether a downstream dependency is usable.
//
// Kept as a function type rather than an interface because there is exactly one
// implementation and the wiring layer already owns the concrete client; a single-method
// interface here would be ceremony without a second implementer.
type ReadinessProbe func() (ready bool, detail string)

// Server holds the handler dependencies.
type Server struct {
	engines   map[domain.EngineName]domain.Detector
	defaultNm domain.EngineName
	policy    domain.Policy
	cfg       config.Config
	log       *slog.Logger
	ready     ReadinessProbe
	version   string
}

// Options configures a Server.
type Options struct {
	Engines       map[domain.EngineName]domain.Detector
	DefaultEngine domain.EngineName
	Policy        domain.Policy
	Config        config.Config
	Logger        *slog.Logger
	Readiness     ReadinessProbe
	Version       string
}

// New builds a Server.
func New(opts Options) *Server {
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	if opts.Readiness == nil {
		opts.Readiness = func() (bool, string) { return true, "" }
	}
	return &Server{
		engines:   opts.Engines,
		defaultNm: opts.DefaultEngine,
		policy:    opts.Policy,
		cfg:       opts.Config,
		log:       opts.Logger,
		ready:     opts.Readiness,
		version:   opts.Version,
	}
}

// Routes returns the fully-wired handler, middleware included.
//
// Middleware order is deliberate and outermost-first: recovery must wrap everything so a
// panic in any later layer still produces a response; request logging sits inside recovery so
// a panicking request is still logged with its timing; CORS is innermost of the three so that
// even an error response carries the headers a browser needs to read it -- without that, the
// UI shows an opaque network failure instead of the actual message.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /detect", s.handleDetect)
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("GET /version", s.handleVersion)
	mux.HandleFunc("GET /engines", s.handleEngines)

	return recoverMiddleware(s.log)(
		loggingMiddleware(s.log)(
			corsMiddleware(s.cfg.CORSOrigins)(mux)))
}

// writeJSON serialises v with the given status.
//
// Encoding errors are logged rather than returned: the status line and headers are already
// on the wire by the time encoding runs, so there is no way left to tell the client anything
// different, and pretending otherwise would just produce a malformed body plus a lie.
func (s *Server) writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		s.log.Error("writing response body", slog.String("error", err.Error()))
	}
}

// writeError emits the shared error envelope.
func (s *Server) writeError(w http.ResponseWriter, status int, msg string, err error) {
	out := errorOut{Error: msg}
	if err != nil && status < 500 {
		// Detail is withheld on 5xx: internal errors can carry upstream URLs, file paths
		// or driver messages, and this endpoint accepts uploads from untrusted callers.
		out.Details = err.Error()
	}
	s.writeJSON(w, status, out)
}

// handleHealth answers GET /health -- process liveness only.
//
// Deliberately does not touch the sidecar. A liveness probe that depends on a downstream
// service turns that service's outage into a restart loop of this one, which removes the
// only component still able to report what is wrong.
func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "version": s.version})
}

// handleReady answers GET /ready -- whether this instance can actually serve traffic.
//
// Returns 503 with a reason when the detection sidecar is unreachable or has no model
// loaded, which is what a load balancer should act on. Side effects: issues one health call
// downstream.
func (s *Server) handleReady(w http.ResponseWriter, _ *http.Request) {
	ok, detail := s.ready()
	status := http.StatusOK
	state := "ready"
	if !ok {
		status, state = http.StatusServiceUnavailable, "not_ready"
	}
	s.writeJSON(w, status, map[string]any{"status": state, "detail": detail, "version": s.version})
}

// handleVersion answers GET /version. No side effects.
func (s *Server) handleVersion(w http.ResponseWriter, _ *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]any{"version": s.version})
}

// handleEngines answers GET /engines -- which detection strategies this instance can run.
//
// Exists because engine availability is configuration-dependent: without an API key the
// vlm and assisted engines are not registered, and a UI that offered them anyway would be
// showing the user a button that always fails. No side effects.
func (s *Server) handleEngines(w http.ResponseWriter, _ *http.Request) {
	names := make([]string, 0, len(s.engines))
	for _, n := range []domain.EngineName{domain.EngineLocal, domain.EngineVLM, domain.EngineAssisted} {
		if _, ok := s.engines[n]; ok {
			names = append(names, string(n))
		}
	}
	s.writeJSON(w, http.StatusOK, map[string]any{
		"engines": names,
		"default": string(s.defaultNm),
	})
}

// ListenAndServe runs the server with sane timeouts until the context is cancelled.
//
// ReadHeaderTimeout is set explicitly to close the slowloris hole that Go's zero-valued
// Server leaves open. WriteTimeout is derived from the configured request timeout with
// headroom, because the vlm engine legitimately takes tens of seconds and a shorter write
// deadline would sever responses mid-flight for the slowest -- and most interesting -- calls.
func (s *Server) ListenAndServe() error {
	srv := &http.Server{
		Addr:              s.cfg.Addr,
		Handler:           s.Routes(),
		ReadHeaderTimeout: 15 * time.Second,
		ReadTimeout:       2 * time.Minute,
		WriteTimeout:      s.cfg.RequestTimeout + 30*time.Second,
		IdleTimeout:       90 * time.Second,
	}
	s.log.Info("http server listening", slog.String("addr", s.cfg.Addr))
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
