package httpapi

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"strconv"
	"time"

	"github.com/vspada/checkbox-detection/backend/internal/detector/localengine"
	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

// formFieldNames are the multipart field names accepted for the upload.
//
// More than one is accepted because the challenge specifies "a document image (sent as a file
// upload)" without naming the field, and a reviewer reaching for curl is as likely to write
// -F image=@page.png as -F file=@page.png. Rejecting a correct upload over a field name would
// be a needless failure at the very first thing anyone tries.
var formFieldNames = []string{"file", "image", "document", "upload"}

// handleDetect implements POST /detect -- the endpoint the challenge specifies.
//
// Request: multipart/form-data with an image in a field named file (or image/document/upload).
//
// Query parameters, all optional:
//
//	engine=local|vlm|assisted   which detection strategy to run (default from config)
//	min_confidence=0.0..1.0     override the confidence floor for this request
//	verbose=true                include confidence, engine attribution and pipeline counters
//
// Response: {"boxes": [{"bbox": [x1,y1,x2,y2], "is_checked": bool}, ...]} exactly as
// specified; the verbose form adds keys without reordering or renaming the specified ones.
//
// Side effects: calls the detection sidecar, and -- for the vlm and assisted engines -- the
// Anthropic API, which costs money per request. Nothing is persisted; the upload is held in
// memory for the life of the request and never written to disk.
//
// Failure modes: 400 for a missing or undecodable image or an unknown engine; 413 for an
// oversized upload; 503 when the sidecar is unreachable; 504 when the request deadline
// elapses; 500 otherwise.
func (s *Server) handleDetect(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout)
	defer cancel()

	engineName, err := domain.ParseEngine(r.URL.Query().Get("engine"), s.defaultNm)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "unknown engine", err)
		return
	}
	engine, ok := s.engines[engineName]
	if !ok {
		s.writeError(w, http.StatusBadRequest, "engine not available on this instance",
			fmt.Errorf("engine %q requires configuration that is not present (an API key, typically)", engineName))
		return
	}

	img, err := s.readUpload(r)
	if err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, errTooLarge) {
			status = http.StatusRequestEntityTooLarge
		}
		s.writeError(w, status, "invalid upload", err)
		return
	}

	policy := s.policy
	if raw := r.URL.Query().Get("min_confidence"); raw != "" {
		v, convErr := strconv.ParseFloat(raw, 64)
		if convErr != nil || v < 0 || v > 1 {
			s.writeError(w, http.StatusBadRequest, "min_confidence must be a number within [0,1]", convErr)
			return
		}
		policy.MinConfidence = v
	}

	started := time.Now()
	result, err := engine.Detect(ctx, img)
	if err != nil {
		s.writeDetectError(w, engineName, err)
		return
	}

	// Policy runs here, outside the engine, so every strategy is filtered and deduplicated
	// by identical rules. That is what makes an accuracy comparison between engines mean
	// anything -- otherwise a "better" engine could just be one with a looser threshold.
	before := len(result.Detections)
	result.Detections = policy.Apply(result.Detections)
	result.Stats.Candidates = before
	result.Stats.Returned = len(result.Detections)
	elapsed := time.Since(started)

	s.log.Info("detect",
		slog.String("engine", string(engineName)),
		slog.String("filename", img.Filename),
		slog.Int("bytes", len(img.Data)),
		slog.Int("candidates", before),
		slog.Int("returned", len(result.Detections)),
		slog.Int("escalated", result.Stats.Escalated),
		slog.Int64("ms", elapsed.Milliseconds()))

	if isTruthy(r.URL.Query().Get("verbose")) {
		s.writeJSON(w, http.StatusOK, toVerbose(result, elapsed.Milliseconds()))
		return
	}
	s.writeJSON(w, http.StatusOK, toStrict(result.Detections))
}

// writeDetectError maps an engine failure onto the right status code.
//
// The distinction that matters operationally is retriable versus not: a 503 tells a client to
// back off and try again, a 400 tells it to stop and fix the request. Collapsing both into
// 500 -- the easy default -- makes an outage indistinguishable from a bad upload in every
// dashboard downstream.
func (s *Server) writeDetectError(w http.ResponseWriter, engine domain.EngineName, err error) {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		s.log.Warn("detect timed out", slog.String("engine", string(engine)))
		s.writeError(w, http.StatusGatewayTimeout, "detection timed out", nil)
	case errors.Is(err, context.Canceled):
		// The client hung up; nothing can be written, but the log line prevents this from
		// looking like an unexplained gap in the metrics.
		s.log.Info("detect cancelled by client", slog.String("engine", string(engine)))
		s.writeError(w, 499, "request cancelled", nil)
	case errors.Is(err, localengine.ErrBadImage):
		s.writeError(w, http.StatusBadRequest, "image could not be decoded", err)
	case errors.Is(err, localengine.ErrUnavailable):
		s.log.Error("detection engine unavailable", slog.String("error", err.Error()))
		s.writeError(w, http.StatusServiceUnavailable, "detection engine unavailable", nil)
	default:
		s.log.Error("detect failed", slog.String("engine", string(engine)), slog.String("error", err.Error()))
		s.writeError(w, http.StatusInternalServerError, "detection failed", nil)
	}
}

var errTooLarge = errors.New("upload exceeds the configured size limit")

// readUpload extracts the image from a multipart request.
//
// The body is wrapped in a MaxBytesReader before parsing rather than checking Content-Length
// afterwards, because Content-Length is caller-supplied and a chunked upload omits it
// entirely; bounding the reader is the only limit an attacker cannot simply lie past.
func (s *Server) readUpload(r *http.Request) (domain.Image, error) {
	r.Body = http.MaxBytesReader(nil, r.Body, s.cfg.MaxUploadBytes)

	if err := r.ParseMultipartForm(8 << 20); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			return domain.Image{}, errTooLarge
		}
		return domain.Image{}, fmt.Errorf("expected multipart/form-data with an image field: %w", err)
	}

	var (
		file   multipart.File
		header *multipart.FileHeader
		err    error
	)
	for _, name := range formFieldNames {
		file, header, err = r.FormFile(name)
		if err == nil {
			break
		}
	}
	if file == nil {
		return domain.Image{}, fmt.Errorf("no image found; expected one of the form fields %v", formFieldNames)
	}
	defer func() { _ = file.Close() }()

	data, err := io.ReadAll(io.LimitReader(file, s.cfg.MaxUploadBytes+1))
	if err != nil {
		return domain.Image{}, fmt.Errorf("reading upload: %w", err)
	}
	if int64(len(data)) > s.cfg.MaxUploadBytes {
		return domain.Image{}, errTooLarge
	}
	if len(data) == 0 {
		return domain.Image{}, errors.New("uploaded file is empty")
	}

	return domain.Image{
		Data:        data,
		Filename:    header.Filename,
		ContentType: header.Header.Get("Content-Type"),
	}, nil
}

// isTruthy accepts the spellings a human types into a query string by hand.
func isTruthy(v string) bool {
	switch v {
	case "1", "true", "TRUE", "True", "yes", "on", "":
		return v != ""
	default:
		return false
	}
}
