// Package config loads and validates runtime configuration from the environment.
//
// Everything the service needs is read once at startup and validated eagerly, so a
// misconfiguration fails the process immediately with a readable message instead of
// surfacing as a confusing 500 on the first request that happens to need the missing value.
package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

// Config is the fully-resolved runtime configuration.
type Config struct {
	// Addr is the listen address for the public HTTP API.
	Addr string
	// DetectorURL is the base URL of the Python detection sidecar.
	DetectorURL string
	// DetectorTimeout bounds one sidecar call.
	DetectorTimeout time.Duration
	// RequestTimeout bounds an entire inbound request.
	RequestTimeout time.Duration
	// MaxUploadBytes rejects oversized uploads before they are buffered.
	//
	// Byte size is the only limit this service can enforce cheaply: it does not decode the
	// image -- the sidecar does -- so pixel-dimension limits live there, where the decode
	// already happens. Enforcing them here would mean decoding every page twice.
	MaxUploadBytes int64
	// DefaultEngine is used when a request does not name one.
	DefaultEngine domain.EngineName
	// Policy holds the detection thresholds.
	Policy domain.Policy
	// CORSOrigins lists browser origins allowed to call the API.
	CORSOrigins []string
	// LogLevel selects slog's minimum level.
	LogLevel string
}

// Load reads configuration from the environment and validates it.
//
// Returns an error rather than calling os.Exit so that tests can exercise validation
// directly. Note what is *not* an error: a missing ANTHROPIC_API_KEY. The service is
// expected to run without one -- an evaluator cloning the repository must get a working
// system from `docker compose up` alone -- so the VLM engines are simply not registered and
// selecting one yields a clear 400 instead of a startup failure.
func Load() (Config, error) {
	cfg := Config{
		Addr:            env("ADDR", ":8080"),
		DetectorURL:     strings.TrimRight(env("DETECTOR_URL", "http://detector:8000"), "/"),
		DetectorTimeout: envDuration("DETECTOR_TIMEOUT", 60*time.Second),
		RequestTimeout:  envDuration("REQUEST_TIMEOUT", 120*time.Second),
		MaxUploadBytes:  int64(envInt("MAX_UPLOAD_BYTES", 25*1024*1024)),
		CORSOrigins:     splitAndTrim(env("CORS_ORIGINS", "*")),
		LogLevel:        env("LOG_LEVEL", "info"),
	}

	policy := domain.DefaultPolicy()
	policy.MinConfidence = envFloat("MIN_CONFIDENCE", policy.MinConfidence)
	policy.IoUThreshold = envFloat("IOU_THRESHOLD", policy.IoUThreshold)
	policy.MaxDetections = envInt("MAX_DETECTIONS", policy.MaxDetections)
	cfg.Policy = policy

	engine, err := domain.ParseEngine(os.Getenv("DEFAULT_ENGINE"), domain.EngineLocal)
	if err != nil {
		return Config{}, fmt.Errorf("DEFAULT_ENGINE: %w", err)
	}
	cfg.DefaultEngine = engine

	return cfg, cfg.Validate()
}

// Validate rejects combinations that would produce confusing runtime behaviour.
//
// Refused at startup rather than at the first request, because a service that boots and then
// fails every call looks healthy to whatever is watching it. Each check below guards a
// setting whose wrong value produces a *plausible* result rather than an obvious error --
// a threshold outside [0,1] silently returns everything or nothing.
func (c Config) Validate() error {
	var errs []error
	if c.Addr == "" {
		errs = append(errs, errors.New("ADDR must not be empty"))
	}
	if c.DetectorURL == "" {
		errs = append(errs, errors.New("DETECTOR_URL must not be empty"))
	}
	if c.MaxUploadBytes <= 0 {
		errs = append(errs, errors.New("MAX_UPLOAD_BYTES must be positive"))
	}
	if c.Policy.MinConfidence < 0 || c.Policy.MinConfidence > 1 {
		errs = append(errs, errors.New("MIN_CONFIDENCE must be within [0,1]"))
	}
	if c.Policy.IoUThreshold < 0 || c.Policy.IoUThreshold > 1 {
		errs = append(errs, errors.New("IOU_THRESHOLD must be within [0,1]"))
	}
	return errors.Join(errs...)
}

func env(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func envDuration(key string, def time.Duration) time.Duration {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func splitAndTrim(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}
