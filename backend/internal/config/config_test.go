package config

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

// clearEnv unsets every variable Load reads, so a test never inherits a value from the
// developer's shell or from a sibling test. t.Setenv restores the previous value at cleanup.
func clearEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"ADDR", "DETECTOR_URL", "DETECTOR_TIMEOUT", "REQUEST_TIMEOUT", "MAX_UPLOAD_BYTES",
		"ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "VLM_TIMEOUT", "VLM_MAX_IMAGE_DIM",
		"DEFAULT_ENGINE", "MIN_CONFIDENCE", "IOU_THRESHOLD", "MAX_DETECTIONS",
		"ESCALATE_ABOVE", "ESCALATE_BELOW", "MAX_ESCALATIONS", "CORS_ORIGINS", "LOG_LEVEL",
	} {
		t.Setenv(k, "")
	}
}

func TestLoadDefaults(t *testing.T) {
	clearEnv(t)
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load with a clean environment failed: %v", err)
	}
	if cfg.Addr != ":8080" {
		t.Errorf("Addr = %q, want :8080", cfg.Addr)
	}
	if cfg.DefaultEngine != domain.EngineLocal {
		t.Errorf("DefaultEngine = %q, want local", cfg.DefaultEngine)
	}
	if cfg.VLMEnabled() {
		t.Error("VLMEnabled is true with no API key set")
	}
	if cfg.Policy.MinConfidence != domain.DefaultPolicy().MinConfidence {
		t.Errorf("MinConfidence = %v, want the domain default", cfg.Policy.MinConfidence)
	}
}

// TestLoadWithoutAPIKeySucceeds is the requirement that an evaluator can clone the repository
// and run it with no credentials at all. If a missing key were an error, `docker compose up`
// would fail on a clean checkout.
func TestLoadWithoutAPIKeySucceeds(t *testing.T) {
	clearEnv(t)
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load without an API key must succeed, got: %v", err)
	}
	if cfg.VLMEnabled() {
		t.Error("vision engines reported as enabled without a key")
	}
}

func TestLoadOverrides(t *testing.T) {
	clearEnv(t)
	t.Setenv("ADDR", ":9999")
	t.Setenv("DETECTOR_URL", "http://engine:1234/")
	t.Setenv("DETECTOR_TIMEOUT", "42s")
	t.Setenv("MIN_CONFIDENCE", "0.75")
	t.Setenv("MAX_UPLOAD_BYTES", "1024")
	t.Setenv("CORS_ORIGINS", "http://a.example , http://b.example")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Addr != ":9999" {
		t.Errorf("Addr = %q", cfg.Addr)
	}
	// The trailing slash must be stripped, or every request URL ends up with a double slash.
	if cfg.DetectorURL != "http://engine:1234" {
		t.Errorf("DetectorURL = %q, want the trailing slash stripped", cfg.DetectorURL)
	}
	if cfg.DetectorTimeout != 42*time.Second {
		t.Errorf("DetectorTimeout = %v", cfg.DetectorTimeout)
	}
	if cfg.Policy.MinConfidence != 0.75 {
		t.Errorf("MinConfidence = %v", cfg.Policy.MinConfidence)
	}
	if len(cfg.CORSOrigins) != 2 || cfg.CORSOrigins[0] != "http://a.example" {
		t.Errorf("CORSOrigins = %v, want two trimmed entries", cfg.CORSOrigins)
	}
}

// TestMalformedValuesFallBackRatherThanFailing covers a deliberate asymmetry: a value that
// cannot be parsed at all falls back to the default, while a value that parses into an
// unusable configuration is refused. A typo in a duration should not stop the service; a
// confidence of 7 should, because it would silently return nothing.
func TestMalformedValuesFallBackRatherThanFailing(t *testing.T) {
	clearEnv(t)
	t.Setenv("DETECTOR_TIMEOUT", "not-a-duration")
	t.Setenv("MAX_DETECTIONS", "several")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unparseable values should fall back, not fail: %v", err)
	}
	if cfg.DetectorTimeout != 60*time.Second {
		t.Errorf("DetectorTimeout = %v, want the 60s default", cfg.DetectorTimeout)
	}
}

func TestValidateRejectsUnusableConfigurations(t *testing.T) {
	base := func() Config {
		return Config{Addr: ":8080", DetectorURL: "http://d", MaxUploadBytes: 1024,
			DefaultEngine: domain.EngineLocal, Policy: domain.DefaultPolicy()}
	}
	cases := []struct {
		name   string
		mutate func(*Config)
		want   string
	}{
		{"empty addr", func(c *Config) { c.Addr = "" }, "ADDR"},
		{"empty detector url", func(c *Config) { c.DetectorURL = "" }, "DETECTOR_URL"},
		{"zero upload limit", func(c *Config) { c.MaxUploadBytes = 0 }, "MAX_UPLOAD_BYTES"},
		{"confidence out of range", func(c *Config) { c.Policy.MinConfidence = 7 }, "MIN_CONFIDENCE"},
		{"iou out of range", func(c *Config) { c.Policy.IoUThreshold = -1 }, "IOU_THRESHOLD"},
		// An inverted band makes escalation a silent no-op: the assisted engine would keep
		// reporting itself as assisted while never escalating anything.
		{"inverted escalation band", func(c *Config) {
			c.Policy.EscalateAbove, c.Policy.EscalateBelow = 0.9, 0.2
		}, "ESCALATE_ABOVE"},
		{"vlm default without a key", func(c *Config) {
			c.DefaultEngine = domain.EngineVLM
		}, "ANTHROPIC_API_KEY"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base()
			tc.mutate(&cfg)
			err := cfg.Validate()
			if err == nil {
				t.Fatalf("expected %s to be rejected", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not name %s", err, tc.want)
			}
		})
	}
}

func TestValidateReportsEveryProblemAtOnce(t *testing.T) {
	// Fixing one variable, restarting, and discovering the next is a miserable loop; all
	// problems are joined into a single error so one restart is enough.
	cfg := Config{Policy: domain.DefaultPolicy()}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected errors")
	}
	var joined interface{ Unwrap() []error }
	if !errors.As(err, &joined) {
		t.Fatalf("expected a joined error, got %T", err)
	}
	if len(joined.Unwrap()) < 3 {
		t.Fatalf("only %d problems reported; expected every one", len(joined.Unwrap()))
	}
}

func TestUnknownDefaultEngineIsRejected(t *testing.T) {
	clearEnv(t)
	t.Setenv("DEFAULT_ENGINE", "telepathy")
	if _, err := Load(); err == nil {
		t.Fatal("an unknown DEFAULT_ENGINE should fail at startup, not at first request")
	}
}

func TestVLMEnabledWithKey(t *testing.T) {
	clearEnv(t)
	t.Setenv("ANTHROPIC_API_KEY", "sk-test")
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !cfg.VLMEnabled() {
		t.Error("VLMEnabled is false despite a key being present")
	}
}
