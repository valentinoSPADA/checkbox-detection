package observability

import (
	"log/slog"
	"testing"
)

func TestNewLoggerLevels(t *testing.T) {
	cases := map[string]slog.Level{
		"debug": slog.LevelDebug,
		"DEBUG": slog.LevelDebug,
		" info": slog.LevelInfo,
		"warn":  slog.LevelWarn,
		"error": slog.LevelError,
	}
	for raw, want := range cases {
		t.Run(raw, func(t *testing.T) {
			log := NewLogger(raw)
			if !log.Enabled(t.Context(), want) {
				t.Fatalf("logger built from %q does not emit at %v", raw, want)
			}
		})
	}
}

// TestUnknownLevelFallsBack: a typo in a log level must never be the reason a service
// refuses to start, so an unrecognised value resolves to info rather than failing.
func TestUnknownLevelFallsBack(t *testing.T) {
	log := NewLogger("chatty")
	if !log.Enabled(t.Context(), slog.LevelInfo) {
		t.Fatal("unknown level did not fall back to info")
	}
	if log.Enabled(t.Context(), slog.LevelDebug) {
		t.Fatal("unknown level fell back to debug, which would flood production logs")
	}
}
