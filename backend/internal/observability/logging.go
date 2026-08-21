// Package observability holds cross-cutting logging setup.
//
// Small on purpose. The JD asks for reliability and observability, and the honest minimum
// that earns those words is structured logs with a request-level record of what each engine
// cost. Metrics and tracing are named in the README's future bucket rather than half-built
// here, because a Prometheus endpoint nobody scrapes is decoration, not observability.
package observability

import (
	"log/slog"
	"os"
	"strings"
)

// NewLogger builds a JSON slog handler at the requested level.
//
// JSON rather than text because these logs are destined for a collector that parses them;
// a human reading them locally can pipe through jq, whereas a collector cannot un-format
// prose. Unrecognised level strings fall back to info rather than failing, since a typo in a
// log level should never be the reason a service refuses to start.
func NewLogger(level string) *slog.Logger {
	var lvl slog.Level
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl}))
}
