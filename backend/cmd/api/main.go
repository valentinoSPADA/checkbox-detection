// Command api is the public HTTP entrypoint for checkbox detection.
//
// It is the composition root and the only place in the program that knows which concrete
// adapters exist. Everything below it depends on the domain.Detector port, which is what
// allows the local pipeline, the vision model and the assisted combination to be swapped by
// a query parameter without any of them knowing the others exist.
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/vspada/checkbox-detection/backend/internal/config"
	"github.com/vspada/checkbox-detection/backend/internal/detector/assisted"
	"github.com/vspada/checkbox-detection/backend/internal/detector/localengine"
	"github.com/vspada/checkbox-detection/backend/internal/detector/vlm"
	"github.com/vspada/checkbox-detection/backend/internal/domain"
	"github.com/vspada/checkbox-detection/backend/internal/httpapi"
	"github.com/vspada/checkbox-detection/backend/internal/observability"
)

// version is overridden at build time via -ldflags "-X main.version=...".
// Reported by /version and on every startup line, so a running container can be tied back to
// a commit without guessing from an image tag.
var version = "dev"

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", slog.String("error", err.Error()))
		os.Exit(1)
	}
}

// run wires everything and blocks until a termination signal arrives.
//
// Separated from main so that every failure path returns an error instead of calling
// os.Exit deep inside initialisation, which would make the startup sequence untestable.
func run() error {
	cfg, err := config.Load()
	if err != nil {
		// Logged with the default logger: configuration is what tells us how to build the
		// real one, so this is the one message that cannot use it.
		return err
	}

	log := observability.NewLogger(cfg.LogLevel)
	slog.SetDefault(log)

	local := localengine.New(cfg.DetectorURL, cfg.DetectorTimeout)
	engines := map[domain.EngineName]domain.Detector{domain.EngineLocal: local}

	// The vision engines are registered only when a key is present. Registering them
	// unconditionally would let a caller select an engine that fails on every request; this
	// way /engines reports the truth and an unavailable engine yields a clear 400.
	if cfg.VLMEnabled() {
		vision, verr := vlm.New(vlm.Config{
			APIKey:      cfg.AnthropicAPIKey,
			Model:       cfg.AnthropicModel,
			MaxImageDim: cfg.VLMMaxImageDim,
			BatchSize:   cfg.VLMBatchSize,
		})
		if verr != nil {
			return verr
		}
		engines[domain.EngineVLM] = vision
		engines[domain.EngineAssisted] = assisted.New(local, vision, cfg.Policy, 3.0, log)
		log.Info("vision engines enabled", slog.String("model", cfg.AnthropicModel))
	} else {
		log.Info("vision engines disabled: ANTHROPIC_API_KEY is not set")
	}

	srv := httpapi.New(httpapi.Options{
		Engines:       engines,
		DefaultEngine: cfg.DefaultEngine,
		Policy:        cfg.Policy,
		Config:        cfg,
		Logger:        log,
		Version:       version,
		Readiness: func() (bool, string) {
			// Bounded independently of the request timeout: a readiness probe that can
			// block for two minutes is useless to the load balancer polling it.
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			ready, detail, herr := local.Health(ctx)
			if herr != nil {
				return false, herr.Error()
			}
			return ready, detail
		},
	})

	// Signal handling exists so that an in-flight page -- which for the vlm engine can be
	// a minute of paid model calls -- is not discarded by a rolling deploy.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()

	log.Info("service started",
		slog.String("version", version),
		slog.String("detector_url", cfg.DetectorURL),
		slog.String("default_engine", string(cfg.DefaultEngine)))

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		log.Info("shutdown signal received")
		return nil
	}
}
