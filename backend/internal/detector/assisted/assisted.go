// Package assisted composes the local pipeline with the vision model.
//
// This is the engine the architecture exists for. The local pipeline is fast, free and
// confident on the easy majority of a page; the vision model is slow, paid and better on the
// ambiguous minority. Running the model over every candidate wastes almost all of that money
// on cases already decided, and running it over none leaves the hard cases wrong. Escalating
// only the uncertainty band spends the budget where it can still change an answer.
package assisted

import (
	"context"
	"log/slog"

	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

// Adjudicator re-judges specific candidate regions with a costlier model.
//
// Declared here as a narrow interface rather than importing the vlm client concretely, so
// the escalation logic can be tested with a stub and so a different second-opinion engine
// could be substituted without touching this package.
type Adjudicator interface {
	Adjudicate(ctx context.Context, img domain.Image, candidates []domain.Detection,
		contextFactor float64) ([]domain.Detection, error)
}

// Engine runs the local detector and escalates its uncertain candidates.
type Engine struct {
	base       domain.Detector
	adjudicate Adjudicator
	policy     domain.Policy
	ctxFactor  float64
	log        *slog.Logger
}

// New builds the assisted engine. contextFactor controls how much surrounding page is
// included in each escalated crop; 3.0 is the tuned default and means the crop is three
// times the candidate's own size.
func New(base domain.Detector, adj Adjudicator, policy domain.Policy, contextFactor float64,
	log *slog.Logger) *Engine {
	if contextFactor <= 0 {
		contextFactor = 3.0
	}
	return &Engine{base: base, adjudicate: adj, policy: policy, ctxFactor: contextFactor, log: log}
}

// Name identifies this engine in responses and metrics.
func (e *Engine) Name() domain.EngineName { return domain.EngineAssisted }

// Detect runs the local pipeline, then asks the vision model about its uncertain candidates.
//
// Escalation is best-effort by design: if the model call fails -- rate limit, timeout,
// upstream outage -- the local detections are returned unchanged and the error is logged
// rather than propagated. A paid enhancement that is unavailable should degrade the answer,
// not destroy it, and a caller who wanted the model's judgement unconditionally can select
// the vlm engine instead and receive its failures directly.
//
// Note that suppression is deliberately not applied here. The caller applies domain.Policy
// to the merged set, so that the local and escalated detections compete under exactly the
// same rules that govern a single-engine response.
func (e *Engine) Detect(ctx context.Context, img domain.Image) (domain.Result, error) {
	base, err := e.base.Detect(ctx, img)
	if err != nil {
		return domain.Result{}, err
	}

	uncertain := e.policy.SelectForEscalation(base.Detections)
	stats := base.Stats
	stats.Escalated = len(uncertain)

	if len(uncertain) == 0 {
		base.Engine = domain.EngineAssisted
		base.Stats = stats
		return base, nil
	}

	revised, err := e.adjudicate.Adjudicate(ctx, img, uncertain, e.ctxFactor)
	if err != nil {
		e.log.Warn("escalation failed; returning local detections unchanged",
			slog.Int("candidates", len(uncertain)), slog.String("error", err.Error()))
		base.Engine = domain.EngineAssisted
		base.Stats = stats
		return base, nil
	}

	merged := e.policy.Merge(base.Detections, revised)
	return domain.Result{
		Detections: merged,
		Width:      base.Width,
		Height:     base.Height,
		Engine:     domain.EngineAssisted,
		Stats:      stats,
	}, nil
}
