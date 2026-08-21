package assisted

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"

	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

type stubBase struct {
	result domain.Result
	err    error
	calls  int
}

func (s *stubBase) Name() domain.EngineName { return domain.EngineLocal }

func (s *stubBase) Detect(context.Context, domain.Image) (domain.Result, error) {
	s.calls++
	return s.result, s.err
}

type stubAdjudicator struct {
	out    []domain.Detection
	err    error
	gotIn  []domain.Detection
	calls  int
	factor float64
}

func (s *stubAdjudicator) Adjudicate(_ context.Context, _ domain.Image,
	candidates []domain.Detection, factor float64) ([]domain.Detection, error) {
	s.calls++
	s.gotIn = candidates
	s.factor = factor
	if s.err != nil {
		return nil, s.err
	}
	return s.out, nil
}

func quietLogger() *slog.Logger {
	return slog.New(slog.NewJSONHandler(io.Discard, nil))
}

func policy() domain.Policy {
	return domain.Policy{MinConfidence: 0.95, IoUThreshold: 0.3,
		EscalateAbove: 0.35, EscalateBelow: 0.85, MaxEscalations: 10}
}

func det(x1 int, conf float64, checked bool) domain.Detection {
	return domain.Detection{Box: domain.NewBox(x1, 0, x1+20, 20), Confidence: conf,
		IsChecked: checked, Source: domain.EngineLocal}
}

func TestEscalatesOnlyTheUncertainBand(t *testing.T) {
	base := &stubBase{result: domain.Result{
		Detections: []domain.Detection{
			det(0, 0.99, true),   // confidently right: nothing to gain
			det(40, 0.60, false), // uncertain: worth a paid call
			det(80, 0.10, false), // confidently negative: money wasted
		},
		Width: 100, Height: 100,
	}}
	adj := &stubAdjudicator{}
	engine := New(base, adj, policy(), 3.0, quietLogger())

	res, err := engine.Detect(context.Background(), domain.Image{})
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if adj.calls != 1 {
		t.Fatalf("adjudicator called %d times, want 1", adj.calls)
	}
	if len(adj.gotIn) != 1 || adj.gotIn[0].Confidence != 0.60 {
		t.Fatalf("escalated the wrong candidates: %+v", adj.gotIn)
	}
	if res.Stats.Escalated != 1 {
		t.Errorf("Escalated = %d, want 1", res.Stats.Escalated)
	}
	if res.Engine != domain.EngineAssisted {
		t.Errorf("engine = %q, want assisted", res.Engine)
	}
}

func TestNoEscalationWhenNothingIsUncertain(t *testing.T) {
	base := &stubBase{result: domain.Result{
		Detections: []domain.Detection{det(0, 0.99, true), det(40, 0.02, false)},
	}}
	adj := &stubAdjudicator{}
	engine := New(base, adj, policy(), 3.0, quietLogger())

	res, err := engine.Detect(context.Background(), domain.Image{})
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if adj.calls != 0 {
		t.Fatal("a paid model call was made with nothing uncertain to ask about")
	}
	if len(res.Detections) != 2 {
		t.Fatalf("detections changed without escalation: %+v", res.Detections)
	}
}

// TestEscalationFailureDegradesRatherThanFails is the behaviour that matters operationally:
// a paid enhancement being unavailable -- rate limit, timeout, outage -- must cost the
// enhancement, not the answer.
func TestEscalationFailureDegradesRatherThanFails(t *testing.T) {
	base := &stubBase{result: domain.Result{
		Detections: []domain.Detection{det(0, 0.60, false), det(40, 0.99, true)},
	}}
	adj := &stubAdjudicator{err: errors.New("429 rate limited")}
	engine := New(base, adj, policy(), 3.0, quietLogger())

	res, err := engine.Detect(context.Background(), domain.Image{})
	if err != nil {
		t.Fatalf("an escalation failure destroyed the response: %v", err)
	}
	if len(res.Detections) != 2 {
		t.Fatalf("local detections were lost: %+v", res.Detections)
	}
	if res.Engine != domain.EngineAssisted {
		t.Errorf("engine = %q; the response should still report how it was produced", res.Engine)
	}
}

func TestBaseFailurePropagates(t *testing.T) {
	// The local engine failing is a real failure -- unlike escalation, there is no partial
	// answer left to return.
	base := &stubBase{err: errors.New("sidecar down")}
	engine := New(base, &stubAdjudicator{}, policy(), 3.0, quietLogger())
	if _, err := engine.Detect(context.Background(), domain.Image{}); err == nil {
		t.Fatal("a base-engine failure was swallowed")
	}
}

func TestAdjudicatedVerdictsReplaceRatherThanDuplicate(t *testing.T) {
	base := &stubBase{result: domain.Result{
		Detections: []domain.Detection{det(0, 0.60, false)},
		Width:      100, Height: 100,
	}}
	adj := &stubAdjudicator{out: []domain.Detection{{
		Box: domain.NewBox(1, 1, 21, 21), Confidence: 0.9,
		IsChecked: true, Source: domain.EngineVLM,
	}}}
	engine := New(base, adj, policy(), 3.0, quietLogger())

	res, err := engine.Detect(context.Background(), domain.Image{})
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("a second opinion duplicated the detection instead of correcting it: %+v",
			res.Detections)
	}
	if !res.Detections[0].IsChecked || res.Detections[0].Source != domain.EngineVLM {
		t.Fatalf("the verdict was not applied: %+v", res.Detections[0])
	}
}

func TestSuppressionIsLeftToTheCaller(t *testing.T) {
	// The engine must not apply Policy.Apply itself: the HTTP layer applies it once to every
	// engine's output, which is the only reason engine accuracies are comparable at all.
	base := &stubBase{result: domain.Result{
		Detections: []domain.Detection{det(0, 0.10, false), det(40, 0.99, true)},
	}}
	engine := New(base, &stubAdjudicator{}, policy(), 3.0, quietLogger())

	res, err := engine.Detect(context.Background(), domain.Image{})
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(res.Detections) != 2 {
		t.Fatalf("the engine filtered its own output: %+v", res.Detections)
	}
}

func TestContextFactorDefaultsWhenNonPositive(t *testing.T) {
	base := &stubBase{result: domain.Result{Detections: []domain.Detection{det(0, 0.60, false)}}}
	adj := &stubAdjudicator{}
	engine := New(base, adj, policy(), 0, quietLogger())

	if _, err := engine.Detect(context.Background(), domain.Image{}); err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if adj.factor <= 0 {
		t.Fatalf("context factor = %v; a non-positive value must fall back to a usable default",
			adj.factor)
	}
}

func TestNameIsAssisted(t *testing.T) {
	engine := New(&stubBase{}, &stubAdjudicator{}, policy(), 3.0, quietLogger())
	if got := engine.Name(); got != domain.EngineAssisted {
		t.Fatalf("Name() = %q", got)
	}
}
