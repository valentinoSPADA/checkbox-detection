package domain

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func det(x1, y1, x2, y2 int, conf float64, checked bool) Detection {
	return Detection{Box: Box{x1, y1, x2, y2}, Confidence: conf, IsChecked: checked, Source: EngineLocal}
}

func TestApplyDropsBelowThreshold(t *testing.T) {
	p := Policy{MinConfidence: 0.6, IoUThreshold: 0.3}
	got := p.Apply([]Detection{
		det(0, 0, 10, 10, 0.9, true),
		det(100, 100, 110, 110, 0.59, false),
		det(200, 200, 210, 210, 0.6, false), // exactly at the floor: inclusive
	})
	if len(got) != 2 {
		t.Fatalf("kept %d detections, want 2: %+v", len(got), got)
	}
}

func TestApplyDropsDegenerateGeometry(t *testing.T) {
	p := Policy{MinConfidence: 0.1, IoUThreshold: 0.3}
	got := p.Apply([]Detection{
		det(10, 10, 10, 10, 0.99, true), // zero area, high confidence
		det(0, 0, 20, 20, 0.5, false),
	})
	if len(got) != 1 || got[0].Box.Area() == 0 {
		t.Fatalf("degenerate box survived: %+v", got)
	}
}

func TestApplyReturnsEmptyNotNil(t *testing.T) {
	// The JSON response must serialise as [] rather than null; a null breaks any client
	// that iterates the field without a nil check.
	got := Policy{MinConfidence: 0.9, IoUThreshold: 0.3}.Apply([]Detection{det(0, 0, 10, 10, 0.1, false)})
	if got == nil {
		t.Fatal("Apply returned nil, want an empty slice")
	}
	if len(got) != 0 {
		t.Fatalf("Apply returned %d detections, want 0", len(got))
	}
}

func TestApplyDoesNotMutateInput(t *testing.T) {
	in := []Detection{det(0, 0, 10, 10, 0.9, true), det(2, 2, 12, 12, 0.8, false)}
	snapshot := append([]Detection(nil), in...)
	_ = Policy{MinConfidence: 0.5, IoUThreshold: 0.3}.Apply(in)
	for i := range in {
		if in[i] != snapshot[i] {
			t.Fatalf("Apply mutated its input at %d: %+v vs %+v", i, in[i], snapshot[i])
		}
	}
}

func TestSuppressKeepsHighestConfidence(t *testing.T) {
	p := Policy{MinConfidence: 0, IoUThreshold: 0.3}
	got := p.Suppress([]Detection{
		det(0, 0, 20, 20, 0.55, false),
		det(1, 1, 21, 21, 0.95, true), // heavy overlap, higher confidence
	})
	if len(got) != 1 {
		t.Fatalf("kept %d, want 1", len(got))
	}
	if got[0].Confidence != 0.95 || !got[0].IsChecked {
		t.Fatalf("suppression kept the weaker detection: %+v", got[0])
	}
}

func TestSuppressKeepsAdjacentCheckboxes(t *testing.T) {
	// Real forms place checkboxes close together. Suppression must not merge two genuinely
	// distinct boxes that happen to be near neighbours -- that would silently halve recall
	// on exactly the dense rows the samples are full of.
	p := Policy{MinConfidence: 0, IoUThreshold: 0.3}
	got := p.Suppress([]Detection{
		det(0, 0, 20, 20, 0.9, false),
		det(24, 0, 44, 20, 0.9, true),
		det(48, 0, 68, 20, 0.9, false),
	})
	if len(got) != 3 {
		t.Fatalf("kept %d adjacent boxes, want 3", len(got))
	}
}

func TestSuppressIsDeterministicOnTies(t *testing.T) {
	// Equal confidences must resolve the same way every run, or the evaluation harness
	// reports a different precision on identical input.
	p := Policy{MinConfidence: 0, IoUThreshold: 0.3}
	in := []Detection{
		det(0, 0, 30, 30, 0.8, false), // larger
		det(2, 2, 22, 22, 0.8, true),  // smaller: wins the tie-break
	}
	first := p.Suppress(in)
	for i := 0; i < 20; i++ {
		again := p.Suppress(in)
		if len(again) != len(first) || again[0].Box != first[0].Box {
			t.Fatalf("Suppress is not deterministic: %+v vs %+v", first, again)
		}
	}
	if first[0].Box.Area() != 400 {
		t.Fatalf("tie-break kept the larger box: %+v", first[0])
	}
}

func TestSuppressHandlesTrivialInputs(t *testing.T) {
	p := Policy{IoUThreshold: 0.3}
	if got := p.Suppress(nil); len(got) != 0 {
		t.Fatalf("Suppress(nil) = %+v", got)
	}
	one := []Detection{det(0, 0, 10, 10, 0.5, false)}
	if got := p.Suppress(one); len(got) != 1 {
		t.Fatalf("Suppress(single) = %+v", got)
	}
}

func TestMaxDetectionsTruncatesRanked(t *testing.T) {
	p := Policy{MinConfidence: 0, IoUThreshold: 0.3, MaxDetections: 2}
	got := p.Apply([]Detection{
		det(0, 0, 10, 10, 0.5, false),
		det(20, 0, 30, 10, 0.9, true),
		det(40, 0, 50, 10, 0.7, false),
	})
	if len(got) != 2 {
		t.Fatalf("kept %d, want 2", len(got))
	}
	// The cap must truncate the ranked list, so the highest-confidence detections survive.
	if got[0].Confidence != 0.9 || got[1].Confidence != 0.7 {
		t.Fatalf("cap truncated an unranked list: %+v", got)
	}
}

// escalationPolicy builds an explicit band so these tests exercise the selection logic
// rather than whatever the tuned defaults happen to be this week. DefaultPolicy's own
// values are asserted separately, in TestDefaultPolicyInvariants.
func escalationPolicy() Policy {
	return Policy{MinConfidence: 0.9, IoUThreshold: 0.3, EscalateAbove: 0.35,
		EscalateBelow: 0.85, MaxEscalations: 40}
}

func TestSelectForEscalationPicksOnlyTheBand(t *testing.T) {
	p := escalationPolicy() // band is (0.35, 0.85)
	got := p.SelectForEscalation([]Detection{
		det(0, 0, 10, 10, 0.99, true),   // confidently right: no information to gain
		det(20, 0, 30, 10, 0.10, false), // confidently negative: asking again is waste
		det(40, 0, 50, 10, 0.55, false), // uncertain
		det(60, 0, 70, 10, 0.80, true),  // uncertain
	})
	if len(got) != 2 {
		t.Fatalf("selected %d, want 2: %+v", len(got), got)
	}
	for _, d := range got {
		if d.Confidence <= p.EscalateAbove || d.Confidence >= p.EscalateBelow {
			t.Fatalf("selected a candidate outside the band: %+v", d)
		}
	}
}

func TestSelectForEscalationOrdersMostUncertainFirst(t *testing.T) {
	// Truncation at MaxEscalations must spend the budget on the least certain candidates,
	// not on whichever happened to be scanned first.
	p := escalationPolicy()
	p.MaxEscalations = 2
	mid := (p.EscalateAbove + p.EscalateBelow) / 2 // 0.60
	got := p.SelectForEscalation([]Detection{
		det(0, 0, 10, 10, 0.84, true),
		det(20, 0, 30, 10, 0.61, false), // closest to the midpoint
		det(40, 0, 50, 10, 0.58, false), // second closest
		det(60, 0, 70, 10, 0.36, true),
	})
	if len(got) != 2 {
		t.Fatalf("selected %d, want 2", len(got))
	}
	d0, d1 := abs(got[0].Confidence-mid), abs(got[1].Confidence-mid)
	if d0 > d1 {
		t.Fatalf("candidates not ordered by uncertainty: %v then %v", got[0].Confidence, got[1].Confidence)
	}
	if got[0].Confidence != 0.61 {
		t.Fatalf("most uncertain candidate not selected first: %+v", got)
	}
}

func TestSelectForEscalationDisabled(t *testing.T) {
	p := escalationPolicy()
	p.MaxEscalations = 0
	got := p.SelectForEscalation([]Detection{det(0, 0, 10, 10, 0.5, false)})
	if len(got) != 0 {
		t.Fatalf("escalation disabled but selected %d", len(got))
	}
	if got == nil {
		t.Fatal("want an empty slice, not nil")
	}
}

func TestSelectForEscalationSkipsDegenerate(t *testing.T) {
	p := escalationPolicy()
	got := p.SelectForEscalation([]Detection{det(10, 10, 10, 10, 0.5, false)})
	if len(got) != 0 {
		t.Fatalf("degenerate box selected for a paid call: %+v", got)
	}
}

func TestMergeReplacesOverlappingAndAppendsNew(t *testing.T) {
	p := Policy{IoUThreshold: 0.3}
	base := []Detection{
		det(0, 0, 20, 20, 0.5, false), // will be corrected
		det(50, 0, 70, 20, 0.9, true), // untouched
	}
	overrides := []Detection{
		{Box: Box{1, 1, 21, 21}, Confidence: 0.95, IsChecked: true, Source: EngineVLM},    // overlaps base[0]
		{Box: Box{100, 0, 120, 20}, Confidence: 0.9, IsChecked: false, Source: EngineVLM}, // new find
	}
	got := p.Merge(base, overrides)
	if len(got) != 3 {
		t.Fatalf("merged to %d detections, want 3: %+v", len(got), got)
	}
	if !got[0].IsChecked || got[0].Source != EngineVLM {
		t.Fatalf("override did not replace the base detection: %+v", got[0])
	}
	if got[1].Source != EngineLocal {
		t.Fatalf("non-overlapping base detection was altered: %+v", got[1])
	}
}

func TestMergeConsumesEachOverrideOnce(t *testing.T) {
	// Two base boxes overlapping one override must not both be replaced by it, or a single
	// second opinion would duplicate itself across neighbours.
	p := Policy{IoUThreshold: 0.1}
	base := []Detection{det(0, 0, 20, 20, 0.5, false), det(2, 2, 22, 22, 0.5, false)}
	overrides := []Detection{{Box: Box{1, 1, 21, 21}, Confidence: 0.95, IsChecked: true, Source: EngineVLM}}
	got := p.Merge(base, overrides)
	vlmCount := 0
	for _, d := range got {
		if d.Source == EngineVLM {
			vlmCount++
		}
	}
	if vlmCount != 1 {
		t.Fatalf("override applied %d times, want 1: %+v", vlmCount, got)
	}
}

func TestMergeWithNoOverridesIsIdentity(t *testing.T) {
	p := Policy{IoUThreshold: 0.3}
	base := []Detection{det(0, 0, 20, 20, 0.5, false)}
	got := p.Merge(base, nil)
	if len(got) != 1 || got[0] != base[0] {
		t.Fatalf("Merge with no overrides changed the base: %+v", got)
	}
}

func TestMergeIgnoresDegenerateOverrides(t *testing.T) {
	p := Policy{IoUThreshold: 0.3}
	base := []Detection{det(0, 0, 20, 20, 0.5, false)}
	got := p.Merge(base, []Detection{{Box: Box{5, 5, 5, 5}, Confidence: 0.99, Source: EngineVLM}})
	if len(got) != 1 || got[0].Source != EngineLocal {
		t.Fatalf("a degenerate override was applied: %+v", got)
	}
}

// TestDomainHasNoOutboundDependencies enforces the hexagonal boundary structurally rather
// than by convention. The claim "the domain is framework-agnostic and unit-testable in
// isolation" is only worth making if something fails when it stops being true; without this
// test, the first import of net/http into a policy file would pass review unnoticed.
func TestDomainHasNoOutboundDependencies(t *testing.T) {
	forbidden := []string{"net/http", "image", "database/sql", "os/exec",
		"github.com/anthropics", "github.com/vspada/checkbox-detection/backend/internal/httpapi",
		"github.com/vspada/checkbox-detection/backend/internal/detector",
		"github.com/vspada/checkbox-detection/backend/internal/imaging"}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("reading package directory: %v", err)
	}
	fset := token.NewFileSet()
	checked := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, filepath.Join(".", e.Name()), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("parsing %s: %v", e.Name(), err)
		}
		checked++
		for _, imp := range file.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			for _, bad := range forbidden {
				if path == bad || strings.HasPrefix(path, bad+"/") {
					t.Errorf("%s imports %q: the domain must not depend on adapters or I/O",
						e.Name(), path)
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("no domain source files were inspected; the guard is not actually running")
	}
}

// TestDefaultPolicyInvariants guards the shipped tuning rather than the logic.
//
// The escalation band must overlap the region *below* MinConfidence: if it sat entirely
// above the floor, escalation would only ever re-confirm detections that were already being
// returned, spending money to change nothing. That is a silent failure -- the assisted
// engine would still report itself as assisted -- so it is asserted rather than trusted.
func TestDefaultPolicyInvariants(t *testing.T) {
	p := DefaultPolicy()
	if p.MinConfidence <= 0 || p.MinConfidence >= 1 {
		t.Fatalf("MinConfidence %v is outside (0,1)", p.MinConfidence)
	}
	if p.EscalateAbove >= p.EscalateBelow {
		t.Fatalf("escalation band is empty: (%v, %v)", p.EscalateAbove, p.EscalateBelow)
	}
	if p.EscalateAbove >= p.MinConfidence {
		t.Fatalf("escalation band starts at %v, at or above the %v floor: escalation could "+
			"only re-confirm detections that are already returned",
			p.EscalateAbove, p.MinConfidence)
	}
	if p.IoUThreshold <= 0 || p.IoUThreshold >= 1 {
		t.Fatalf("IoUThreshold %v is outside (0,1)", p.IoUThreshold)
	}
}

func TestFloorForFallsBackToTheDefault(t *testing.T) {
	p := Policy{MinConfidence: 0.9, SourceMinConfidence: map[EngineName]float64{EngineVLM: 0.5}}
	if got := p.FloorFor(EngineVLM); got != 0.5 {
		t.Fatalf("FloorFor(vlm) = %v, want 0.5", got)
	}
	if got := p.FloorFor(EngineLocal); got != 0.9 {
		t.Fatalf("FloorFor(local) = %v, want the default 0.9", got)
	}
	// An engine added later must inherit the default bar, not escape thresholding entirely.
	if got := p.FloorFor(EngineName("future-engine")); got != 0.9 {
		t.Fatalf("FloorFor(unknown) = %v, want the default 0.9", got)
	}
}

// TestApplyUsesPerSourceFloors covers the bug this field was added for: under one shared
// floor of 0.95, a Claude verdict returned at 0.90 -- a strong signal from a stronger model --
// was discarded, so every escalated candidate was paid for and then thrown away.
func TestApplyUsesPerSourceFloors(t *testing.T) {
	p := Policy{
		MinConfidence:       0.95,
		IoUThreshold:        0.3,
		SourceMinConfidence: map[EngineName]float64{EngineVLM: 0.5},
	}
	got := p.Apply([]Detection{
		{Box: Box{0, 0, 20, 20}, Confidence: 0.90, Source: EngineVLM},     // kept: vlm floor
		{Box: Box{40, 0, 60, 20}, Confidence: 0.90, Source: EngineLocal},  // dropped: local floor
		{Box: Box{80, 0, 100, 20}, Confidence: 0.96, Source: EngineLocal}, // kept
	})
	if len(got) != 2 {
		t.Fatalf("kept %d, want 2: %+v", len(got), got)
	}
	sources := map[EngineName]bool{}
	for _, d := range got {
		sources[d.Source] = true
	}
	if !sources[EngineVLM] || !sources[EngineLocal] {
		t.Fatalf("expected one detection from each engine, got %+v", got)
	}
}

// TestDefaultPolicyEscalationCanActuallyPromote guards the whole point of the assisted
// engine. If the VLM floor were ever raised to the local floor, escalation would become a
// pure cost with no possible effect on the response -- and nothing else in the suite would
// notice, because the engine would still return a plausible answer.
func TestDefaultPolicyEscalationCanActuallyPromote(t *testing.T) {
	p := DefaultPolicy()
	// A candidate the local model is unsure about, which escalation should be able to rescue.
	uncertain := Detection{Box: Box{0, 0, 20, 20}, Confidence: 0.85, Source: EngineLocal}
	if len(p.SelectForEscalation([]Detection{uncertain})) != 1 {
		t.Fatal("a candidate inside the band was not selected for escalation")
	}
	// The same box, adjudicated by the vision model at its typical certainty, must survive.
	adjudicated := Detection{Box: Box{0, 0, 20, 20}, Confidence: 0.90, Source: EngineVLM}
	if len(p.Apply([]Detection{adjudicated})) != 1 {
		t.Fatalf("an adjudicated verdict at %v was dropped by the %v floor: escalation "+
			"cannot change any answer", adjudicated.Confidence, p.FloorFor(EngineVLM))
	}
}
