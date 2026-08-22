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
	return Policy{MinConfidence: 0.9, IoUThreshold: 0.3}
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
// Both bounds are exclusive on purpose. A MinConfidence of exactly 1 returns nothing on every
// page -- and label smoothing already caps this model's softmax near 0.98, so it is reachable
// by a plausible typo rather than an absurd one. An IoUThreshold of 0 suppresses every box
// that touches another, which on a form whose checkboxes sit in adjacent cells deletes most
// of the answer.
func TestDefaultPolicyInvariants(t *testing.T) {
	p := DefaultPolicy()
	if p.MinConfidence <= 0 || p.MinConfidence >= 1 {
		t.Fatalf("MinConfidence %v is outside (0,1)", p.MinConfidence)
	}
	if p.IoUThreshold <= 0 || p.IoUThreshold >= 1 {
		t.Fatalf("IoUThreshold %v is outside (0,1)", p.IoUThreshold)
	}
}

// TestFloorForFallsBackToTheDefault covers a mechanism the shipped policy does not currently
// use: with one engine registered, SourceMinConfidence is nil. It is tested anyway, because
// the day a second engine is added is the day someone discovers whether per-source floors
// work -- and the last time two engines shared one floor, every verdict from the second was
// silently discarded.
func TestFloorForFallsBackToTheDefault(t *testing.T) {
	const other = EngineName("second-engine")
	p := Policy{MinConfidence: 0.9, SourceMinConfidence: map[EngineName]float64{other: 0.5}}
	if got := p.FloorFor(other); got != 0.5 {
		t.Fatalf("FloorFor(second-engine) = %v, want 0.5", got)
	}
	if got := p.FloorFor(EngineLocal); got != 0.9 {
		t.Fatalf("FloorFor(local) = %v, want the default 0.9", got)
	}
	// An engine with no entry must inherit the default bar, not escape thresholding entirely.
	if got := p.FloorFor(EngineName("future-engine")); got != 0.9 {
		t.Fatalf("FloorFor(unknown) = %v, want the default 0.9", got)
	}
}
