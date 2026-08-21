package domain

import "testing"

func TestNewBoxNormalisesInvertedCorners(t *testing.T) {
	// Both adapters can emit inverted corners: a language model has no ordering guarantee,
	// and the sidecar's coordinates pass through a scale step. Normalising at construction
	// is what lets every geometry function below assume x1 <= x2.
	got := NewBox(40, 90, 10, 20)
	want := Box{X1: 10, Y1: 20, X2: 40, Y2: 90}
	if got != want {
		t.Fatalf("NewBox(40,90,10,20) = %+v, want %+v", got, want)
	}
}

func TestBoxValid(t *testing.T) {
	cases := []struct {
		name string
		box  Box
		want bool
	}{
		{"ordinary", Box{10, 10, 30, 30}, true},
		{"zero width", Box{10, 10, 10, 30}, false},
		{"zero height", Box{10, 10, 30, 10}, false},
		{"negative origin", Box{-1, 10, 30, 30}, false},
		{"fully degenerate", Box{}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.box.Valid(); got != tc.want {
				t.Fatalf("Valid() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestBoxIoU(t *testing.T) {
	cases := []struct {
		name string
		a, b Box
		want float64
	}{
		{"identical", Box{0, 0, 10, 10}, Box{0, 0, 10, 10}, 1.0},
		{"disjoint", Box{0, 0, 10, 10}, Box{20, 20, 30, 30}, 0.0},
		{"touching edges only", Box{0, 0, 10, 10}, Box{10, 0, 20, 10}, 0.0},
		// 5x10 overlap over a union of 150 => 50/150.
		{"half overlap", Box{0, 0, 10, 10}, Box{5, 0, 15, 10}, 1.0 / 3.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.a.IoU(tc.b)
			if diff := got - tc.want; diff > 1e-9 || diff < -1e-9 {
				t.Fatalf("IoU = %v, want %v", got, tc.want)
			}
			if rev := tc.b.IoU(tc.a); rev != got {
				t.Fatalf("IoU is not symmetric: %v vs %v", got, rev)
			}
		})
	}
}

func TestBoxIoUDegenerateNeverSuppresses(t *testing.T) {
	// A zero-area box must score 0 against everything. If it scored 1 against a box it sits
	// inside, NMS would let a malformed adapter response delete a valid detection -- silently
	// losing a real checkbox rather than merely adding a bad one.
	degenerate := Box{10, 10, 10, 10}
	real := Box{0, 0, 20, 20}
	if got := degenerate.IoU(real); got != 0 {
		t.Fatalf("degenerate IoU = %v, want 0", got)
	}
}

func TestBoxClamp(t *testing.T) {
	got := Box{-5, -5, 120, 200}.Clamp(100, 150)
	want := Box{0, 0, 100, 150}
	if got != want {
		t.Fatalf("Clamp = %+v, want %+v", got, want)
	}
}

func TestBoxScaleRoundsToNearest(t *testing.T) {
	// 3 * 1.5 = 4.5 must round to 5, not truncate to 4: truncation accumulates a systematic
	// leftward/upward bias across a page of boxes mapped back from a downscaled copy.
	got := Box{3, 3, 7, 7}.Scale(1.5, 1.5)
	want := Box{5, 5, 11, 11}
	if got != want {
		t.Fatalf("Scale = %+v, want %+v", got, want)
	}
}

func TestBoxScaleIdentity(t *testing.T) {
	b := Box{10, 20, 30, 40}
	if got := b.Scale(1, 1); got != b {
		t.Fatalf("Scale(1,1) = %+v, want %+v", got, b)
	}
}

func TestBoxAreaAndDimensions(t *testing.T) {
	b := Box{10, 20, 40, 60}
	if b.Width() != 30 || b.Height() != 40 {
		t.Fatalf("Width/Height = %d/%d, want 30/40", b.Width(), b.Height())
	}
	if b.Area() != 1200 {
		t.Fatalf("Area = %d, want 1200", b.Area())
	}
	if got := (Box{10, 10, 10, 20}).Area(); got != 0 {
		t.Fatalf("degenerate Area = %d, want 0", got)
	}
}

func TestParseEngine(t *testing.T) {
	cases := []struct {
		in      string
		want    EngineName
		wantErr bool
	}{
		{"", EngineLocal, false},
		{"local", EngineLocal, false},
		{"vlm", EngineVLM, false},
		{"assisted", EngineAssisted, false},
		{"nonsense", "", true},
		{"LOCAL", "", true}, // case-sensitive on purpose: silent coercion hides typos
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got, err := ParseEngine(tc.in, EngineLocal)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ParseEngine(%q) expected an error", tc.in)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseEngine(%q) unexpected error: %v", tc.in, err)
			}
			if got != tc.want {
				t.Fatalf("ParseEngine(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}
