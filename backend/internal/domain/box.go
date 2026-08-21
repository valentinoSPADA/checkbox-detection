// Package domain holds the core detection model and policy. It is framework-agnostic by
// construction: nothing in this package imports net/http, an image library, or any adapter.
// That constraint is what makes the detection policy unit-testable without a running
// sidecar, a model artifact, or a network, and it is enforced by a test in policy_test.go
// rather than left as a convention.
package domain

// Box is an axis-aligned bounding box in source-image pixel coordinates, expressed as the
// top-left and bottom-right corners exactly as the challenge's response schema requires.
//
// Coordinates are integers rather than floats because they index pixels of a specific
// decoded image; carrying sub-pixel precision here would imply an accuracy the upstream
// proposal grid does not have, and would force every consumer to decide how to round.
type Box struct {
	X1 int `json:"x1"`
	Y1 int `json:"y1"`
	X2 int `json:"x2"`
	Y2 int `json:"y2"`
}

// NewBox builds a Box from a corner pair, normalising the ordering so that X1 <= X2 and
// Y1 <= Y2. Adapters receive coordinates from a Python service and from a language model,
// and neither is guaranteed to order corners; normalising once at the boundary means no
// downstream geometry has to defend against inverted rectangles.
func NewBox(x1, y1, x2, y2 int) Box {
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if y1 > y2 {
		y1, y2 = y2, y1
	}
	return Box{X1: x1, Y1: y1, X2: x2, Y2: y2}
}

// Width returns the horizontal extent in pixels.
func (b Box) Width() int { return b.X2 - b.X1 }

// Height returns the vertical extent in pixels.
func (b Box) Height() int { return b.Y2 - b.Y1 }

// Area returns the pixel area, and zero for a degenerate box. Callers relying on this for
// ratios must still guard against zero: a zero-area box is possible whenever a model
// returns a collapsed rectangle.
func (b Box) Area() int {
	w, h := b.Width(), b.Height()
	if w <= 0 || h <= 0 {
		return 0
	}
	return w * h
}

// Valid reports whether the box has positive extent in both axes and non-negative origin.
// Used to reject malformed geometry arriving from an adapter before it reaches suppression,
// where a zero-area box would otherwise survive every IoU comparison (IoU is defined as 0
// against everything) and appear in the response as an invisible detection.
func (b Box) Valid() bool {
	return b.X1 >= 0 && b.Y1 >= 0 && b.Width() > 0 && b.Height() > 0
}

// Intersection returns the overlapping area of two boxes, or zero when they are disjoint.
func (b Box) Intersection(o Box) int {
	w := min(b.X2, o.X2) - max(b.X1, o.X1)
	h := min(b.Y2, o.Y2) - max(b.Y1, o.Y1)
	if w <= 0 || h <= 0 {
		return 0
	}
	return w * h
}

// IoU returns intersection-over-union in [0, 1].
//
// Returns 0 when either box is degenerate. That choice matters: it means a zero-area box
// never suppresses a real one during NMS, so a malformed adapter response degrades into an
// extra detection rather than silently deleting valid neighbours. Valid guards against the
// box surviving to the response.
func (b Box) IoU(o Box) float64 {
	inter := b.Intersection(o)
	if inter == 0 {
		return 0
	}
	union := b.Area() + o.Area() - inter
	if union <= 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// Clamp restricts the box to the given page bounds.
//
// Applied to language-model output in particular, which routinely returns coordinates a few
// pixels outside the image or scaled to a resized copy of it. Clamping rather than rejecting
// keeps an almost-correct detection instead of discarding it, which is the right trade when
// the alternative is losing a real checkbox at the page margin.
func (b Box) Clamp(width, height int) Box {
	return Box{
		X1: clampInt(b.X1, 0, width),
		Y1: clampInt(b.Y1, 0, height),
		X2: clampInt(b.X2, 0, width),
		Y2: clampInt(b.Y2, 0, height),
	}
}

// Scale multiplies the box by independent x and y factors and rounds to the nearest pixel.
//
// Needed when an adapter reasons about a resized copy of the page: the VLM adapter downsizes
// large scans before upload because the model's own preprocessing would otherwise do it
// invisibly, and the returned coordinates must be mapped back to the original resolution.
func (b Box) Scale(sx, sy float64) Box {
	return Box{
		X1: int(float64(b.X1)*sx + 0.5),
		Y1: int(float64(b.Y1)*sy + 0.5),
		X2: int(float64(b.X2)*sx + 0.5),
		Y2: int(float64(b.Y2)*sy + 0.5),
	}
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
