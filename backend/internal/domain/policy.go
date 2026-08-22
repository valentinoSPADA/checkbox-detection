package domain

import "sort"

// Policy is the decision layer applied to raw candidates from any engine.
//
// It lives in Go rather than in the Python sidecar on purpose. Which candidates survive, how
// overlaps are resolved and how many detections a caller receives are product decisions, not
// image processing; pushing them into the sidecar would leave the Go service a transparent
// proxy with no logic worth testing, and would put the rules that define the API's contract
// in the component least able to state them.
//
// It is also what keeps the Detector port meaningful: an engine returns evidence, and the
// same rules are applied to it whichever engine produced it.
type Policy struct {
	// MinConfidence is the floor a candidate must clear to be returned.
	MinConfidence float64
	// SourceMinConfidence overrides that floor for a named engine.
	//
	// Empty today: one engine ships, so one floor applies. It is kept because the reason it
	// exists is structural rather than incidental -- confidences from different producers are
	// not on the same scale, and the moment a second engine is registered, holding both to one
	// number is a category error. That was measured once already, with a shared 0.95 floor
	// silently discarding every verdict from an engine whose scores clustered near 0.9.
	SourceMinConfidence map[EngineName]float64
	// IoUThreshold is the overlap above which the weaker of two candidates is suppressed.
	IoUThreshold float64
	// MaxDetections caps the response size; 0 means unlimited.
	MaxDetections int
}

// DefaultPolicy returns the tuned defaults used by the service.
//
// MinConfidence is 0.90, and that number is measured rather than assumed. It is also a
// property of the trained model rather than of the problem, so it is re-swept whenever the
// classifier is retrained -- a threshold inherited from a previous model is a silent bug.
//
// Swept end to end against eval/ground_truth.json, scoring the live API:
//
//	0.70 -> P 0.871  R 0.743  F1 0.802
//	0.80 -> P 0.870  R 0.735  F1 0.797
//	0.90 -> P 0.930  R 0.703  F1 0.801   <- chosen
//	0.95 -> P 0.931  R 0.605  F1 0.733
//
// 0.70 and 0.90 tie on F1, and 0.90 wins on precision by six points at the same F1, so it is
// the better point on a flat stretch. The flatness is itself the result worth noting: the
// previous synthetic-only model swung from 0.884 precision at 0.95 to 0.533 at 0.90, which
// means its threshold was balanced on a knife edge. This one holds 0.93 across the plateau.
//
// The two populations separate by confidence rather than by geometry -- there is no size or
// shape test that divides them (see docs/prototype-log.md, where four were measured and
// removed). An early default of 0.60 sat below the noise floor and returned twelve times too
// many boxes; the model was fine, the threshold was not.
//
// Note the ceiling: label smoothing of 0.05 during training caps the softmax near 0.98, so a
// floor of 0.99 returns nothing at all. Anything above ~0.97 is off the usable scale.
func DefaultPolicy() Policy {
	return Policy{
		MinConfidence: 0.90,
		IoUThreshold:  0.30,
		MaxDetections: 0,
		// No per-source overrides while one engine ships. See the field's comment for why the
		// mechanism stays.
		SourceMinConfidence: nil,
	}
}

// Apply filters, deduplicates and orders candidates into a final answer.
//
// Order of operations is load-bearing. Invalid geometry is dropped first so that degenerate
// boxes cannot participate in suppression; the confidence floor is applied before NMS so a
// low-confidence box cannot suppress a high-confidence neighbour; and the cap is applied last
// so it truncates the ranked result rather than an arbitrary prefix.
//
// The input slice is not modified. Returns an empty, non-nil slice when nothing survives, so
// the JSON response serialises as [] rather than null -- a null there would break any client
// that iterates the field without a nil check.
func (p Policy) Apply(candidates []Detection) []Detection {
	kept := make([]Detection, 0, len(candidates))
	for _, d := range candidates {
		if d.Box.Valid() && d.Confidence >= p.FloorFor(d.Source) {
			kept = append(kept, d)
		}
	}
	kept = p.Suppress(kept)
	if p.MaxDetections > 0 && len(kept) > p.MaxDetections {
		kept = kept[:p.MaxDetections]
	}
	return kept
}

// FloorFor returns the confidence floor that applies to detections from one engine.
//
// Falls back to MinConfidence when no per-source override is configured, so a new engine is
// held to the default bar rather than to no bar at all.
func (p Policy) FloorFor(source EngineName) float64 {
	if floor, ok := p.SourceMinConfidence[source]; ok {
		return floor
	}
	return p.MinConfidence
}

// Suppress performs greedy non-maximum suppression, keeping the highest-confidence box of
// each overlapping cluster.
//
// Ties are broken by area (smaller wins) and then by position, so the function is
// deterministic: two runs over the same candidates always produce the same response. That
// matters more than it sounds -- without it, the evaluation harness would report different
// precision on identical input, and a flaky metric is worse than a low one.
//
// Complexity is O(n^2) in the worst case. That is acceptable because the sidecar has already
// collapsed tens of thousands of raw proposals to a few hundred candidates before this runs;
// if that ever stopped holding, the fix is spatial bucketing, not a different ranking rule.
func (p Policy) Suppress(candidates []Detection) []Detection {
	if len(candidates) < 2 {
		out := make([]Detection, len(candidates))
		copy(out, candidates)
		return out
	}
	ranked := make([]Detection, len(candidates))
	copy(ranked, candidates)
	sort.SliceStable(ranked, func(i, j int) bool {
		a, b := ranked[i], ranked[j]
		if a.Confidence != b.Confidence {
			return a.Confidence > b.Confidence
		}
		if a.Box.Area() != b.Box.Area() {
			return a.Box.Area() < b.Box.Area()
		}
		if a.Box.Y1 != b.Box.Y1 {
			return a.Box.Y1 < b.Box.Y1
		}
		return a.Box.X1 < b.Box.X1
	})

	kept := make([]Detection, 0, len(ranked))
	for _, cand := range ranked {
		overlapped := false
		for _, k := range kept {
			if cand.Box.IoU(k.Box) > p.IoUThreshold {
				overlapped = true
				break
			}
		}
		if !overlapped {
			kept = append(kept, cand)
		}
	}
	return kept
}
