package domain

import "sort"

// Policy is the decision layer applied to raw candidates from any engine.
//
// It lives in Go rather than in the Python sidecar on purpose. Suppression, thresholding and
// escalation are business decisions with cost consequences -- escalation in particular spends
// real money per call -- and pushing them into the sidecar would leave the Go service a
// transparent proxy with no logic worth testing. Keeping them here also means the same rules
// apply identically to sidecar output and to language-model output, which is what makes the
// two engines comparable at all.
type Policy struct {
	// MinConfidence is the floor a candidate must clear to be returned.
	MinConfidence float64
	// IoUThreshold is the overlap above which the weaker of two candidates is suppressed.
	IoUThreshold float64
	// MaxDetections caps the response size; 0 means unlimited.
	MaxDetections int
	// EscalateBelow and EscalateAbove bound the uncertainty band whose members are worth a
	// second opinion from a more expensive engine.
	EscalateBelow float64
	EscalateAbove float64
	// MaxEscalations caps how many candidates a single request may escalate, bounding both
	// latency and spend.
	MaxEscalations int
}

// DefaultPolicy returns the tuned defaults used by the service.
//
// MinConfidence sits at 0.60 rather than 0.50 because the classifier is trained on synthetic
// data: its probabilities are well separated on clean inputs but drift toward the middle on
// real artefacts, so a floor just above the coin-flip line removes most of the domain-gap
// noise while costing few true positives. The escalation band deliberately starts *below*
// MinConfidence, so escalation can recover detections the floor would otherwise drop.
func DefaultPolicy() Policy {
	return Policy{
		MinConfidence:  0.60,
		IoUThreshold:   0.30,
		MaxDetections:  0,
		EscalateBelow:  0.85,
		EscalateAbove:  0.35,
		MaxEscalations: 40,
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
		if d.Box.Valid() && d.Confidence >= p.MinConfidence {
			kept = append(kept, d)
		}
	}
	kept = p.Suppress(kept)
	if p.MaxDetections > 0 && len(kept) > p.MaxDetections {
		kept = kept[:p.MaxDetections]
	}
	return kept
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

// SelectForEscalation picks the candidates worth a second opinion from a costlier engine.
//
// The rule is an uncertainty band, not a "worst N" list: a candidate the model is confidently
// wrong about will not be caught by asking again, and a candidate it is confidently right
// about is money wasted. Only the middle carries information. Candidates are returned
// ordered by ascending distance from the band's midpoint -- most uncertain first -- so that
// truncating at MaxEscalations spends the budget on the least certain cases rather than on
// whichever happened to be scanned first.
//
// Returns an empty slice when escalation is disabled (MaxEscalations <= 0) or when the band
// is empty, so callers can treat "no escalation" and "nothing uncertain" identically.
func (p Policy) SelectForEscalation(candidates []Detection) []Detection {
	if p.MaxEscalations <= 0 {
		return []Detection{}
	}
	band := make([]Detection, 0, len(candidates))
	for _, d := range candidates {
		if !d.Box.Valid() {
			continue
		}
		if d.Confidence > p.EscalateAbove && d.Confidence < p.EscalateBelow {
			band = append(band, d)
		}
	}
	mid := (p.EscalateAbove + p.EscalateBelow) / 2
	sort.SliceStable(band, func(i, j int) bool {
		return abs(band[i].Confidence-mid) < abs(band[j].Confidence-mid)
	})
	if len(band) > p.MaxEscalations {
		band = band[:p.MaxEscalations]
	}
	return band
}

// Merge folds a second engine's verdicts back over a base set of detections.
//
// Used by the assisted engine: overrides carry the escalated answers, base carries everything
// the local engine produced. An override replaces the base detection it overlaps with, rather
// than being appended, so that a second opinion corrects a box instead of duplicating it; an
// override that matches nothing is added, because the costlier engine is allowed to find what
// the cheaper one missed entirely.
//
// Overlap is decided by IoU against the same threshold used for suppression, so "these are
// the same checkbox" means one thing throughout the system.
func (p Policy) Merge(base, overrides []Detection) []Detection {
	if len(overrides) == 0 {
		return base
	}
	out := make([]Detection, 0, len(base)+len(overrides))
	consumed := make([]bool, len(overrides))
	for _, b := range base {
		replaced := false
		for i, o := range overrides {
			if consumed[i] || !o.Box.Valid() {
				continue
			}
			if b.Box.IoU(o.Box) > p.IoUThreshold {
				out = append(out, o)
				consumed[i] = true
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, b)
		}
	}
	for i, o := range overrides {
		if !consumed[i] && o.Box.Valid() {
			out = append(out, o)
		}
	}
	return out
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
