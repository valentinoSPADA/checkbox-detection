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
	// SourceMinConfidence overrides that floor per producing engine.
	//
	// Confidences from different engines are not on the same scale, and treating them as if
	// they were is a category error with a concrete cost. The local classifier is a
	// synthetic-trained softmax whose useful signal lives above 0.95; Claude reports a
	// self-assessed certainty that clusters near 0.9 even when it is sure. Under one shared
	// floor of 0.95, every escalated verdict was silently discarded -- the assisted engine
	// paid for model calls that could not change the answer by construction.
	SourceMinConfidence map[EngineName]float64
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
	//
	// Measured on the four samples, the number of candidates inside the uncertainty band is
	// 57, 136, 146 and 448. A flat cap therefore allocates help in inverse proportion to how
	// much a page needs it: at 40 the clean page got 70% coverage and the watermarked page --
	// the one with the worst recall -- got 9%, which is why assisted output was nearly
	// identical to local. The cap is still flat, because per-page adaptive budgeting is a
	// spend-control policy that belongs to whoever pays the bill, but it is now set where it
	// covers most pages rather than almost none.
	MaxEscalations int
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
//
// The escalation band deliberately spans the region just below the floor, so escalation can
// recover detections the floor would otherwise drop rather than merely re-confirming ones it
// already keeps.
func DefaultPolicy() Policy {
	return Policy{
		MinConfidence:  0.90,
		IoUThreshold:   0.30,
		MaxDetections:  0,
		EscalateBelow:  0.92,
		EscalateAbove:  0.70,
		MaxEscalations: 120,
		SourceMinConfidence: map[EngineName]float64{
			// Claude's self-reported certainty is not the local model's softmax and must
			// not be held to the same bar; a verdict it returns at 0.9 is a stronger signal
			// than the local classifier at 0.9, not a weaker one.
			EngineVLM: 0.50,
		},
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
