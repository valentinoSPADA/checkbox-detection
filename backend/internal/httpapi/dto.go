package httpapi

import "github.com/vspada/checkbox-detection/backend/internal/domain"

// The challenge specifies the response as exactly:
//
//	{"boxes": [{"bbox": [x1, y1, x2, y2], "is_checked": true}, ...]}
//
// That shape is honoured literally by default. Confidence, engine attribution and pipeline
// counters are genuinely useful -- the UI colours boxes by confidence, and the counters are
// what make a bad page diagnosable -- but adding keys to a specified contract risks breaking
// whatever reads it. They are therefore served only when ?verbose=true is passed, so the
// default response is byte-compatible with the specification and the richer view is opt-in.

// boxOut is the strict, specified form of one detection.
type boxOut struct {
	BBox      []int `json:"bbox"`
	IsChecked bool  `json:"is_checked"`
}

// verboseBoxOut adds diagnostics on top of the specified fields. The first two keys are
// identical in name, type and order to boxOut, so a client reading only those sees no
// difference.
type verboseBoxOut struct {
	BBox       []int             `json:"bbox"`
	IsChecked  bool              `json:"is_checked"`
	Confidence float64           `json:"confidence"`
	Source     domain.EngineName `json:"source"`
}

// detectOut is the specified response envelope.
type detectOut struct {
	Boxes []boxOut `json:"boxes"`
}

// verboseDetectOut is the opt-in envelope.
type verboseDetectOut struct {
	Boxes []verboseBoxOut `json:"boxes"`
	Meta  metaOut         `json:"meta"`
}

// metaOut reports how the answer was produced.
type metaOut struct {
	Engine    domain.EngineName `json:"engine"`
	Width     int               `json:"width"`
	Height    int               `json:"height"`
	ElapsedMS int64             `json:"elapsed_ms"`
	Stats     domain.Stats      `json:"stats"`
}

// errorOut is the single error shape used by every failing route, so that clients need one
// branch rather than one per endpoint.
type errorOut struct {
	Error   string `json:"error"`
	Details string `json:"details,omitempty"`
}

// toStrict converts detections into the specified response shape.
//
// Always allocates, so an empty result serialises as [] rather than null. A null there would
// crash any client that iterates without a nil check, and "no checkboxes on this page" is a
// perfectly ordinary outcome for, say, a photograph or a blank continuation sheet.
func toStrict(dets []domain.Detection) detectOut {
	boxes := make([]boxOut, 0, len(dets))
	for _, d := range dets {
		boxes = append(boxes, boxOut{
			BBox:      []int{d.Box.X1, d.Box.Y1, d.Box.X2, d.Box.Y2},
			IsChecked: d.IsChecked,
		})
	}
	return detectOut{Boxes: boxes}
}

// toVerbose converts detections into the diagnostic response shape.
func toVerbose(res domain.Result, elapsedMS int64) verboseDetectOut {
	boxes := make([]verboseBoxOut, 0, len(res.Detections))
	for _, d := range res.Detections {
		boxes = append(boxes, verboseBoxOut{
			BBox:       []int{d.Box.X1, d.Box.Y1, d.Box.X2, d.Box.Y2},
			IsChecked:  d.IsChecked,
			Confidence: round3(d.Confidence),
			Source:     d.Source,
		})
	}
	return verboseDetectOut{
		Boxes: boxes,
		Meta: metaOut{
			Engine:    res.Engine,
			Width:     res.Width,
			Height:    res.Height,
			ElapsedMS: elapsedMS,
			Stats:     res.Stats,
		},
	}
}

// round3 trims confidence to three decimals. Float noise in a response body is not precision,
// it is just diff churn in tests and logs.
func round3(v float64) float64 {
	return float64(int(v*1000+0.5)) / 1000
}
