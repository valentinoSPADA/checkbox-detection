package domain

import (
	"context"
	"errors"
	"fmt"
)

// EngineName identifies which Detector produced a result.
//
// Carried on every Detection rather than only on the response envelope. One engine ships
// today, so the field is constant in practice; it stays per-detection because a response that
// mixes producers is the case the Detector port exists to allow, and retrofitting provenance
// onto individual boxes after the fact means finding every place one was constructed.
type EngineName string

const (
	// EngineLocal is the two-stage CV + CNN pipeline running in the Python sidecar.
	EngineLocal EngineName = "local"
)

// ErrUnknownEngine is returned when a request names an engine that is not registered.
var ErrUnknownEngine = errors.New("unknown detection engine")

// ParseEngine resolves a request-supplied engine name, falling back to fallback when empty.
// Returns ErrUnknownEngine for anything unrecognised so the HTTP layer answers 400 rather
// than silently running a different engine than the caller asked for -- silent substitution
// would make the accuracy comparison in the writeup unreproducible.
func ParseEngine(s string, fallback EngineName) (EngineName, error) {
	switch EngineName(s) {
	case "":
		return fallback, nil
	case EngineLocal:
		return EngineLocal, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrUnknownEngine, s)
	}
}

// Detection is one classified checkbox.
//
// Confidence is the probability of the winning checkbox class -- that is, of "this is a
// checkbox and it is in the state reported" -- rather than of the box's existence alone.
// Keeping the two questions in one number is what lets a single threshold govern both
// detection and classification, and it is why the classifier was given a three-way head.
type Detection struct {
	Box        Box        `json:"bbox"`
	IsChecked  bool       `json:"is_checked"`
	Confidence float64    `json:"confidence"`
	Source     EngineName `json:"source"`
}

// Stats carries per-request counters through to the response for diagnostics.
//
// RawProposals and ScoredProposals expose the recall/precision funnel: a page where
// ScoredProposals is high but Returned is near zero points at the classifier, whereas a page
// where RawProposals is already near zero points at binarisation or at the size sweep. Without
// these two numbers a bad result is indistinguishable between the two stages.
type Stats struct {
	RawProposals    int `json:"raw_proposals,omitempty"`
	ScoredProposals int `json:"scored_proposals,omitempty"`
	Candidates      int `json:"candidates"`
	Returned        int `json:"returned"`
}

// Result is what a Detector returns for one page.
type Result struct {
	Detections []Detection
	Width      int
	Height     int
	Engine     EngineName
	Stats      Stats
}

// Image is an undecoded upload handed to a Detector.
//
// Deliberately raw bytes rather than a decoded image type: the domain must not depend on an
// image library, and both real adapters need the original encoded bytes anyway -- the
// sidecar re-uploads them and the VLM adapter base64-encodes them.
type Image struct {
	Data        []byte
	Filename    string
	ContentType string
}

// Detector is the port every detection strategy implements.
//
// Implementations must honour ctx cancellation, because the VLM adapter can block for tens of
// seconds on a slow model call and the HTTP layer bounds the whole request with a deadline.
// An implementation that ignores ctx would leak a goroutine per abandoned request.
type Detector interface {
	// Detect finds and classifies checkboxes in one page.
	Detect(ctx context.Context, img Image) (Result, error)
	// Name reports which engine this is, for response labelling and metrics.
	Name() EngineName
}
