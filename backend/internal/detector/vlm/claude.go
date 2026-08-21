// Package vlm adapts Anthropic's Claude vision models to the domain.Detector port.
//
// It exists to answer a question the local engine cannot: the local pipeline is trained on
// synthetic crops and knows nothing about a form it has never seen a shape of, whereas a
// general vision model reasons about what a checkbox *is*. The trade is stark -- roughly
// three orders of magnitude more latency and a real per-page cost -- which is why this
// adapter is a selectable strategy rather than the default, and why the assisted engine
// exists to spend it only where it changes an answer.
package vlm

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"golang.org/x/sync/errgroup"

	"github.com/vspada/checkbox-detection/backend/internal/domain"
	"github.com/vspada/checkbox-detection/backend/internal/imaging"
)

// ErrNoStructuredOutput reports that the model answered without the forced tool call.
//
// Treated as a hard error rather than being papered over with a text-parsing fallback: if
// the model stops honouring the tool contract, silently switching to regex-scraping its prose
// would mask a real regression behind slightly-worse numbers, which is the failure mode worth
// avoiding most in an AI integration.
var ErrNoStructuredOutput = errors.New("model returned no structured tool call")

const toolName = "report_checkboxes"

// Defaults chosen for detection quality rather than cost; every one is overridable.
const (
	// DefaultTileRows/Cols split a page before sending it. A full URAR page downscaled to
	// the model's ~1568 px working width leaves checkboxes about 14 px across, which is
	// below what any vision model localises reliably. Tiling restores usable scale.
	DefaultTileRows = 4
	DefaultTileCols = 2
	// DefaultTileOverlap keeps boxes on a seam whole in at least one tile.
	DefaultTileOverlap = 0.06
	// DefaultTileMaxDim bounds each tile after cropping.
	DefaultTileMaxDim = 1400
	// DefaultConcurrency caps in-flight model calls per request, bounding both burst spend
	// and the chance of tripping a rate limit on a single large page.
	DefaultConcurrency = 4
	// DefaultMaxTokens is generous because a dense tile can legitimately hold 60+ boxes and
	// a truncated tool call is unparseable -- the whole tile is lost, not just its tail.
	DefaultMaxTokens = 8000
)

// Client calls a Claude vision model and adapts its answers into domain detections.
type Client struct {
	api         anthropic.Client
	model       string
	maxImageDim int
	tileRows    int
	tileCols    int
	tileOverlap float64
	concurrency int
	maxTokens   int64
}

// Config configures a Client. Zero-valued fields fall back to the Default* constants.
type Config struct {
	APIKey      string
	Model       string
	MaxImageDim int
	TileRows    int
	TileCols    int
	TileOverlap float64
	Concurrency int
	MaxTokens   int64
	// BaseURL overrides the Anthropic API endpoint.
	//
	// Exists so this adapter can be tested against a stub server. Without it the tiling,
	// the tile-to-page coordinate mapping and the partial-failure behaviour would all be
	// unreachable by any test that does not spend money on every run -- which in practice
	// means untested.
	BaseURL string
}

// New builds a Client. Returns an error when no API key is available, so that the wiring
// layer can decline to register the VLM engines rather than exposing engines that would fail
// on first use.
func New(cfg Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("vlm: ANTHROPIC_API_KEY is required")
	}
	opts := []option.RequestOption{option.WithAPIKey(cfg.APIKey)}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	c := &Client{
		api:         anthropic.NewClient(opts...),
		model:       orString(cfg.Model, "claude-haiku-4-5"),
		maxImageDim: orInt(cfg.MaxImageDim, 1568),
		tileRows:    orInt(cfg.TileRows, DefaultTileRows),
		tileCols:    orInt(cfg.TileCols, DefaultTileCols),
		tileOverlap: orFloat(cfg.TileOverlap, DefaultTileOverlap),
		concurrency: orInt(cfg.Concurrency, DefaultConcurrency),
		maxTokens:   int64(orInt(int(cfg.MaxTokens), DefaultMaxTokens)),
	}
	return c, nil
}

// Name identifies this adapter in responses and metrics.
func (c *Client) Name() domain.EngineName { return domain.EngineVLM }

// boxTool is the schema the model is forced to fill.
//
// Coordinates are demanded in the tile's own pixel space rather than normalised to 0-1
// because the model is markedly better at reading pixel positions off an image it can see
// the dimensions of than at producing fractions, and because normalised values would have to
// be de-normalised against a size the model only estimated.
func boxTool() anthropic.ToolUnionParam {
	t := anthropic.ToolParam{
		Name: toolName,
		Description: anthropic.String(
			"Report every checkbox found in the image, with its pixel bounding box and whether it is filled."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"boxes": map[string]any{
					"type":        "array",
					"description": "One entry per checkbox. Empty array if the image contains none.",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"x1":         map[string]any{"type": "integer", "description": "left edge in pixels"},
							"y1":         map[string]any{"type": "integer", "description": "top edge in pixels"},
							"x2":         map[string]any{"type": "integer", "description": "right edge in pixels"},
							"y2":         map[string]any{"type": "integer", "description": "bottom edge in pixels"},
							"is_checked": map[string]any{"type": "boolean", "description": "true if the box carries any mark"},
							"confidence": map[string]any{"type": "number", "description": "0..1 certainty in this entry"},
						},
						"required":             []string{"x1", "y1", "x2", "y2", "is_checked", "confidence"},
						"additionalProperties": false,
					},
				},
			},
			Required:    []string{"boxes"},
			ExtraFields: map[string]any{"additionalProperties": false},
		},
	}
	return anthropic.ToolUnionParam{OfTool: &t}
}

// Prompts live in their own files rather than as Go string constants because the offline
// annotation tool in detector/training reads these exact bytes to label training data. Two
// copies of a prompt drift the moment either is tuned, and a drifted annotation prompt
// silently teaches the local model a slightly different notion of "checkbox" than the one
// the runtime enforces -- a bug with no stack trace.
//
//go:embed prompts/detect.txt
var detectPrompt string

//go:embed prompts/adjudicate.txt
var adjudicatePrompt string

// wireBoxes matches the tool's input schema.
type wireBoxes struct {
	Boxes []struct {
		X1         int     `json:"x1"`
		Y1         int     `json:"y1"`
		X2         int     `json:"x2"`
		Y2         int     `json:"y2"`
		IsChecked  bool    `json:"is_checked"`
		Confidence float64 `json:"confidence"`
	} `json:"boxes"`
}

// Detect runs tiled vision detection over one page.
//
// Tiles are processed concurrently, bounded by the configured concurrency. A tile that fails
// does NOT fail the page: its error is recorded and the remaining tiles' detections are
// returned, because losing one eighth of a page is a far better outcome than losing all of
// it to a single rate-limit response. If every tile fails, the first error is returned.
//
// Coordinates are mapped tile -> downscaled page -> original page before being returned, and
// clamped to the page bounds. Suppression of the duplicates that tile overlap produces is
// left to domain.Policy, which the caller applies.
func (c *Client) Detect(ctx context.Context, img domain.Image) (domain.Result, error) {
	src, err := imaging.Decode(img.Data)
	if err != nil {
		return domain.Result{}, err
	}
	origW, origH := src.Bounds().Dx(), src.Bounds().Dy()

	fitted, scaleX, scaleY := imaging.FitWithin(src, c.maxImageDim*max(c.tileRows, c.tileCols))
	tiles := imaging.Split(fitted, c.tileRows, c.tileCols, c.tileOverlap)

	var (
		mu       sync.Mutex
		all      []domain.Detection
		firstErr error
		failures int
	)

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(c.concurrency)
	for _, tile := range tiles {
		tile := tile
		g.Go(func() error {
			dets, err := c.detectTile(gctx, tile)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				failures++
				if firstErr == nil {
					firstErr = fmt.Errorf("tile %d: %w", tile.Index, err)
				}
				// Swallowed on purpose: returning it would cancel the errgroup and
				// discard the tiles that did succeed.
				return nil
			}
			all = append(all, dets...)
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return domain.Result{}, err
	}
	if failures == len(tiles) && firstErr != nil {
		return domain.Result{}, firstErr
	}

	out := make([]domain.Detection, 0, len(all))
	for _, d := range all {
		box := d.Box.Scale(scaleX, scaleY).Clamp(origW, origH)
		if !box.Valid() {
			continue
		}
		d.Box = box
		out = append(out, d)
	}

	return domain.Result{
		Detections: out,
		Width:      origW,
		Height:     origH,
		Engine:     domain.EngineVLM,
		Stats:      domain.Stats{Candidates: len(out)},
	}, nil
}

// detectTile issues one model call and returns detections in the parent image's coordinates.
func (c *Client) detectTile(ctx context.Context, tile imaging.Tile) ([]domain.Detection, error) {
	shrunk, sx, sy := imaging.FitWithin(tile.Image, c.maxImageDim)
	encoded, err := imaging.Encode(shrunk)
	if err != nil {
		return nil, err
	}

	parsed, err := c.callTool(ctx, encoded, detectPrompt)
	if err != nil {
		return nil, err
	}

	dets := make([]domain.Detection, 0, len(parsed.Boxes))
	for _, b := range parsed.Boxes {
		box := domain.NewBox(b.X1, b.Y1, b.X2, b.Y2).Scale(sx, sy)
		box = domain.Box{
			X1: box.X1 + tile.OffsetX, Y1: box.Y1 + tile.OffsetY,
			X2: box.X2 + tile.OffsetX, Y2: box.Y2 + tile.OffsetY,
		}
		if !box.Valid() {
			continue
		}
		conf := b.Confidence
		if conf <= 0 || conf > 1 {
			// Models are inconsistent about supplying confidence even when the schema
			// requires it. A neutral-but-passing default is used so a missing score does
			// not silently delete an otherwise good detection at the policy threshold.
			conf = 0.75
		}
		dets = append(dets, domain.Detection{
			Box: box, IsChecked: b.IsChecked, Confidence: conf, Source: domain.EngineVLM,
		})
	}
	return dets, nil
}

// callTool sends one image with a forced tool call and returns the decoded tool input.
func (c *Client) callTool(ctx context.Context, png []byte, prompt string) (wireBoxes, error) {
	msg, err := c.api.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: c.maxTokens,
		Tools:     []anthropic.ToolUnionParam{boxTool()},
		// Forcing the tool removes the "model answers in prose" failure mode entirely,
		// which matters more here than the flexibility auto tool choice would give: there
		// is exactly one thing this call is for.
		ToolChoice: anthropic.ToolChoiceParamOfTool(toolName),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewImageBlock(anthropic.Base64ImageSourceParam{
					Data:      base64.StdEncoding.EncodeToString(png),
					MediaType: anthropic.Base64ImageSourceMediaTypeImagePNG,
				}),
				anthropic.NewTextBlock(prompt),
			),
		},
	})
	if err != nil {
		return wireBoxes{}, fmt.Errorf("anthropic call: %w", err)
	}

	for _, block := range msg.Content {
		if use, ok := block.AsAny().(anthropic.ToolUseBlock); ok && use.Name == toolName {
			var parsed wireBoxes
			// Input is raw JSON; unmarshalling rather than string-matching is required
			// because escaping of the serialised input is not stable across models.
			if err := json.Unmarshal([]byte(use.JSON.Input.Raw()), &parsed); err != nil {
				return wireBoxes{}, fmt.Errorf("decoding tool input: %w", err)
			}
			return parsed, nil
		}
	}
	return wireBoxes{}, ErrNoStructuredOutput
}

// Adjudicate asks the model to re-judge specific candidate regions.
//
// Used by the assisted engine. Each candidate is cropped with surrounding context and sent
// as a numbered image in a single message, which is what makes escalation affordable: forty
// uncertain candidates cost one call rather than forty. The model answers per index, and
// indices it omits are simply left unchanged by the caller rather than being deleted --
// absence of a verdict is not a verdict.
//
// The returned slice is aligned with candidates by index; entries the model did not address
// carry the original detection untouched.
func (c *Client) Adjudicate(ctx context.Context, img domain.Image, candidates []domain.Detection,
	contextFactor float64) ([]domain.Detection, error) {
	if len(candidates) == 0 {
		return nil, nil
	}
	src, err := imaging.Decode(img.Data)
	if err != nil {
		return nil, err
	}

	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(candidates)*2+1)
	blocks = append(blocks, anthropic.NewTextBlock(adjudicatePrompt))
	for i, cand := range candidates {
		crop := cropAround(src, cand.Box, contextFactor)
		encoded, err := imaging.Encode(crop)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks,
			anthropic.NewTextBlock(fmt.Sprintf("Region %d:", i)),
			anthropic.NewImageBlock(anthropic.Base64ImageSourceParam{
				Data:      base64.StdEncoding.EncodeToString(encoded),
				MediaType: anthropic.Base64ImageSourceMediaTypeImagePNG,
			}))
	}

	verdicts, err := c.callVerdicts(ctx, blocks)
	if err != nil {
		return nil, err
	}

	out := make([]domain.Detection, len(candidates))
	copy(out, candidates)
	for _, v := range verdicts {
		if v.Index < 0 || v.Index >= len(out) {
			continue
		}
		switch v.Verdict {
		case "not_a_checkbox":
			// Zeroing the confidence lets the existing policy threshold drop it, instead
			// of introducing a second deletion path that could disagree with the first.
			out[v.Index].Confidence = 0
		case "checked", "unchecked":
			out[v.Index].IsChecked = v.Verdict == "checked"
			out[v.Index].Confidence = clamp01(v.Confidence, 0.9)
		default:
			continue
		}
		out[v.Index].Source = domain.EngineVLM
	}
	return out, nil
}

type verdict struct {
	Index      int     `json:"index"`
	Verdict    string  `json:"verdict"`
	Confidence float64 `json:"confidence"`
}

func verdictTool() anthropic.ToolUnionParam {
	t := anthropic.ToolParam{
		Name:        "report_verdicts",
		Description: anthropic.String("Report one verdict per numbered region."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"verdicts": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"index":      map[string]any{"type": "integer"},
							"verdict":    map[string]any{"type": "string", "enum": []string{"checked", "unchecked", "not_a_checkbox"}},
							"confidence": map[string]any{"type": "number"},
						},
						"required":             []string{"index", "verdict", "confidence"},
						"additionalProperties": false,
					},
				},
			},
			Required:    []string{"verdicts"},
			ExtraFields: map[string]any{"additionalProperties": false},
		},
	}
	return anthropic.ToolUnionParam{OfTool: &t}
}

func (c *Client) callVerdicts(ctx context.Context, blocks []anthropic.ContentBlockParamUnion) ([]verdict, error) {
	msg, err := c.api.Messages.New(ctx, anthropic.MessageNewParams{
		Model:      anthropic.Model(c.model),
		MaxTokens:  c.maxTokens,
		Tools:      []anthropic.ToolUnionParam{verdictTool()},
		ToolChoice: anthropic.ToolChoiceParamOfTool("report_verdicts"),
		Messages:   []anthropic.MessageParam{anthropic.NewUserMessage(blocks...)},
	})
	if err != nil {
		return nil, fmt.Errorf("anthropic call: %w", err)
	}
	for _, block := range msg.Content {
		if use, ok := block.AsAny().(anthropic.ToolUseBlock); ok && use.Name == "report_verdicts" {
			var parsed struct {
				Verdicts []verdict `json:"verdicts"`
			}
			if err := json.Unmarshal([]byte(use.JSON.Input.Raw()), &parsed); err != nil {
				return nil, fmt.Errorf("decoding tool input: %w", err)
			}
			return parsed.Verdicts, nil
		}
	}
	return nil, ErrNoStructuredOutput
}

// cropAround cuts a padded region centred on box, so the model can see whether the candidate
// sits in whitespace or inside a table -- the context that decides most of these calls.
func cropAround(src image.Image, box domain.Box, factor float64) image.Image {
	if factor < 1 {
		factor = 3.0
	}
	cx, cy := (box.X1+box.X2)/2, (box.Y1+box.Y2)/2
	half := int(float64(max(box.Width(), box.Height())) * factor / 2)
	if half < 12 {
		half = 12
	}
	return imaging.Crop(src, image.Rect(cx-half, cy-half, cx+half, cy+half))
}

func clamp01(v, def float64) float64 {
	if v <= 0 || v > 1 {
		return def
	}
	return v
}

func orString(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

func orInt(v, def int) int {
	if v == 0 {
		return def
	}
	return v
}

func orFloat(v, def float64) float64 {
	if v == 0 {
		return def
	}
	return v
}
