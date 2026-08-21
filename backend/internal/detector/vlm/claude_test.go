package vlm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/vspada/checkbox-detection/backend/internal/domain"
)

// fakeAPI stands in for the Anthropic Messages endpoint. Pointing the SDK at it via
// Config.BaseURL is what makes the tiling, the coordinate mapping and the partial-failure
// behaviour testable at all -- otherwise every run of this suite would cost money, which in
// practice means it would never run.
func fakeAPI(t *testing.T, handler http.HandlerFunc) string {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv.URL
}

// toolUseResponse builds a Messages API response carrying a forced tool call.
func toolUseResponse(name string, input any) []byte {
	raw, _ := json.Marshal(input)
	body := map[string]any{
		"id": "msg_test", "type": "message", "role": "assistant",
		"model": "claude-haiku-4-5", "stop_reason": "tool_use",
		"content": []map[string]any{
			{"type": "tool_use", "id": "tu_1", "name": name, "input": json.RawMessage(raw)},
		},
		"usage": map[string]int{"input_tokens": 10, "output_tokens": 10},
	}
	out, _ := json.Marshal(body)
	return out
}

func textOnlyResponse() []byte {
	body := map[string]any{
		"id": "msg_test", "type": "message", "role": "assistant",
		"model": "claude-haiku-4-5", "stop_reason": "end_turn",
		"content": []map[string]any{{"type": "text", "text": "I could not find any checkboxes."}},
		"usage":   map[string]int{"input_tokens": 10, "output_tokens": 10},
	}
	out, _ := json.Marshal(body)
	return out
}

func pngImage(t *testing.T, w, h int) domain.Image {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: 250, G: 250, B: 250, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encoding fixture: %v", err)
	}
	return domain.Image{Data: buf.Bytes(), Filename: "page.png", ContentType: "image/png"}
}

func TestNewRequiresAPIKey(t *testing.T) {
	// The wiring layer relies on this to decline registering the engine, rather than
	// exposing one that fails on its first request.
	if _, err := New(Config{}); err == nil {
		t.Fatal("a client was built with no API key")
	}
}

func TestNewAppliesDefaults(t *testing.T) {
	c, err := New(Config{APIKey: "sk-test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if c.tileRows != DefaultTileRows || c.tileCols != DefaultTileCols {
		t.Errorf("tile grid = %dx%d", c.tileRows, c.tileCols)
	}
	if c.model == "" || c.maxTokens == 0 || c.concurrency == 0 {
		t.Errorf("zero-valued config left a field unset: %+v", c)
	}
}

func TestNameIsVLM(t *testing.T) {
	c, _ := New(Config{APIKey: "sk-test"})
	if got := c.Name(); got != domain.EngineVLM {
		t.Fatalf("Name() = %q", got)
	}
}

func TestDetectMapsTileCoordinatesToThePage(t *testing.T) {
	// Every tile reports the same local box. If tile offsets were dropped, all detections
	// would pile up in the top-left corner -- a bug that produces plausible output and is
	// invisible without this assertion.
	var calls atomic.Int32
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse(toolName, map[string]any{
			"boxes": []map[string]any{
				{"x1": 5, "y1": 5, "x2": 25, "y2": 25, "is_checked": true, "confidence": 0.9},
			},
		}))
	})

	client, err := New(Config{APIKey: "sk-test", BaseURL: url, TileRows: 2, TileCols: 2,
		TileOverlap: 0})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := client.Detect(context.Background(), pngImage(t, 400, 400))
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if calls.Load() != 4 {
		t.Fatalf("made %d model calls, want one per tile (4)", calls.Load())
	}
	if len(res.Detections) != 4 {
		t.Fatalf("got %d detections, want 4", len(res.Detections))
	}
	origins := map[image.Point]bool{}
	for _, d := range res.Detections {
		origins[image.Pt(d.Box.X1, d.Box.Y1)] = true
		if d.Source != domain.EngineVLM {
			t.Errorf("detection missing source attribution: %+v", d)
		}
	}
	if len(origins) != 4 {
		t.Fatalf("tile offsets were not applied; detections collapsed to %d distinct positions",
			len(origins))
	}
	if res.Width != 400 || res.Height != 400 {
		t.Errorf("page size = %dx%d, want the ORIGINAL 400x400", res.Width, res.Height)
	}
}

func TestDetectSurvivesPartialTileFailure(t *testing.T) {
	// Losing one eighth of a page to a rate limit is a far better outcome than losing all
	// of it, so a failing tile must not cancel its siblings.
	var seen atomic.Int32
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if seen.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse(toolName, map[string]any{
			"boxes": []map[string]any{
				{"x1": 1, "y1": 1, "x2": 21, "y2": 21, "is_checked": false, "confidence": 0.8},
			},
		}))
	})

	client, _ := New(Config{APIKey: "sk-test", BaseURL: url, TileRows: 2, TileCols: 2,
		TileOverlap: 0, Concurrency: 1})
	res, err := client.Detect(context.Background(), pngImage(t, 200, 200))
	if err != nil {
		t.Fatalf("one failing tile failed the whole page: %v", err)
	}
	if len(res.Detections) == 0 {
		t.Fatal("no detections survived a single tile failure")
	}
}

func TestDetectFailsWhenEveryTileFails(t *testing.T) {
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"down"}}`))
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url, TileRows: 1, TileCols: 2,
		TileOverlap: 0, Concurrency: 1})
	if _, err := client.Detect(context.Background(), pngImage(t, 200, 200)); err == nil {
		t.Fatal("a total upstream outage was reported as success")
	}
}

func TestDetectRejectsUndecodableImage(t *testing.T) {
	client, _ := New(Config{APIKey: "sk-test", BaseURL: "http://127.0.0.1:0"})
	_, err := client.Detect(context.Background(), domain.Image{Data: []byte("not an image")})
	if err == nil {
		t.Fatal("garbage bytes were accepted")
	}
}

func TestDetectDropsDegenerateBoxes(t *testing.T) {
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse(toolName, map[string]any{
			"boxes": []map[string]any{
				{"x1": 10, "y1": 10, "x2": 10, "y2": 10, "is_checked": true, "confidence": 0.9},
				{"x1": 5, "y1": 5, "x2": 25, "y2": 25, "is_checked": true, "confidence": 0.9},
			},
		}))
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url, TileRows: 1, TileCols: 1})
	res, err := client.Detect(context.Background(), pngImage(t, 200, 200))
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	for _, d := range res.Detections {
		if !d.Box.Valid() {
			t.Fatalf("a zero-area box reached the response: %+v", d.Box)
		}
	}
}

func TestDetectDefaultsMissingConfidence(t *testing.T) {
	// Models are inconsistent about supplying confidence even when the schema requires it;
	// a missing score must not silently delete an otherwise good detection at the threshold.
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse(toolName, map[string]any{
			"boxes": []map[string]any{
				{"x1": 5, "y1": 5, "x2": 25, "y2": 25, "is_checked": true, "confidence": 0},
			},
		}))
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url, TileRows: 1, TileCols: 1})
	res, _ := client.Detect(context.Background(), pngImage(t, 200, 200))
	if len(res.Detections) != 1 {
		t.Fatalf("got %d detections, want 1", len(res.Detections))
	}
	if res.Detections[0].Confidence <= 0 || res.Detections[0].Confidence > 1 {
		t.Fatalf("confidence = %v, want a usable default", res.Detections[0].Confidence)
	}
}

func TestDetectErrorsWhenModelSkipsTheTool(t *testing.T) {
	// Silently falling back to scraping prose would mask a real regression behind
	// slightly-worse numbers, which is the failure mode most worth avoiding here.
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(textOnlyResponse())
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url, TileRows: 1, TileCols: 1})
	_, err := client.Detect(context.Background(), pngImage(t, 200, 200))
	if !errors.Is(err, ErrNoStructuredOutput) {
		t.Fatalf("err = %v, want ErrNoStructuredOutput", err)
	}
}

func TestAdjudicateAppliesVerdicts(t *testing.T) {
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse("report_verdicts", map[string]any{
			"verdicts": []map[string]any{
				{"index": 0, "verdict": "checked", "confidence": 0.93},
				{"index": 1, "verdict": "not_a_checkbox", "confidence": 0.88},
				// index 2 deliberately omitted: absence of a verdict is not a verdict.
			},
		}))
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url})

	in := []domain.Detection{
		{Box: domain.NewBox(10, 10, 30, 30), Confidence: 0.5, Source: domain.EngineLocal},
		{Box: domain.NewBox(50, 10, 70, 30), Confidence: 0.5, Source: domain.EngineLocal},
		{Box: domain.NewBox(90, 10, 110, 30), Confidence: 0.5, IsChecked: true,
			Source: domain.EngineLocal},
	}
	out, err := client.Adjudicate(context.Background(), pngImage(t, 200, 200), in, 3.0)
	if err != nil {
		t.Fatalf("Adjudicate: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("got %d verdicts, want the input length", len(out))
	}
	if !out[0].IsChecked || out[0].Source != domain.EngineVLM {
		t.Errorf("checked verdict not applied: %+v", out[0])
	}
	if out[1].Confidence != 0 {
		t.Errorf("not_a_checkbox should zero the confidence so the existing threshold drops "+
			"it, got %v", out[1].Confidence)
	}
	if out[2].Source != domain.EngineLocal || !out[2].IsChecked {
		t.Errorf("an unaddressed candidate was altered: %+v", out[2])
	}
}

func TestAdjudicateNoCandidatesMakesNoCall(t *testing.T) {
	var calls atomic.Int32
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url})
	out, err := client.Adjudicate(context.Background(), pngImage(t, 50, 50), nil, 3.0)
	if err != nil || out != nil {
		t.Fatalf("out=%v err=%v, want nil,nil", out, err)
	}
	if calls.Load() != 0 {
		t.Fatal("a paid call was made with nothing to adjudicate")
	}
}

func TestAdjudicateIgnoresOutOfRangeIndices(t *testing.T) {
	// A model returning an index outside the batch must not panic or corrupt a neighbour.
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse("report_verdicts", map[string]any{
			"verdicts": []map[string]any{
				{"index": 99, "verdict": "checked", "confidence": 0.9},
				{"index": -1, "verdict": "checked", "confidence": 0.9},
			},
		}))
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url})
	in := []domain.Detection{{Box: domain.NewBox(0, 0, 20, 20), Confidence: 0.5,
		Source: domain.EngineLocal}}
	out, err := client.Adjudicate(context.Background(), pngImage(t, 100, 100), in, 3.0)
	if err != nil {
		t.Fatalf("Adjudicate: %v", err)
	}
	if len(out) != 1 || out[0].Source != domain.EngineLocal {
		t.Fatalf("an out-of-range index corrupted the result: %+v", out)
	}
}

func TestCropAroundStaysInsideTheImage(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for _, box := range []domain.Box{
		domain.NewBox(0, 0, 10, 10),     // top-left corner
		domain.NewBox(90, 90, 100, 100), // bottom-right corner
		domain.NewBox(45, 45, 55, 55),   // centre
	} {
		out := cropAround(src, box, 3.0)
		if out.Bounds().Dx() < 0 || out.Bounds().Dy() < 0 {
			t.Fatalf("negative crop for %v: %v", box, out.Bounds())
		}
	}
}

func TestCropAroundDefaultsBadFactor(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 100, 100))
	if out := cropAround(src, domain.NewBox(40, 40, 60, 60), 0); out.Bounds().Dx() == 0 {
		t.Fatal("a non-positive context factor produced an empty crop")
	}
}

func TestPromptsAreEmbedded(t *testing.T) {
	// go:embed failures surface at compile time, but an empty file would not -- and an empty
	// prompt is a silently much worse detector.
	for name, prompt := range map[string]string{
		"detect": detectPrompt, "adjudicate": adjudicatePrompt,
	} {
		if len(prompt) < 200 {
			t.Errorf("%s prompt is %d bytes; it looks empty or truncated", name, len(prompt))
		}
	}
}

func TestToolSchemasAreWellFormed(t *testing.T) {
	for _, tool := range []struct {
		name string
		u    any
	}{{"boxes", boxTool()}, {"verdicts", verdictTool()}} {
		raw, err := json.Marshal(tool.u)
		if err != nil {
			t.Fatalf("%s tool does not marshal: %v", tool.name, err)
		}
		if !bytes.Contains(raw, []byte("additionalProperties")) {
			t.Errorf("%s tool schema is not strict: %s", tool.name, raw)
		}
	}
}

func TestHelperDefaults(t *testing.T) {
	if got := orString("", "fallback"); got != "fallback" {
		t.Errorf("orString = %q", got)
	}
	if got := orString("set", "fallback"); got != "set" {
		t.Errorf("orString = %q", got)
	}
	if got := orInt(0, 7); got != 7 {
		t.Errorf("orInt = %d", got)
	}
	if got := orFloat(0, 1.5); got != 1.5 {
		t.Errorf("orFloat = %v", got)
	}
	if got := clamp01(0, 0.9); got != 0.9 {
		t.Errorf("clamp01(0) = %v, want the default", got)
	}
	if got := clamp01(1.7, 0.9); got != 0.9 {
		t.Errorf("clamp01(out of range) = %v, want the default", got)
	}
	if got := clamp01(0.42, 0.9); got != 0.42 {
		t.Errorf("clamp01(valid) = %v", got)
	}
}

func TestAdjudicateChunksIntoBatches(t *testing.T) {
	// A single message asking for hundreds of verdicts overruns max_tokens and returns a
	// truncated, unparseable tool call -- losing the whole page's adjudication instead of one
	// batch of it. Chunking is what makes a high escalation cap usable at all.
	var calls atomic.Int32
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse("report_verdicts", map[string]any{
			"verdicts": []map[string]any{{"index": 0, "verdict": "checked", "confidence": 0.9}},
		}))
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url, BatchSize: 5, Concurrency: 2})

	in := make([]domain.Detection, 23)
	for i := range in {
		in[i] = domain.Detection{Box: domain.NewBox(i*30, 0, i*30+20, 20),
			Confidence: 0.5, Source: domain.EngineLocal}
	}
	out, err := client.Adjudicate(context.Background(), pngImage(t, 800, 100), in, 3.0)
	if err != nil {
		t.Fatalf("Adjudicate: %v", err)
	}
	if calls.Load() != 5 { // ceil(23/5)
		t.Fatalf("made %d calls for 23 candidates at batch size 5, want 5", calls.Load())
	}
	if len(out) != 23 {
		t.Fatalf("got %d results, want the input length", len(out))
	}
}

func TestAdjudicateOffsetsIndicesPerBatch(t *testing.T) {
	// Verdict indices are local to their batch. If the offset were dropped, every batch's
	// verdicts would be applied to the first few candidates -- a bug that still returns a
	// plausible-looking result.
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse("report_verdicts", map[string]any{
			"verdicts": []map[string]any{{"index": 0, "verdict": "checked", "confidence": 0.9}},
		}))
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url, BatchSize: 2, Concurrency: 1})

	in := make([]domain.Detection, 6)
	for i := range in {
		in[i] = domain.Detection{Box: domain.NewBox(i*30, 0, i*30+20, 20),
			Confidence: 0.5, Source: domain.EngineLocal}
	}
	out, err := client.Adjudicate(context.Background(), pngImage(t, 400, 100), in, 3.0)
	if err != nil {
		t.Fatalf("Adjudicate: %v", err)
	}
	// Index 0 of each batch of 2 => candidates 0, 2 and 4 judged; 1, 3 and 5 untouched.
	for i, d := range out {
		wantJudged := i%2 == 0
		judged := d.Source == domain.EngineVLM
		if judged != wantJudged {
			t.Fatalf("candidate %d judged=%v, want %v -- batch offsets are wrong", i, judged, wantJudged)
		}
	}
}

func TestAdjudicateSurvivesPartialBatchFailure(t *testing.T) {
	var seen atomic.Int32
	url := fakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if seen.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolUseResponse("report_verdicts", map[string]any{
			"verdicts": []map[string]any{{"index": 0, "verdict": "checked", "confidence": 0.9}},
		}))
	})
	client, _ := New(Config{APIKey: "sk-test", BaseURL: url, BatchSize: 2, Concurrency: 1})

	in := make([]domain.Detection, 6)
	for i := range in {
		in[i] = domain.Detection{Box: domain.NewBox(i*30, 0, i*30+20, 20),
			Confidence: 0.5, Source: domain.EngineLocal}
	}
	out, err := client.Adjudicate(context.Background(), pngImage(t, 400, 100), in, 3.0)
	if err != nil {
		t.Fatalf("one failing batch failed the whole adjudication: %v", err)
	}
	// The failed batch leaves its candidates with the verdict they already had, which beats
	// discarding every batch that succeeded.
	judged := 0
	for _, d := range out {
		if d.Source == domain.EngineVLM {
			judged++
		}
	}
	if judged == 0 {
		t.Fatal("no verdicts survived a single batch failure")
	}
}
