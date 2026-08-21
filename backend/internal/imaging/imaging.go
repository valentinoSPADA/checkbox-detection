// Package imaging holds the pixel manipulation the VLM adapter needs: decoding, resizing,
// tiling and re-encoding.
//
// It is an adapter-side concern, not a domain one. The domain deals in boxes and
// confidences and must not learn what a pixel is; this package exists so that the one
// adapter which genuinely needs to touch images can, without that dependency leaking inward.
package imaging

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"  // registered for decoding only
	_ "image/jpeg" // registered for decoding only
	"image/png"

	xdraw "golang.org/x/image/draw"
)

// PNG is the wire format used when handing images to the model.
//
// PNG rather than JPEG because the entire signal here is hairline rules one or two pixels
// wide. JPEG's chroma subsampling and ringing blur exactly those, and a checkbox border that
// survives the eye at quality 90 can still be smeared enough to change the model's answer.
// The size penalty is real but bounded, since pages are downscaled first.
const PNG = "image/png"

// Decode parses encoded image bytes. Supported formats are PNG, JPEG and GIF; TIFF is
// deliberately not registered, because the model endpoint would reject it anyway and failing
// here produces a clearer message than failing at the API boundary.
func Decode(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decoding image: %w", err)
	}
	return img, nil
}

// Encode renders an image to PNG bytes.
func Encode(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, fmt.Errorf("encoding png: %w", err)
	}
	return buf.Bytes(), nil
}

// FitWithin scales img down so neither side exceeds maxDim, preserving aspect ratio.
//
// Returns the image unchanged when it already fits or when maxDim <= 0, and reports the
// scale factors needed to map coordinates in the returned image back to the original.
// Callers must apply those factors: a box found in a downscaled copy is wrong by exactly
// that ratio in the source, and this is the single easiest place in the whole system to
// silently produce plausible-looking but misplaced boxes.
//
// Uses Catmull-Rom rather than nearest-neighbour because a one-pixel rule vanishes entirely
// under point sampling, which would delete the very feature being detected.
func FitWithin(img image.Image, maxDim int) (out image.Image, scaleX, scaleY float64) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if maxDim <= 0 || (w <= maxDim && h <= maxDim) {
		return img, 1, 1
	}
	ratio := float64(maxDim) / float64(max(w, h))
	nw, nh := max(1, int(float64(w)*ratio)), max(1, int(float64(h)*ratio))
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, b, xdraw.Over, nil)
	return dst, float64(w) / float64(nw), float64(h) / float64(nh)
}

// Tile is one piece of a page, carrying the offset needed to map its coordinates home.
type Tile struct {
	Image image.Image
	// OffsetX and OffsetY locate the tile's top-left corner in the parent image.
	OffsetX int
	OffsetY int
	// Index is the tile's position in the split, used for logging and error attribution.
	Index int
}

// Split cuts an image into a rows x cols grid with a fractional overlap between neighbours.
//
// Overlap exists to stop the grid from bisecting checkboxes. A box cut in half by a tile
// edge is unrecognisable in both halves, so without overlap every seam becomes a band of
// guaranteed misses; with it, any box smaller than the overlap appears whole in at least one
// tile. The duplicates this creates along seams are removed later by suppression, which is
// the cheaper problem to have.
//
// Returns a single full-image tile when rows and cols are both <= 1.
func Split(img image.Image, rows, cols int, overlap float64) []Tile {
	b := img.Bounds()
	if rows <= 1 && cols <= 1 {
		return []Tile{{Image: img, OffsetX: b.Min.X, OffsetY: b.Min.Y, Index: 0}}
	}
	rows, cols = max(1, rows), max(1, cols)
	if overlap < 0 {
		overlap = 0
	}

	w, h := b.Dx(), b.Dy()
	stepX, stepY := w/cols, h/rows
	padX, padY := int(float64(stepX)*overlap), int(float64(stepY)*overlap)

	tiles := make([]Tile, 0, rows*cols)
	idx := 0
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			x0 := clamp(b.Min.X+c*stepX-padX, b.Min.X, b.Max.X)
			y0 := clamp(b.Min.Y+r*stepY-padY, b.Min.Y, b.Max.Y)
			x1 := clamp(b.Min.X+(c+1)*stepX+padX, b.Min.X, b.Max.X)
			y1 := clamp(b.Min.Y+(r+1)*stepY+padY, b.Min.Y, b.Max.Y)
			if x1-x0 < 2 || y1-y0 < 2 {
				continue
			}
			tiles = append(tiles, Tile{
				Image:   crop(img, image.Rect(x0, y0, x1, y1)),
				OffsetX: x0 - b.Min.X,
				OffsetY: y0 - b.Min.Y,
				Index:   idx,
			})
			idx++
		}
	}
	return tiles
}

// Crop returns the sub-image described by r, clamped to the source bounds.
//
// Always copies rather than returning a view. A view would share the parent's pixel buffer,
// and the tiles are encoded concurrently by several goroutines; sharing the buffer is safe
// for reads but makes any future in-place operation a data race waiting to happen.
func Crop(img image.Image, r image.Rectangle) image.Image {
	return crop(img, r.Intersect(img.Bounds()))
}

func crop(img image.Image, r image.Rectangle) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(dst, dst.Bounds(), img, r.Min, draw.Src)
	return dst
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Outline draws a hollow rectangle of the given width onto img, in place where possible.
//
// Exists for one reason: a crop cut around a checkbox on a dense form contains the candidate's
// NEIGHBOURS, and a model asked "is this checked?" will happily answer about the wrong box.
// Measured on sample_1, that mistake accounted for every single disagreement between the
// adjudicator and the pixels -- fourteen boxes with exactly 0.0% interior ink reported as
// checked, each one sitting directly above or below a marked box. Words alone did not fix it
// (the prompt already said to ignore the edges); a visible marker does, because it removes the
// ambiguity instead of asking the model to resolve it.
//
// Returns an RGBA copy when img is not already drawable, so callers need not care which. The
// source must be bounded; an unbounded image such as image.Uniform is not a meaningful input.
func Outline(img image.Image, r image.Rectangle, col color.Color, width int) image.Image {
	if width < 1 {
		width = 1
	}
	dst, ok := img.(draw.Image)
	if !ok {
		b := img.Bounds()
		rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(rgba, rgba.Bounds(), img, b.Min, draw.Src)
		dst = rgba
	}
	r = r.Intersect(dst.Bounds())
	if r.Empty() {
		return dst
	}
	src := image.NewUniform(col)
	// Four bands rather than a stroked path: the edges are clipped independently, so a marker
	// on a candidate touching the crop border degrades to the sides that fit instead of vanishing.
	for _, band := range []image.Rectangle{
		image.Rect(r.Min.X, r.Min.Y, r.Max.X, r.Min.Y+width),
		image.Rect(r.Min.X, r.Max.Y-width, r.Max.X, r.Max.Y),
		image.Rect(r.Min.X, r.Min.Y, r.Min.X+width, r.Max.Y),
		image.Rect(r.Max.X-width, r.Min.Y, r.Max.X, r.Max.Y),
	} {
		draw.Draw(dst, band.Intersect(dst.Bounds()), src, image.Point{}, draw.Src)
	}
	return dst
}
