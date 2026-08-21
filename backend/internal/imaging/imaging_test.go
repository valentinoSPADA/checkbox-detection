package imaging

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"testing"
)

func testImage(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// A gradient, so resampling errors are visible rather than hidden by flat colour.
			img.Set(x, y, color.RGBA{R: uint8(x % 256), G: uint8(y % 256), B: 128, A: 255})
		}
	}
	return img
}

func encodePNG(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encoding fixture: %v", err)
	}
	return buf.Bytes()
}

func TestDecodeAcceptsPNGAndJPEG(t *testing.T) {
	png := encodePNG(t, testImage(20, 10))
	got, err := Decode(png)
	if err != nil {
		t.Fatalf("Decode(png): %v", err)
	}
	if got.Bounds().Dx() != 20 || got.Bounds().Dy() != 10 {
		t.Errorf("png bounds = %v", got.Bounds())
	}

	var jbuf bytes.Buffer
	if err := jpeg.Encode(&jbuf, testImage(30, 15), nil); err != nil {
		t.Fatalf("encoding jpeg fixture: %v", err)
	}
	if _, err := Decode(jbuf.Bytes()); err != nil {
		t.Fatalf("Decode(jpeg): %v", err)
	}
}

func TestDecodeRejectsNonImage(t *testing.T) {
	if _, err := Decode([]byte("definitely not an image")); err == nil {
		t.Fatal("garbage bytes were accepted as an image")
	}
}

func TestDecodeRejectsTruncatedImage(t *testing.T) {
	full := encodePNG(t, testImage(40, 40))
	if _, err := Decode(full[:len(full)/2]); err == nil {
		t.Fatal("a truncated PNG was accepted")
	}
}

func TestEncodeRoundTrips(t *testing.T) {
	data, err := Encode(testImage(12, 8))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	back, err := Decode(data)
	if err != nil {
		t.Fatalf("Decode after Encode: %v", err)
	}
	if back.Bounds().Dx() != 12 || back.Bounds().Dy() != 8 {
		t.Errorf("round-trip bounds = %v", back.Bounds())
	}
}

func TestFitWithinLeavesSmallImagesAlone(t *testing.T) {
	src := testImage(100, 50)
	out, sx, sy := FitWithin(src, 200)
	if out != src {
		t.Error("an image that already fits was resampled anyway")
	}
	if sx != 1 || sy != 1 {
		t.Errorf("scale = %v,%v, want 1,1", sx, sy)
	}
}

func TestFitWithinDisabledByNonPositiveMax(t *testing.T) {
	src := testImage(100, 50)
	if out, _, _ := FitWithin(src, 0); out != src {
		t.Error("maxDim of 0 should disable resizing")
	}
}

// TestFitWithinScaleFactorsMapCoordinatesHome is the property the whole VLM path depends on:
// a box found in the downscaled copy is wrong by exactly this ratio in the original, and
// getting it wrong produces plausible-looking boxes in the wrong places.
func TestFitWithinScaleFactorsMapCoordinatesHome(t *testing.T) {
	src := testImage(1000, 500)
	out, sx, sy := FitWithin(src, 100)

	if out.Bounds().Dx() > 100 || out.Bounds().Dy() > 100 {
		t.Fatalf("resized bounds %v exceed maxDim", out.Bounds())
	}
	// A point at the far edge of the resized image must map back to the far edge of the source.
	gotX := float64(out.Bounds().Dx()) * sx
	if gotX < 990 || gotX > 1010 {
		t.Errorf("x scale maps the right edge to %v, want ~1000", gotX)
	}
	gotY := float64(out.Bounds().Dy()) * sy
	if gotY < 495 || gotY > 505 {
		t.Errorf("y scale maps the bottom edge to %v, want ~500", gotY)
	}
}

func TestFitWithinPreservesAspectRatio(t *testing.T) {
	out, _, _ := FitWithin(testImage(800, 200), 100)
	ratio := float64(out.Bounds().Dx()) / float64(out.Bounds().Dy())
	if ratio < 3.5 || ratio > 4.5 {
		t.Fatalf("aspect ratio = %v, want ~4", ratio)
	}
}

func TestSplitSingleTileWhenGridIsTrivial(t *testing.T) {
	tiles := Split(testImage(100, 100), 1, 1, 0.1)
	if len(tiles) != 1 {
		t.Fatalf("got %d tiles, want 1", len(tiles))
	}
	if tiles[0].OffsetX != 0 || tiles[0].OffsetY != 0 {
		t.Errorf("single tile carries a non-zero offset: %+v", tiles[0])
	}
}

func TestSplitProducesTheRequestedGrid(t *testing.T) {
	tiles := Split(testImage(400, 300), 3, 2, 0.0)
	if len(tiles) != 6 {
		t.Fatalf("got %d tiles, want 6", len(tiles))
	}
	for i, tile := range tiles {
		if tile.Index != i {
			t.Errorf("tile %d has index %d", i, tile.Index)
		}
		if tile.Image.Bounds().Dx() <= 0 || tile.Image.Bounds().Dy() <= 0 {
			t.Errorf("tile %d is empty", i)
		}
	}
}

// TestSplitOverlapWidensTiles guards the reason overlap exists: without it a checkbox
// straddling a seam is unrecognisable in both halves, so every seam becomes a band of
// guaranteed misses.
func TestSplitOverlapWidensTiles(t *testing.T) {
	plain := Split(testImage(400, 400), 2, 2, 0.0)
	lapped := Split(testImage(400, 400), 2, 2, 0.25)
	if len(plain) != len(lapped) {
		t.Fatalf("tile counts differ: %d vs %d", len(plain), len(lapped))
	}
	if lapped[0].Image.Bounds().Dx() <= plain[0].Image.Bounds().Dx() {
		t.Fatalf("overlap did not widen the tile: %v vs %v",
			lapped[0].Image.Bounds(), plain[0].Image.Bounds())
	}
}

func TestSplitOffsetsCoverTheImage(t *testing.T) {
	tiles := Split(testImage(400, 400), 2, 2, 0.0)
	seenX := map[int]bool{}
	seenY := map[int]bool{}
	for _, tile := range tiles {
		seenX[tile.OffsetX] = true
		seenY[tile.OffsetY] = true
	}
	if len(seenX) != 2 || len(seenY) != 2 {
		t.Fatalf("offsets do not tile the image: x=%v y=%v", seenX, seenY)
	}
}

func TestSplitNegativeOverlapIsTreatedAsZero(t *testing.T) {
	if tiles := Split(testImage(200, 200), 2, 2, -0.5); len(tiles) != 4 {
		t.Fatalf("got %d tiles with a negative overlap, want 4", len(tiles))
	}
}

func TestCropClampsToBounds(t *testing.T) {
	// A crop request running off the page must yield a smaller image, never panic.
	out := Crop(testImage(100, 100), image.Rect(-20, -20, 50, 50))
	if out.Bounds().Dx() != 50 || out.Bounds().Dy() != 50 {
		t.Fatalf("clamped crop bounds = %v, want 50x50", out.Bounds())
	}
}

func TestCropFullyOutsideIsEmptyNotPanicking(t *testing.T) {
	out := Crop(testImage(100, 100), image.Rect(500, 500, 600, 600))
	if out.Bounds().Dx() != 0 || out.Bounds().Dy() != 0 {
		t.Fatalf("out-of-bounds crop = %v, want empty", out.Bounds())
	}
}

func TestCropCopiesRatherThanAliasing(t *testing.T) {
	// Tiles are encoded concurrently; sharing the parent's pixel buffer would make any
	// future in-place operation a data race.
	src := image.NewRGBA(image.Rect(0, 0, 10, 10))
	src.Set(1, 1, color.RGBA{R: 255, A: 255})
	out := Crop(src, image.Rect(0, 0, 5, 5))

	src.Set(1, 1, color.RGBA{B: 255, A: 255})
	r, _, _, _ := out.At(1, 1).RGBA()
	if r == 0 {
		t.Fatal("the crop aliases the source buffer")
	}
}
