"""Procedural training-data generator for the Stage 2 checkbox classifier.

The challenge ships four images and no annotations, so a training set has to be
manufactured. The value of synthesis here is not merely volume: it lets every failure
mode observed on the real samples be produced *deliberately and labelled correctly*,
instead of being hoped for. Each of these is generated on purpose:

* checkboxes that share a border with a table rule (the failure that broke the
  connected-component prototype);
* boxes on shaded backgrounds, reproducing sample 3's blue table rows;
* a page-wide tinted wash, reproducing sample 4's red watermark;
* freehand marks -- check marks, scribbles, dots, strokes that spill outside the box --
  alongside the clean printed `X`, because sample 2 contains all of them;
* letter counters (`o`, `a`, `8`, `D`, `B`) rendered from real system fonts at real small
  sizes, because those are what every geometric prototype kept mistaking for boxes;
* the inside of the black section rail that runs down the left edge of every sample, set with
  white vertical lettering -- a region where every pixel is ink, so Stage 1 nominates
  rectangles by the hundred and a model that has never seen a dark crop accepts them.

Crops are rendered at a random *native* pixel size and only then resized to the model's
input, so the model learns the same resolution loss it will meet at inference: a 12 px
checkbox upscaled to 40 px is blurry in a specific way, and training on crisp renders
would leave that unlearned.

Labels: 0 = not_a_checkbox, 1 = unchecked, 2 = checked.
"""

from __future__ import annotations

import itertools
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np

try:  # Pillow is used only for glyph negatives; the generator degrades without it.
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:  # pragma: no cover - exercised only on a stripped environment
    _HAS_PIL = False

LABEL_NEGATIVE = 0
LABEL_UNCHECKED = 1
LABEL_CHECKED = 2
NUM_CLASSES = 3

INPUT_SIZE = 40
CONTEXT = 1.35  # must match preprocess.crop_with_context

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]

# Glyphs whose enclosed counters are square-ish at small sizes. These are the exact shapes
# the geometric prototypes false-positived on, so they are over-represented as negatives.
_COUNTER_GLYPHS = "oaepbdg08BDOQRP@#&"


def _available_fonts() -> list[str]:
    return [p for p in _FONT_CANDIDATES if os.path.exists(p)]


@dataclass
class GenConfig:
    """Knobs for the generator, surfaced so training runs are reproducible and tunable."""

    min_side: int = 10
    max_side: int = 64
    seed: int = 1234


class SyntheticGenerator:
    """Draws labelled crops that mimic what Stage 1 hands to Stage 2 at inference time."""

    def __init__(self, cfg: GenConfig | None = None) -> None:
        self.cfg = cfg or GenConfig()
        self.rng = random.Random(self.cfg.seed)
        self.np_rng = np.random.default_rng(self.cfg.seed)
        self.fonts = _available_fonts()

    # ---------------------------------------------------------------- helpers

    def _paper(self, size: int) -> np.ndarray:
        """A background patch: white, grey, blue-shaded or watermark-tinted, with gradient.

        Reproduces sample 3 (blue table rows) and sample 4 (red watermark wash). Returned
        as single-channel because the whole pipeline is grayscale; the tint is applied as
        the luminance a coloured wash would produce.
        """
        base = self.rng.choice([255, 255, 255, 248, 240, 232, 224, 210])
        img = np.full((size, size), base, np.float32)
        if self.rng.random() < 0.35:  # soft illumination gradient, as in a scan
            gx = np.linspace(self.rng.uniform(-14, 14), self.rng.uniform(-14, 14), size)
            gy = np.linspace(self.rng.uniform(-14, 14), self.rng.uniform(-14, 14), size)
            img += gx[None, :] + gy[:, None]
        return np.clip(img, 0, 255)

    def _ink(self) -> float:
        """A stroke darkness. Not always 0: printed rules on a scan are grey, not black."""
        return float(self.rng.choice([0, 0, 10, 25, 40, 60, 85]))

    def _rules(self, img: np.ndarray, bx: int, by: int, bs: int, thick: int) -> None:
        """Optionally attach table rules to the box, including rules that share its border.

        This is the single most important negative-proofing detail in the generator: on the
        real forms a checkbox commonly sits flush against a cell rule, and a model trained
        only on isolated boxes learns 'surrounded by whitespace' as a feature and then
        rejects half the real page.
        """
        n = img.shape[0]
        ink = self._ink()
        if self.rng.random() < 0.45:  # horizontal rule through the box's top or bottom edge
            y = self.rng.choice([by, by + bs, self.rng.randint(0, n - 1)])
            cv2.line(img, (0, y), (n, y), ink, max(1, thick - 1))
        if self.rng.random() < 0.45:  # vertical rule flush with a side
            x = self.rng.choice([bx, bx + bs, self.rng.randint(0, n - 1)])
            cv2.line(img, (x, 0), (x, n), ink, max(1, thick - 1))
        if self.rng.random() < 0.30:  # neighbouring text pressing in from one side
            self._text_fragment(img)

    def _text_fragment(self, img: np.ndarray) -> None:
        """Paint a scrap of label text at the window edge, as a real crop would contain."""
        n = img.shape[0]
        ink = self._ink()
        side = self.rng.choice(["l", "r", "t", "b"])
        for _ in range(self.rng.randint(1, 4)):
            wpx = self.rng.randint(2, max(3, n // 6))
            hpx = self.rng.randint(2, max(3, n // 5))
            if side == "l":
                x, y = self.rng.randint(0, max(1, n // 8)), self.rng.randint(0, n - hpx)
            elif side == "r":
                x, y = self.rng.randint(int(n * 0.85), n - 1), self.rng.randint(0, n - hpx)
            elif side == "t":
                x, y = self.rng.randint(0, n - wpx), self.rng.randint(0, max(1, n // 8))
            else:
                x, y = self.rng.randint(0, n - wpx), self.rng.randint(int(n * 0.85), n - 1)
            cv2.rectangle(img, (x, y), (x + wpx, y + hpx), ink, -1)

    def _mark(self, img: np.ndarray, bx: int, by: int, bs: int) -> None:
        """Draw one 'checked' mark. Styles mirror what the four samples actually contain."""
        ink = self._ink()
        t = max(1, round(bs * self.rng.uniform(0.06, 0.16)))
        # Marks routinely overflow the box on hand-filled forms, so allow negative padding.
        pad = int(bs * self.rng.uniform(-0.12, 0.22))
        x0, y0, x1, y1 = bx + pad, by + pad, bx + bs - pad, by + bs - pad
        style = self.rng.choices(
            ["x", "check", "scribble", "fill", "dot", "slash"],
            weights=[42, 22, 12, 8, 8, 8],
        )[0]
        if style == "x":
            cv2.line(img, (x0, y0), (x1, y1), ink, t)
            cv2.line(img, (x0, y1), (x1, y0), ink, t)
        elif style == "check":
            mx, my = x0 + (x1 - x0) * 0.38, y1 - (y1 - y0) * 0.18
            cv2.line(img, (int(x0), int((y0 + y1) / 2)), (int(mx), int(my)), ink, t)
            cv2.line(img, (int(mx), int(my)), (int(x1), int(y0 - (y1 - y0) * 0.15)), ink, t)
        elif style == "scribble":
            pts = [(self.rng.randint(x0, max(x0 + 1, x1)),
                    self.rng.randint(y0, max(y0 + 1, y1))) for _ in range(self.rng.randint(4, 8))]
            for a, b in itertools.pairwise(pts):
                cv2.line(img, a, b, ink, t)
        elif style == "fill":
            cv2.rectangle(img, (x0, y0), (x1, y1), ink, -1)
        elif style == "dot":
            r = max(1, int(bs * self.rng.uniform(0.12, 0.30)))
            cv2.circle(img, ((bx + bs // 2), (by + bs // 2)), r, ink, -1)
        else:  # single diagonal stroke
            cv2.line(img, (x0, y1), (x1, y0), ink, t)

    def _degrade(self, img: np.ndarray) -> np.ndarray:
        """Apply scan/compression artefacts, then resize to the model input size.

        Order matters: rotation and blur are applied at native resolution, JPEG last, so the
        artefacts compose the way they do on a real scanned-then-compressed document.
        """
        n = img.shape[0]
        if self.rng.random() < 0.5:  # slight skew, as in a scanned page
            ang = self.rng.uniform(-2.5, 2.5)
            m = cv2.getRotationMatrix2D((n / 2, n / 2), ang, 1.0)
            img = cv2.warpAffine(img, m, (n, n), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        if self.rng.random() < 0.55:
            k = self.rng.choice([3, 3, 5])
            img = cv2.GaussianBlur(img, (k, k), self.rng.uniform(0.4, 1.3))
        if self.rng.random() < 0.6:
            img = img + self.np_rng.normal(0, self.rng.uniform(2, 12), img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.35:  # JPEG ringing, as in sample 2
            q = self.rng.randint(30, 85)
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

    # ---------------------------------------------------------------- samples

    def _checkbox(self, checked: bool) -> np.ndarray:
        """Render a checkbox crop framed the way crop_with_context will frame a proposal."""
        side = self.rng.randint(self.cfg.min_side, self.cfg.max_side)
        ctx = CONTEXT * self.rng.uniform(0.92, 1.12)  # proposal scale is never exact
        n = max(INPUT_SIZE // 2, round(side * ctx))
        img = self._paper(n)

        bs = side
        off = (n - bs) // 2
        jitter = max(1, int(bs * 0.08))
        bx = int(np.clip(off + self.rng.randint(-jitter, jitter), 0, n - bs - 1))
        by = int(np.clip(off + self.rng.randint(-jitter, jitter), 0, n - bs - 1))
        thick = max(1, round(bs * self.rng.uniform(0.035, 0.11)))

        self._rules(img, bx, by, bs, thick)
        # Slight aspect jitter: printed checkboxes are rarely exactly square.
        bw = int(bs * self.rng.uniform(0.90, 1.10))
        bw = min(bw, n - bx - 1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bs), self._ink(), thick)
        if checked:
            self._mark(img, bx, by, bs)
        elif self.rng.random() < 0.18:  # dust / faint speck inside an empty box
            px = self.rng.randint(bx + thick + 1, max(bx + thick + 2, bx + bw - 1))
            py = self.rng.randint(by + thick + 1, max(by + thick + 2, by + bs - 1))
            cv2.circle(img, (px, py), max(1, bs // 22), self._ink(), -1)
        return self._degrade(img)

    def _glyph_negative(self) -> np.ndarray:
        """Render a letter/digit whose counter a geometric detector mistakes for a box."""
        side = self.rng.randint(self.cfg.min_side, self.cfg.max_side)
        n = max(INPUT_SIZE // 2, round(side * CONTEXT))
        img = self._paper(n)
        ch = self.rng.choice(_COUNTER_GLYPHS)
        if _HAS_PIL and self.fonts:
            pil = Image.fromarray(img.astype(np.uint8))
            draw = ImageDraw.Draw(pil)
            font = ImageFont.truetype(self.rng.choice(self.fonts),
                                      max(6, int(n * self.rng.uniform(0.7, 1.15))))
            draw.text((n * self.rng.uniform(0.05, 0.3), n * self.rng.uniform(-0.15, 0.12)),
                      ch, fill=int(self._ink()), font=font)
            img = np.asarray(pil).astype(np.float32)
        else:  # pragma: no cover - fallback when no TrueType font is installed
            cv2.putText(img, ch, (int(n * 0.1), int(n * 0.85)), cv2.FONT_HERSHEY_SIMPLEX,
                        n / 32.0, self._ink(), max(1, n // 16))
        return self._degrade(img)

    def _text_negative(self) -> np.ndarray:
        """Render text -- bare, or enclosed by a rule -- as a negative.

        This is the highest-value negative class, and its absence was a measured failure:
        a model trained without it scored 0.44 mean probability of "checked" across 35k real
        proposals, because "a bordered region with ink inside it" is exactly what a filled
        checkbox looks like *and* exactly what a table cell containing "93" or "Bay" looks
        like. Appraisal forms are mostly the latter, so without this class the classifier
        reports thousands of filled checkboxes per page.
        """
        side = self.rng.randint(self.cfg.min_side, self.cfg.max_side)
        n = max(INPUT_SIZE // 2, round(side * CONTEXT))
        img = self._paper(n)
        ink = self._ink()
        boxed = self.rng.random() < 0.55

        if boxed:
            # A cell drawn like a checkbox, but holding content a checkbox never holds.
            t = max(1, round(side * self.rng.uniform(0.035, 0.10)))
            bx = by = max(0, (n - side) // 2)
            cv2.rectangle(img, (bx, by), (bx + side, by + side), ink, t)

        content = self.rng.choice([
            "93", "0", "1,700", "55", "%", "N;Res;", "Bay", "12", "4.67", "2022", "IL",
            "$", "A", "No", "Yes", "sf", "14.02", "515", "1.0", "7", "X2", "3-6",
        ])
        if _HAS_PIL and self.fonts:
            pil = Image.fromarray(img.astype(np.uint8))
            draw = ImageDraw.Draw(pil)
            size = max(5, int(n * self.rng.uniform(0.30, 0.62)))
            font = ImageFont.truetype(self.rng.choice(self.fonts), size)
            draw.text((n * self.rng.uniform(0.06, 0.34), n * self.rng.uniform(0.12, 0.42)),
                      content, fill=int(ink), font=font)
            img = np.asarray(pil).astype(np.float32)
        else:  # pragma: no cover - no TrueType font installed
            cv2.putText(img, content, (int(n * 0.1), int(n * 0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, n / 55.0, ink, 1)
        return self._degrade(img)

    def _dark_block_negative(self) -> np.ndarray:
        """Render a crop taken from inside a solid dark band carrying light lettering.

        Every one of the four samples has a black section rail down the left edge with the
        section name set vertically in white -- SUBJECT, CONTRACT, NEIGHBORHOOD. Inside a
        solid block every pixel is ink, so Stage 1 finds long runs everywhere and nominates
        rectangles by the hundred, and the classifier confidently accepted them because no
        training crop had ever been mostly dark.

        Fixing this in the generator rather than with a geometric filter is deliberate. The
        obvious filter -- reject candidates surrounded by too much ink -- separates the rail
        cleanly on the clean page (0.22 removes 51 of 53 false positives, costing nothing)
        and destroys the watermarked page (the same 0.22 drops 46% of its true detections),
        because absolute ink density is a property of the page, not of the object. A
        page-dependent constant is exactly the kind of tuning that looks fine on four samples
        and fails on the fifth. Teaching the model what the inside of a black bar looks like
        has no such coupling.
        """
        side = self.rng.randint(self.cfg.min_side, self.cfg.max_side)
        n = max(INPUT_SIZE // 2, round(side * CONTEXT))
        dark = float(self.rng.choice([0, 0, 8, 20, 35]))
        img = np.full((n, n), dark, np.float32)

        # The rail has an edge somewhere in most crops taken from it; a crop wholly inside is
        # also generated, which is the harder and more common case.
        if self.rng.random() < 0.45:
            edge = self.rng.randint(1, n - 1)
            if self.rng.random() < 0.5:
                img[:, :edge] = 245.0  # paper to the left of the rail
            else:
                img[:, edge:] = 245.0

        light = float(self.rng.choice([255, 250, 235, 215]))
        if _HAS_PIL and self.fonts:
            pil = Image.fromarray(img.astype(np.uint8))
            draw = ImageDraw.Draw(pil)
            font = ImageFont.truetype(self.rng.choice(self.fonts),
                                      max(6, int(n * self.rng.uniform(0.45, 0.95))))
            # The rail text is set vertically, so a crop from it shows one or two letters.
            draw.text((n * self.rng.uniform(-0.05, 0.4), n * self.rng.uniform(-0.2, 0.35)),
                      self.rng.choice("SUBJECTCONRAIGHBOD"), fill=int(light), font=font)
            img = np.asarray(pil).astype(np.float32)
        else:  # pragma: no cover - no TrueType font installed
            cv2.putText(img, "S", (int(n * 0.2), int(n * 0.8)), cv2.FONT_HERSHEY_SIMPLEX,
                        n / 40.0, light, max(1, n // 12))
        return self._degrade(img)

    def _structure_negative(self) -> np.ndarray:
        """Render table junctions, partial boxes and oversized cells -- the other confusers."""
        side = self.rng.randint(self.cfg.min_side, self.cfg.max_side)
        n = max(INPUT_SIZE // 2, round(side * CONTEXT))
        img = self._paper(n)
        ink = self._ink()
        t = max(1, round(side * self.rng.uniform(0.04, 0.12)))
        kind = self.rng.choices(
            ["junction", "partial", "oversized", "stripes", "blob", "blank", "ellipse"],
            weights=[20, 22, 16, 14, 10, 10, 8],
        )[0]
        if kind == "junction":  # two rules crossing: passes a naive four-edge test
            cv2.line(img, (0, self.rng.randint(0, n - 1)), (n, self.rng.randint(0, n - 1)), ink, t)
            cv2.line(img, (self.rng.randint(0, n - 1), 0), (self.rng.randint(0, n - 1), n), ink, t)
        elif kind == "partial":  # a box missing one or two sides
            bs = int(side * self.rng.uniform(0.8, 1.0))
            bx = by = max(0, (n - bs) // 2)
            edges = [((bx, by), (bx + bs, by)), ((bx, by + bs), (bx + bs, by + bs)),
                     ((bx, by), (bx, by + bs)), ((bx + bs, by), (bx + bs, by + bs))]
            self.rng.shuffle(edges)
            for a, b in edges[: self.rng.randint(1, 3)]:
                cv2.line(img, a, b, ink, t)
        elif kind == "oversized":  # a table cell far larger than the window
            cv2.rectangle(img, (int(-n * 0.4), int(-n * 0.4)), (int(n * 1.4), int(n * 1.4)), ink, t)
        elif kind == "stripes":  # body text: several short horizontal strokes
            for _ in range(self.rng.randint(3, 7)):
                y = self.rng.randint(0, n - 1)
                x = self.rng.randint(0, n // 2)
                cv2.line(img, (x, y), (min(n, x + self.rng.randint(n // 4, n)), y), ink,
                         self.rng.randint(1, max(1, t)))
        elif kind == "blob":
            cv2.rectangle(img, (self.rng.randint(0, n // 2), self.rng.randint(0, n // 2)),
                          (self.rng.randint(n // 2, n), self.rng.randint(n // 2, n)), ink, -1)
        elif kind == "ellipse":
            cv2.ellipse(img, (n // 2, n // 2),
                        (int(side * 0.45), int(side * self.rng.uniform(0.3, 0.5))),
                        self.rng.randint(0, 180), 0, 360, ink, t)
        # "blank" intentionally draws nothing: empty paper must be a confident negative.
        return self._degrade(img)

    # ---------------------------------------------------------------- dataset

    def sample(self) -> tuple[np.ndarray, int]:
        """Draw one labelled crop.

        Negatives are the majority class -- roughly 55% -- which mirrors reality rather than
        aesthetics: Stage 1 nominates tens of thousands of regions per page and only a
        hundred or so are checkboxes, so a balanced training set would leave the classifier
        badly miscalibrated for the distribution it actually meets. Text negatives get the
        largest single share because they were the measured failure mode.
        """
        r = self.rng.random()
        if r < 0.22:
            return self._checkbox(checked=False), LABEL_UNCHECKED
        if r < 0.45:
            return self._checkbox(checked=True), LABEL_CHECKED
        if r < 0.66:
            return self._text_negative(), LABEL_NEGATIVE
        if r < 0.78:
            return self._glyph_negative(), LABEL_NEGATIVE
        if r < 0.88:
            return self._dark_block_negative(), LABEL_NEGATIVE
        return self._structure_negative(), LABEL_NEGATIVE

    def dataset(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Build ``n`` samples as (float32 images in [0,1] of shape (n,1,40,40), int labels)."""
        xs = np.empty((n, 1, INPUT_SIZE, INPUT_SIZE), np.float32)
        ys = np.empty((n,), np.int64)
        for i in range(n):
            img, label = self.sample()
            xs[i, 0] = img.astype(np.float32) / 255.0
            ys[i] = label
        return xs, ys
