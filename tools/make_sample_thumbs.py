"""Rebuild the sample thumbnails the UI's picker shows.

    python tools/make_sample_thumbs.py

Run this after adding or replacing a page in `samples/`. The output is committed, because it
is small, deterministic, and needed by a frontend build that has no image library of its own.

Small on purpose: the strip renders four postage stamps, and shipping four full pages -- 2.8
MB, one of them 2550x4200 -- to draw them at 120 px would put megabytes in front of the first
paint of a demo that scales to zero. The full page is fetched only when one is chosen.
"""

from __future__ import annotations

import pathlib

import cv2

# Twice the rendered width, so the thumbnails stay sharp on a high-density display.
TARGET_WIDTH = 260
JPEG_QUALITY = 82

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / "samples"
DEST = REPO / "frontend" / "public" / "samples"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in sorted(SOURCE.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"skipping unreadable {path.name}")
            continue
        h, w = img.shape[:2]
        # INTER_AREA is the correct filter for downscaling. Anything else aliases the thin
        # rules these pages are made of into a grey mush -- which matters here, because the
        # thumbnail's whole job is to be recognisable as an appraisal form.
        thumb = cv2.resize(
            img, (TARGET_WIDTH, round(h * TARGET_WIDTH / w)), interpolation=cv2.INTER_AREA
        )
        out = DEST / (path.stem + ".jpg")
        cv2.imwrite(str(out), thumb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        size = out.stat().st_size
        total += size
        print(f"{out.name:34s} {thumb.shape[1]}x{thumb.shape[0]}  {size / 1024:5.1f} KB")

    print(f"total {total / 1024:.0f} KB")
    print(
        "thumbnail dimensions are declared in frontend/src/lib/samples.ts; "
        "update them there if a page's aspect ratio changed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
