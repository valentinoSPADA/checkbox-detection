# Design source

The working files behind the UI redesign. `Main.dc.html` is the artboard, `canvas.json` its
layout, `form-crop.jpg` the sample region it renders, and `boxes.json` the real detections
that back the overlay in the mockup — the boxes in the design are measured output from the
running service, not drawn by hand.

The published canvas is regenerated from these files; the seeded `.html` is build output and
is deliberately not tracked (see `.gitignore`).

The design that shipped from here lives in `frontend/src` — `styles.css` carries the tokens,
and `DESIGN.md` records where they came from.
