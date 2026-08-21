# Labelled real crops

`annotations.npz` holds 1832 crops taken from the four supplied sample pages, each one a real
Stage 1 proposal, labelled `not_a_checkbox` / `unchecked` / `checked`. `annotations.json`
carries the same records with their source image and bounding box;
`annotations_preview.png` is the contact sheet for spot-checking.

**Why this is committed rather than regenerated.** Producing it costs real API credit, and a
reviewer should be able to reproduce the trained model without an API key of their own:

    python -m training.train --annotations data/annotations.npz

**How it was produced.** `python -m training.annotate --per-image 500` — see that module for
the sampling and for the two gates the labels pass through. In short: proposals are sampled
across three confidence strata rather than uniformly, regions with no paper inside them are
settled locally instead of being paid for, and every model verdict must agree with a direct
pixel measurement or the crop is discarded rather than resolved in favour of either source.

**Label quality, measured rather than assumed.** A 32-crop audit of each positive class puts
accuracy around 90%: the residual errors are letter counters (`O`, `0`, `p`) judged to be
empty boxes. The first, ungated run of this tool measured roughly 10% on the same audit. The
labels are weak supervision and are treated as such — noisy, useful, and never the arbiter of
whether the detector is right.

`source` on each record says which decided it: `model` for a Claude verdict that survived the
pixel gate, `pixels` for a region settled locally without a call.
