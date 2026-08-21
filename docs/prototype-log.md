# Prototype log

A record of what was measured, in the order it was measured, including the approaches that
failed. It exists because the architecture in `DESIGN.md` is a consequence of these numbers,
and a decision presented without the evidence that forced it reads as a preference.

All measurements are on the four images extracted from the challenge PDF (`samples/`).

## Stage 1: four generations of geometric detector

| # | Approach | Result | Why it failed |
|---|---|---|---|
| v1 | Morphological line masks → enclosed-cell segmentation | Detections landed on letter counters (`o`, `a`, `8`) | The opening kernel must be long enough to exclude a bold-text stroke (~12 px at 300 dpi) and short enough to keep a checkbox rule (~22 px). No such kernel exists. |
| v2 | Connected components + border-band verification | Better precision, poor recall | The band test (`max` over a 20% band) is far too permissive at 10 px, where the band is 2 px wide. |
| v3 | Connected components + straight-run verification | Precision good; **recall collapsed to 5 detections on sample 3** | On dense forms a checkbox *touches* the adjacent table rule, so its connected component becomes the entire table grid. Component-based detection cannot work on these documents. |
| v4 | Multi-scale rectangle assembly from straight ink runs | Recall solved; 2 198 survivors on sample 1 against ~120 true | Assembly is immune to shared rules, but any two long rules plus any two long verticals form a rectangle, and these pages are full of long rules. |

v4 also carried a bug worth recording: a ring-density filter measured over a band of
`0.16 × side` rejected *real* checkboxes, because a 45 px box with a 4 px rule scores ≈ 0.57
against a 0.70 threshold. The sampling band has to approximate the rule thickness. The same
mistake was made a second time later (below), which is why it is written down.

**Conclusion.** Each generation traded recall against precision without winning both. The
ceiling is a property of the problem: geometry cannot distinguish a checkbox from a
checkbox-sized table cell. This is what justifies the learned Stage 2.

## Stage 1 parameter measurements

Raw proposal count on `sample_1_urar_1004.png` (2550×4200):

| `BRIDGE` | `span` | Raw proposals |
|---|---|---|
| 1 | 0.80 | 78 448 |
| 3 | 0.80 | 108 187 |
| 5 | 0.80 | **320 118** |

`BRIDGE` is the morphological closing applied along each axis before run lengths are
measured, to tolerate dropouts in scanned rules. At 5 it fuses the glyphs of a text line into
one long run and treats that run as a rule — a 4× cost the classifier pays on every page, in
exchange for tolerating border gaps wider than any real scan produces. Settled at 3, which
closes the one- and two-pixel dropouts that actually occur.

## Geometric filters that were tried and removed

After assembly, two integral-image filters were added to cut the candidate count, then
removed when measured properly:

| Filter | Kept on sample 1 |
|---|---|
| Ring density ≥ 0.5 (band sized to the rule, candidate padded 1 px) | 87.0% |
| Ring density ≥ 0.7 | 57.9% |
| ≤ 1 side shared with a rule longer than 2.5 × side | 32.6% |
| ≤ 2 sides shared (needed to keep boxes inside table rows) | 78.4% |

None discriminates. That is not surprising in hindsight: the candidates were *assembled from*
inked runs, so having an inked perimeter is true of all of them by construction. The variant
that filtered hardest (≤ 1 shared side) also rejects a checkbox sitting between two table
rules, which is a common real layout — so it trades recall for precision, which is exactly the
trade this stage must not make.

The filters were removed rather than kept at a weak setting. Machinery that removes 13% of
candidates and adds a failure mode is not worth its own explanation.

## Stage 2: synthetic training and the domain gap

| Model | Synthetic val accuracy | Candidates ≥ 0.6 on sample 1 (true ≈ 120) |
|---|---|---|
| v1, no text negatives | 0.9972 | 11 448, of which 10 442 "checked" |
| v2, text and boxed-text negatives added | 0.9965 | 9 737, of which 8 775 "checked" |

The diagnostic that explained it: mean `p_checked` across 35k real proposals was 0.44, while
the *highest*-confidence detections were genuine checkboxes (empty boxes at 0.98–0.99, X-marked
boxes at 0.98). The model had learned "a bordered region containing ink", which describes a
filled checkbox and a populated table cell equally well. Appraisal forms are mostly the latter.

Adding boxed-text negatives helped (−15%) but did not close the gap. It cannot: a generator
only contains failure modes someone thought of, and the real distribution has more of them
than anyone enumerates in an afternoon.

**Conclusion.** The remaining error is a labelled-data problem, not a tuning problem. That is
what `detector/training/annotate.py` exists for — Claude labels real Stage 1 proposals, and
the small model trains on the real distribution instead of an imagined one.
