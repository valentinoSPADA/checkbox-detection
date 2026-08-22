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

## The measurement was wrong before the model was: unmarked adjudication crops

Investigating a request to visualise sample 1's detections turned up a defect in the *scoring*
rather than the detector. At threshold 0.95 the engine returns 117 boxes on sample 1 — the
ground truth also holds 117 — and 14 of them disagreed on filled/unfilled.

Every one of the 14 ran the same direction: engine "unchecked", ground truth "checked". A
one-directional error is a signature, not noise, so the boxes were measured directly rather
than argued about. Ink inside the candidate's own border, with the border ring excluded:

| | median interior ink |
|---|---|
| engine says checked (36 boxes) | 0.219 |
| engine says unchecked (81 boxes) | 0.000 |
| **the 14 disputed** | **0.000, all fourteen** |

Not "low". Zero, in all fourteen. The boxes are empty and the engine was right about every
one of them; sample 1's real filled/unfilled accuracy at this threshold is **1.000**, not the
0.869 on record.

**Cause.** `build_ground_truth.py`, `annotate.py` and the Go adapter's `cropAround` all cut the
judging crop at 3.0x the candidate's own size. Context is what makes the judgement possible —
it is how a checkbox is told from a small table cell — but on a URAR form a window that wide
contains two or three checkboxes stacked vertically. Every one of the 14 sits directly above
or below a box carrying an X. The model was asked "is this checked?" about a picture
containing an X, and said yes.

**What did not work.** The prompt already carried the instruction: *"The region is centred on
the candidate; ignore other checkboxes near the edges of the crop."* It is a correct
instruction and it did not hold, because when boxes tile the region uniformly, "the centred
one" is a judgement the model has to make before it can follow the rule.

**Fix.** Stop describing the referent and draw it. `imaging.Outline` (Go) and
`engine.preprocess.mark_candidate` (Python) ring the candidate in red — outside its own edge,
so the mark under judgement is never covered — and the prompt now names the ring and states
explicitly that a marked neighbour is irrelevant. Red because these scans are black on white:
nothing red survives binarisation to be mistaken for ink.

**Why this mattered more than the metric.** The same crop path serves the `assisted` engine at
runtime, so every escalated candidate was being adjudicated with the same ambiguity — the
escalation was importing the defect into live answers, not just into the scorecard. And it
would have propagated straight into the retraining set: `annotate.py`'s whole purpose is to
label real proposals, and unmarked crops would have taught the small model that a blank box
beside a marked one is itself marked. The bug was one run away from being baked into weights.

**Cost of the correction.** The published results table is left as measured; regenerating it
means re-running the adjudicator against the API, which costs credit. Every filled/unfilled
figure in the README is therefore a floor, and says so.

## Retraining on real labels: what it cost to find out it worked

Running `training.annotate` for the first time produced 1600 labels that were unusable, and
the audit is the only reason that was discovered rather than trained on. Of 40 crops labelled
`unchecked`, roughly 36 contained no checkbox at all. The full diagnosis is in the commit that
fixed it; the short version is three separate defects, one of them introduced by the previous
fix in this same log:

| Defect | Evidence | Fix |
|---|---|---|
| The prompt asserted a false premise ("EXACTLY ONE BOX IS RINGED") | ~90% of sampled positives wrong | The ring marks a *region*, which may hold nothing |
| Confident tails sampled as one pool | 97 `checked` out of 1600, on pages holding ~400 real boxes | Three strata sampled independently |
| Prompting could not close the rest | two revisions: ~90% -> ~45% | Police every verdict with a pixel measurement |

The third is the interesting one. The measurement is interior brightness, and it works because
a checkbox is a *container*: its inside is paper whether or not anyone ticked it. On the 117
detections of sample 1 confirmed genuine by direct ink measurement, unchecked boxes read 1.000
(all 81) and checked read 0.716-0.828 (all 36), against ~0.0 for the interior of the black
section rail. All 117 survive the gate that this calibrates.

Final label quality: ~90% on a 32-crop audit of each positive class, against ~10% ungated.

### The ground truth had to be rebuilt before the result could be read

The retrained model scored **worse** on the first comparison — F1 0.801 against the previous
0.819. Rendering the 20 boxes it "missed" on sample 1 explained why: 11 were genuine
checkboxes scored 0.81-0.947, just under a threshold calibrated for a different model; the
other 9 were black-rail blobs the old ground truth had recorded as checkboxes and the new
model now rejects so firmly they fall below 0.05. It was being penalised for being right.

So the ground truth was rebuilt with the marked crops. Pixel-contradicted labels roughly
halved:

| Ground truth | says checked, zero interior ink | says unchecked, interior heavily inked |
|---|---|---|
| v1, unmarked crops | 24/162 (14.8%) | 26/177 (14.7%) |
| v2, marked crops | 15/183 (8.2%) | 19/217 (8.8%) |

The pool for v2 is the **union** of both models' candidates. Building it from the current
model's pool alone would have been circular: that pool is 165 candidates on sample 1 against
the previous model's 1722, so boxes the new model stopped proposing would simply have left the
ground truth and its recall could not have fallen whatever it lost.

### Result, both models against the same rebuilt reference

| Model | floor | Precision | Recall | F1 |
|---|---|---|---|---|
| Synthetic only | 0.95 | 0.884 | 0.670 | 0.762 |
| Synthetic only | 0.90 | 0.533 | 0.695 | 0.603 |
| **+ real labels** | **0.90** | **0.930** | **0.703** | **0.801** |
| + real labels | 0.95 | 0.931 | 0.605 | 0.733 |

**+3.9 F1, +4.6 precision, +3.3 recall.** The second row is the more informative one: the
synthetic-only model lost 35 points of precision moving its floor by 0.05, so its operating
point was balanced on a knife edge. The retrained model holds 0.93 across 0.70-0.90. A model
that is insensitive to its own threshold is worth more than a slightly higher peak, because
the peak was never going to survive the next page.

### Two things this run establishes for the next one

* **A threshold belongs to a model, not to a problem.** Keeping 0.95 across a retrain cost 10
  points of F1 and read as a regression. `DefaultPolicy` now carries the sweep and the
  instruction to redo it.
* **Synthetic validation cannot referee this.** It read 0.9965 for both models. Real held-out
  validation reads 0.9651 and moves independently -- epoch 5 fell 5 points on real crops while
  synthetic barely twitched. `train.py --annotations` now holds out real crops and reports
  both, splitting before oversampling so copies of one crop cannot land on both sides.

## Hand labels: the size signal, and an evaluation that had to be thrown out

627 candidates labelled by hand — the whole surfaced pool across the four pages, plus a sample
of the classifier's rejects so that a *miss* could be recorded at all. No skips.

Agreement with the model then in service: **67.9%**. The first version of that report said
44.5%, because it counted `rejected -> not_a_checkbox` as a disagreement when it is the same
answer in two vocabularies. A metric that punishes a model for being right is worse than none.

Of 201 corrections, only **27 sat above the 0.90 floor** and therefore reached the API. Their
median size was **12 px**, which pointed straight at the actual defect.

### The classifier was size-blind by construction

It receives a 40x40 crop. A 10 px letter counter and a 50 px checkbox arrive identical. It was
not under-trained on the distinction — it was never shown the feature that carries it.

The hand labels quantified what the models could not:

| Label | n | min side | median | max |
|---|---|---|---|---|
| checked | 90 | **20 px** | 50 | 56 |
| unchecked | 194 | **20 px** | 50 | 56 |
| not a checkbox | 343 | 10 px | **10 px** | 54 |

### Why the fix is relative, not absolute

An 18 px floor would have removed 85% of the rejects at zero cost to the confirmed boxes. It
was not taken, because a pixel count means different things on different pages: 10 px is
0.0039 of sample 1 and 0.0063 of sample 2. Measured as a fraction of page width instead:

| | side / page width |
|---|---|
| 284 confirmed checkboxes | 0.0094 - 0.0220 |
| 343 rejected candidates | median 0.0047 |

So Stage 1 now sweeps `0.0065 - 0.0300` of width — 31% below the smallest confirmed box, 37%
above the largest — with an absolute fallback for inputs too small for a fraction to mean
anything. The rule may only ever *narrow* the sweep; a 200 px crop still gets the old 10-70.

| Sample | sweep | raw proposals before | after | cut |
|---|---|---|---|---|
| 1 | 17-77 | 108 187 | 31 142 | 71% |
| 3 | 17-77 | 52 273 | 13 662 | 74% |
| 4 | 17-77 | 48 937 | 10 581 | 78% |

**284 of 284 confirmed checkboxes are still proposed.**

### Two bugs found while doing this, both mine

**A validation leak I had already warned myself about.** `import_labels` repeated each hand
label four times before writing the file; `train.py` then split a validation set off the
result, so four copies of one crop landed on both sides. Held-out "real" accuracy read 0.9990
— a memorisation score wearing a generalisation label. The comment *"Split BEFORE
oversampling"* was already in `train.py`; the leak was reintroduced one layer up, where that
comment could not see it. Repetition is now a weight in the file, applied after the split, and
a test reconstructs the scenario.

**An evaluation that measured nothing.** Scoring the retrained model against the hand labels
gave F1 0.991 and recall 1.000 — on pages 470 of whose labels it had trained on. The fix was
leave-one-page-out: train on samples 1-3 only, score sample 4.

| | GT | Precision | Recall | F1 | Filled/unfilled |
|---|---|---|---|---|---|
| Sample 4, unseen page | 79 | 1.000 | 1.000 | 1.000 | 1.000 |

Verified by hashing the model inside the running container rather than trusting the file
name — the four runs wrote to one path, and three had already overwritten each other.

### What the hand labels proved about the model-made ones

The Claude ground truth holds **44 boxes on sample 4 that are not checkboxes**, every one of
them exactly 10 px. Scored against it, sample 4's recall was 0.650 and most of the "misses"
were things that should never have been found. The measurement had been the bottleneck for
longer than the model was.
