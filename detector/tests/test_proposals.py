"""Tests for Stage 1, the geometric proposal generator.

These tests are written against synthetic pages whose checkbox positions are known exactly,
because the four supplied samples carry no ground truth. Each case encodes one property the
proposal stage must hold, and several encode a failure a previous prototype actually had --
those are the ones worth keeping, since they are the regressions most likely to recur.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from engine.preprocess import binarize, crop_with_context, decode, mark_candidate, to_gray
from engine.proposals import MAX_SIDE, MIN_SIDE, Proposal, _run_lengths, propose
from training.annotate import interior_brightness, stratified_sample, verdict_agrees
from training.import_labels import LABELS, SAME_BOX_IOU, iou


def page(width: int = 300, height: int = 200, shade: int = 255) -> np.ndarray:
    """A blank grayscale page at a given background level."""
    return np.full((height, width), shade, np.uint8)


def draw_box(img: np.ndarray, x: int, y: int, side: int, thickness: int = 2,
             ink: int = 0) -> None:
    """Draw a checkbox outline in place."""
    cv2.rectangle(img, (x, y), (x + side, y + side), ink, thickness)


def covers(proposals: list[Proposal], x: int, y: int, side: int, slack: int = 4) -> bool:
    """Whether any proposal lands on the box at (x, y) with roughly the right size."""
    return any(
        abs(p.x - x) <= slack and abs(p.y - y) <= slack
        and abs(p.w - side) <= slack + 2 and abs(p.h - side) <= slack + 2
        for p in proposals
    )


class TestRunLengths:
    def test_counts_rightward_run(self):
        ink = np.array([[True, True, True, False, True]])
        assert _run_lengths(ink, axis=1).tolist() == [[3, 2, 1, 0, 1]]

    def test_counts_downward_run(self):
        ink = np.array([[True], [True], [False], [True]])
        assert _run_lengths(ink, axis=0).ravel().tolist() == [2, 1, 0, 1]

    def test_all_blank_is_all_zero(self):
        ink = np.zeros((4, 4), bool)
        assert _run_lengths(ink, axis=1).sum() == 0

    def test_all_ink_row_counts_down_to_one(self):
        ink = np.ones((1, 5), bool)
        assert _run_lengths(ink, axis=1).tolist() == [[5, 4, 3, 2, 1]]


class TestPropose:
    def test_finds_an_isolated_box(self):
        img = page()
        draw_box(img, 50, 50, 24)
        got = propose(binarize(img))
        assert covers(got, 50, 50, 24), f"missed the box; got {len(got)} proposals"

    @pytest.mark.parametrize("side", [12, 20, 28, 40, 60])
    def test_is_scale_invariant_across_the_supported_range(self, side):
        # Sample 2 is a zoomed crop whose boxes are several times the size of sample 1's, so
        # a single expected size cannot exist and the sweep has to cover the whole band.
        img = page(400, 300)
        draw_box(img, 100, 100, side, thickness=max(1, side // 12))
        got = propose(binarize(img))
        assert covers(got, 100, 100, side, slack=6), f"missed a {side}px box"

    def test_finds_a_box_touching_a_table_rule(self):
        # This is the exact case that broke the connected-component prototype: the box and
        # the rule become one component, so component-based detection loses the box entirely.
        img = page()
        draw_box(img, 60, 80, 24)
        cv2.line(img, (0, 80), (300, 80), 0, 2)   # rule through the top edge
        cv2.line(img, (60, 0), (60, 200), 0, 2)   # rule through the left edge
        got = propose(binarize(img))
        assert covers(got, 60, 80, 24), "a box sharing borders with table rules was lost"

    def test_finds_a_box_on_a_shaded_background(self):
        # Sample 3 places checkboxes on blue-shaded table rows; under a global threshold the
        # shading either swallows the rules or floods the row.
        img = page(shade=205)
        draw_box(img, 40, 40, 22, ink=20)
        got = propose(binarize(img))
        assert covers(got, 40, 40, 22), "a box on a shaded row was lost"

    def test_finds_a_marked_box(self):
        img = page()
        draw_box(img, 70, 60, 26)
        cv2.line(img, (74, 64), (92, 82), 0, 3)   # the X
        cv2.line(img, (74, 82), (92, 64), 0, 3)
        got = propose(binarize(img))
        assert covers(got, 70, 60, 26), "a marked box was lost"

    def test_tolerates_a_border_gap_up_to_the_bridge_width(self):
        # Scanned rules drop pixels, and a longest-run test is unforgiving of that: without
        # the morphological bridge, a four-pixel dropout in a thirty-pixel border halves the
        # run and vetoes the whole box. BRIDGE closes gaps strictly narrower than itself.
        img = page()
        draw_box(img, 50, 50, 30, thickness=2)
        cv2.rectangle(img, (62, 49), (63, 52), 255, -1)  # erase a 2 px slice of the top rule
        got = propose(binarize(img))
        assert covers(got, 50, 50, 30, slack=6), "a box with a bridgeable border gap was lost"

    def test_a_gap_wider_than_the_bridge_is_not_silently_recovered(self):
        # The complement of the case above, kept so the bridge cannot be widened casually:
        # bridging arbitrarily large gaps would start manufacturing rules out of adjacent
        # glyphs and flood the classifier with false proposals.
        img = page()
        draw_box(img, 50, 50, 30, thickness=2)
        cv2.rectangle(img, (56, 49), (78, 52), 255, -1)  # erase most of the top rule
        got = propose(binarize(img))
        assert not covers(got, 50, 50, 30, slack=2), "a box missing its top rule was proposed"

    def test_finds_every_box_in_a_dense_row(self):
        img = page(500, 120)
        xs = [30, 110, 190, 270, 350]
        for x in xs:
            draw_box(img, x, 40, 26)
        got = propose(binarize(img))
        missed = [x for x in xs if not covers(got, x, 40, 26)]
        assert not missed, f"missed boxes at x={missed}"

    def test_blank_page_yields_nothing(self):
        assert propose(binarize(page())) == []

    def test_respects_the_size_band(self):
        img = page(400, 400)
        draw_box(img, 20, 20, 6)     # below MIN_SIDE
        draw_box(img, 150, 150, 200)  # above MAX_SIDE
        got = propose(binarize(img))
        for p in got:
            assert MIN_SIDE <= p.w <= MAX_SIDE, f"proposal outside the band: {p}"

    def test_returns_proposals_unsuppressed(self):
        # Suppression is deferred until confidences exist so it can keep the best-scoring
        # box rather than an arbitrary geometric pick; overlapping duplicates here are
        # expected, not a defect.
        img = page()
        draw_box(img, 50, 50, 30, thickness=3)
        got = propose(binarize(img))
        assert len(got) > 1, "expected overlapping raw proposals at several scales"

    def test_proposal_bbox_matches_the_challenge_schema(self):
        p = Proposal(10, 20, 30, 40)
        assert p.as_bbox() == [10, 20, 40, 60]


class TestPreprocess:
    def test_decode_rejects_non_image_bytes(self):
        with pytest.raises(ValueError):
            decode(b"this is definitely not a png")

    def test_decode_roundtrips_png(self):
        img = page(40, 30)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        got = decode(buf.tobytes())
        assert got.shape[:2] == (30, 40)

    def test_to_gray_passes_through_single_channel(self):
        g = page(10, 10)
        assert to_gray(g) is g

    def test_binarize_marks_dark_pixels_as_ink(self):
        img = page()
        cv2.rectangle(img, (40, 40), (80, 80), 0, -1)
        ink = binarize(img)
        assert ink.dtype == bool
        assert ink[60, 60] or ink[41, 41], "a solid dark block produced no ink"
        assert not ink[5, 5], "blank paper was marked as ink"

    def test_crop_with_context_is_larger_than_the_box(self):
        img = page(200, 200)
        draw_box(img, 80, 80, 20)
        patch = crop_with_context(img, 80, 80, 20, 20, context=1.35, out=40)
        assert patch.shape == (40, 40)
        assert patch.dtype == np.float32
        assert patch.min() >= 0.0 and patch.max() <= 1.0

    def test_crop_at_the_page_margin_replicates_rather_than_pads_black(self):
        # Zero-padding would give a margin checkbox an artificial dark border, which the
        # classifier would read as a fifth rule and score as a stronger box than it is.
        img = page(100, 100)
        draw_box(img, 4, 4, 18)  # near the corner, so the context window runs off the page
        patch = crop_with_context(img, 4, 4, 18, 18, context=2.5, out=40)
        assert patch[0, 0] > 0.5, "page margin was padded with black instead of replicated"

    def test_crop_of_a_degenerate_region_is_safe(self):
        img = page(50, 50)
        patch = crop_with_context(img, 49, 49, 1, 1, context=1.0, out=40)
        assert patch.shape == (40, 40)


class TestMarkCandidate:
    """The ring must land on the candidate, and must not cover it.

    These assertions are about position, not aesthetics: a ring drawn over the candidate's
    interior would hide the very mark the annotator is asked to judge, and a ring drawn in the
    wrong place would point at a neighbour -- which is the defect it exists to fix.
    """

    def test_ring_surrounds_the_centre_without_covering_it(self):
        patch = np.full((96, 96), 255, np.uint8)
        out = mark_candidate(patch, context=3.0)
        assert out.shape == (96, 96, 3)

        # The candidate occupies a third of the side, so its interior is the middle ~32 px.
        interior = out[42:54, 42:54]
        assert (interior == 255).all(), "the ring must not intrude on the judged interior"

        red = (out[:, :, 2] == 255) & (out[:, :, 0] == 0) & (out[:, :, 1] == 0)
        assert red.any(), "a marker must actually be drawn"

        ys, xs = np.nonzero(red)
        # Centred, and sized to the candidate rather than to the crop.
        assert abs((ys.min() + ys.max()) / 2 - 48) <= 1
        assert abs((xs.min() + xs.max()) / 2 - 48) <= 1
        assert 30 <= (xs.max() - xs.min()) <= 44

    def test_accepts_float_input_from_crop_with_context(self):
        # annotate.py passes float32 [0,1]; build_ground_truth.py passes uint8.
        out = mark_candidate(np.zeros((96, 96), np.float32), context=3.0)
        assert out.dtype == np.uint8
        red = (out[:, :, 2] == 255) & (out[:, :, 0] == 0) & (out[:, :, 1] == 0)
        assert red.any()

    def test_ring_scales_with_the_context_factor(self):
        """A wider crop means a proportionally smaller candidate, so a smaller ring."""
        def span(ctx):
            # All three channels: on a white patch the red channel alone is 255 everywhere.
            out = mark_candidate(np.full((96, 96), 255, np.uint8), ctx)
            red = (out[:, :, 2] == 255) & (out[:, :, 0] == 0) & (out[:, :, 1] == 0)
            xs = np.nonzero(red)[1]
            return xs.max() - xs.min()

        assert span(1.5) > span(3.0) > span(6.0)


class TestStratifiedSample:
    """Budget allocation, which is where the first annotation run went wrong.

    Sampling the confident tails as one pool let the low tail -- larger by two orders of
    magnitude -- swallow the share meant for confident checkboxes, and the labelled set came
    back with 97 positives out of 1600. These tests pin the three-way split.
    """

    def _scores(self):
        # Shaped like a real page: thousands of confident rejects, a thin uncertain band,
        # a hundred or so confident detections.
        return np.concatenate([
            np.full(5000, 0.05),   # low tail
            np.full(300, 0.5),     # uncertain band
            np.full(120, 0.95),    # high tail
        ])

    def test_confident_positives_are_not_swallowed_by_the_low_tail(self):
        scores = self._scores()
        idx = stratified_sample(scores, 400, np.random.default_rng(0))
        picked = scores[idx]
        assert (picked >= 0.8).sum() >= 100, "the high tail must get its own share"
        assert (picked <= 0.2).sum() <= 130, "the low tail must not take more than its share"

    def test_spends_the_whole_budget(self):
        scores = self._scores()
        idx = stratified_sample(scores, 400, np.random.default_rng(0))
        assert len(idx) == 400
        assert len(set(idx.tolist())) == 400, "no crop may be paid for twice"

    def test_short_stratum_hands_its_surplus_back(self):
        # No confident detections at all: the budget must still be spent, not truncated.
        scores = np.concatenate([np.full(5000, 0.05), np.full(300, 0.5)])
        idx = stratified_sample(scores, 400, np.random.default_rng(0))
        assert len(idx) == 400

    def test_returns_everything_when_the_pool_is_smaller_than_the_budget(self):
        scores = np.full(50, 0.5)
        idx = stratified_sample(scores, 400, np.random.default_rng(0))
        assert len(idx) == 50


class TestPixelGate:
    """The measurement that polices the annotator's verdicts.

    Calibration, stated once so the numbers are not mysterious: on the 117 detections of
    sample 1 confirmed genuine by direct ink measurement, unchecked boxes measure interior
    brightness 1.000 (all 81) and checked boxes 0.716-0.828 (all 36). Every bound below sits
    outside that range with margin, and all 117 survive the gate.
    """

    def _box(self, interior: int, side: int = 40):
        """A drawn box: black border, `interior` grey inside."""
        img = np.full((side * 3, side * 3), 255, np.uint8)
        cv2.rectangle(img, (side, side), (2 * side, 2 * side), 0, 2)
        img[side + 3:2 * side - 3, side + 3:2 * side - 3] = interior
        return img

    def test_measures_the_inside_not_the_border(self):
        # A thickly printed empty box must still read as empty; otherwise the gate would
        # reject boxes for being well printed.
        img = self._box(255)
        assert interior_brightness(img, 40, 40, 40, 40) > 0.95

    def test_a_dark_region_reads_as_void(self):
        assert interior_brightness(np.zeros((120, 120), np.uint8), 40, 40, 40, 40) < 0.05

    def test_a_degenerate_region_does_not_crash_or_condemn(self):
        # Zero-area interiors happen at the smallest proposal sizes; defaulting to 1.0 lets
        # the annotator decide rather than silently auto-rejecting the crop.
        assert interior_brightness(np.full((10, 10), 255, np.uint8), 0, 0, 2, 2) == 1.0

    def test_not_a_checkbox_is_never_contradicted(self):
        # The safe class is always allowed through: a wrong negative costs one crop, a wrong
        # positive teaches the detector that page furniture is a form control.
        for v in (0.0, 0.5, 1.0):
            assert verdict_agrees("not_a_checkbox", v)

    def test_unchecked_requires_a_blank_interior(self):
        assert verdict_agrees("unchecked", 1.0)
        assert not verdict_agrees("unchecked", 0.05), "a rail interior is not a blank box"

    def test_checked_requires_a_mark_that_neither_fills_nor_vanishes(self):
        assert verdict_agrees("checked", 0.76)   # measured centre of the real range
        assert not verdict_agrees("checked", 0.0)    # solid dark: nothing to mark
        assert not verdict_agrees("checked", 1.0)    # perfectly blank: nothing was marked

    def test_the_confirmed_real_range_passes_end_to_end(self):
        for v in (0.716, 0.828):
            assert verdict_agrees("checked", v)
        assert verdict_agrees("unchecked", 1.000)


class TestHandLabelMerge:
    """The rule that decides whose label wins when a person and a model describe one box.

    Getting this wrong is silent: an exact-coordinate match would leave the model's mistake in
    the training set right beside the human correction of it, so the model would be taught the
    error precisely where it was caught.
    """

    def test_a_shifted_box_is_still_the_same_box(self):
        # The two label sets come from the same proposal pool, but a box can move a pixel or
        # two between runs. That must not read as a different checkbox.
        assert iou([100, 100, 152, 152], [101, 101, 153, 153]) > SAME_BOX_IOU

    def test_a_neighbouring_box_is_not_the_same_box(self):
        # The checkbox one row down, which on these forms is roughly one side away. If this
        # matched, a single hand label would silently delete its neighbour's label too.
        assert iou([100, 100, 152, 152], [100, 155, 152, 207]) < SAME_BOX_IOU

    def test_degenerate_boxes_never_match(self):
        assert iou([10, 10, 10, 10], [10, 10, 10, 10]) == 0.0

    def test_labels_map_to_the_classifier_three_classes(self):
        # The importer writes indices straight into the training set; a mismatch with the
        # model's class order would train every label onto the wrong output.
        assert LABELS == {"not_a_checkbox": 0, "unchecked": 1, "checked": 2}
