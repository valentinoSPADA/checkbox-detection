"""Tests for the sidecar's HTTP surface and its configuration handling."""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import _log_level, _safe_for_log, app


def png_bytes(width: int = 200, height: int = 150) -> bytes:
    """A small blank page, encoded as PNG."""
    img = np.full((height, width), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (66, 66), 0, 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestLogLevel:
    """LOG_LEVEL is shared with the Go gateway, which spells its levels in lowercase.

    Python's logging module raises on anything but uppercase, so an unforgiving reading of
    this variable crashes the sidecar at import time while the gateway starts fine — a real
    bug this suite exists to prevent recurring.
    """

    @pytest.mark.parametrize("raw", ["info", "INFO", "Info", " info "])
    def test_accepts_any_casing(self, raw):
        assert _log_level(raw) == logging.INFO

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("debug", logging.DEBUG), ("warning", logging.WARNING), ("error", logging.ERROR)],
    )
    def test_resolves_the_other_levels(self, raw, expected):
        assert _log_level(raw) == expected

    def test_unset_defaults_to_info(self):
        assert _log_level(None) == logging.INFO

    def test_unknown_value_falls_back_rather_than_raising(self):
        # A typo in a log level must never be the reason a service refuses to start.
        assert _log_level("chatty") == logging.INFO


class TestHealth:
    def test_reports_model_state(self):
        with TestClient(app) as client:
            body = client.get("/health").json()
        assert body["status"] in {"ok", "degraded"}
        assert isinstance(body["model_loaded"], bool)

    def test_returns_200_even_when_degraded(self):
        # Orchestrators gate traffic on the model_loaded field, not on the status code: a
        # model-less container should be observable, not merely unreachable.
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200


class TestDetect:
    def test_rejects_undecodable_bytes(self):
        with TestClient(app) as client:
            resp = client.post("/v1/detect", files={"file": ("x.png", b"not an image")})
        # 400 when the model is loaded; 503 when it is not, since the request never gets
        # far enough to be decoded. Both are correct, and neither is a 500.
        assert resp.status_code in {400, 503}

    def test_rejects_oversized_upload(self, monkeypatch):
        import app.main as main

        monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
        with TestClient(app) as client:
            resp = client.post("/v1/detect", files={"file": ("x.png", png_bytes())})
        assert resp.status_code in {413, 503}

    def test_returns_candidates_for_a_valid_page(self):
        with TestClient(app) as client:
            resp = client.post("/v1/detect", files={"file": ("page.png", png_bytes())})
            if resp.status_code == 503:
                pytest.skip("model artifact not built in this environment")
            body = resp.json()
        assert body["width"] == 200 and body["height"] == 150
        assert isinstance(body["candidates"], list)
        # The counters are what make a bad page diagnosable; their absence would hide
        # whether a poor result came from Stage 1 or Stage 2.
        assert body["raw_proposals"] >= 0
        assert body["scored_proposals"] >= 0

    def test_floor_is_honoured(self):
        with TestClient(app) as client:
            low = client.post("/v1/detect?floor=0.0", files={"file": ("p.png", png_bytes())})
            if low.status_code == 503:
                pytest.skip("model artifact not built in this environment")
            high = client.post("/v1/detect?floor=0.99", files={"file": ("p.png", png_bytes())})
        assert len(high.json()["candidates"]) <= len(low.json()["candidates"])

    def test_floor_outside_range_is_rejected(self):
        with TestClient(app) as client:
            assert client.post(
                "/v1/detect?floor=2.0", files={"file": ("p.png", png_bytes())}
            ).status_code == 422


def test_upload_stream_is_not_required_to_be_seekable():
    """The handler reads the upload once; it must not depend on rewinding the stream."""
    stream = io.BytesIO(png_bytes())
    with TestClient(app) as client:
        resp = client.post("/v1/detect", files={"file": ("p.png", stream)})
    assert resp.status_code in {200, 503}


class TestSafeForLog:
    """Filenames are attacker-chosen and the log line is JSON built by a format string.

    A name carrying a quote or a newline therefore forges log entries -- and every field after
    it in the record. Sanitising is the fix; these pin the cases that matter.
    """

    def test_ordinary_names_pass_through(self):
        assert _safe_for_log("sample_1_urar_1004.png") == "sample_1_urar_1004.png"

    def test_quotes_are_escaped_not_dropped(self):
        # Escaped rather than removed: the name still identifies the upload in a support
        # conversation, which is the only reason it is logged at all.
        assert _safe_for_log('a"b') == 'a\\"b'
        # One backslash in, two out. Spelled with chr(92) because the shell heredoc
        # that first wrote this test silently ate one of the escapes.
        assert _safe_for_log('back' + chr(92) + 'slash') == 'back' + chr(92) * 2 + 'slash'

    def test_a_forged_log_record_cannot_survive(self):
        got = _safe_for_log('x\n{"level":"ERROR","msg":"breach"}')
        assert "\n" not in got
        assert '\\"' in got, "the injected quotes must be escaped, not passed through"

    def test_control_characters_are_stripped(self):
        assert _safe_for_log("ok\x00\x07\x1b[31m.png") == "ok[31m.png"

    def test_length_is_bounded(self):
        # The field's length is attacker-chosen too; a 10 MB filename should not become a
        # 10 MB log line.
        got = _safe_for_log("A" * 5000)
        assert len(got) < 200

    def test_a_missing_filename_is_a_placeholder_not_the_word_none(self):
        assert _safe_for_log(None) == "-"
        assert _safe_for_log("") == "-"
