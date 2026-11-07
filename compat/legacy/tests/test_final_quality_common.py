from __future__ import annotations

from qwen_asr.final_quality_common import fail, float_or_none, int_or_none, normalize_status, passed, skip, warn


def test_status_helpers_preserve_quality_report_schema() -> None:
    assert passed("check", "ok", count=1) == {
        "name": "check",
        "status": "PASS",
        "message": "ok",
        "count": 1,
    }
    assert warn("check", "review")["status"] == "WARN"
    assert fail("check", "bad")["status"] == "FAIL"
    assert skip("check", "unused") == {
        "name": "check",
        "status": "PASS",
        "skipped": True,
        "message": "unused",
    }


def test_status_and_optional_number_normalization() -> None:
    assert normalize_status("pass") == "PASS"
    assert normalize_status("unexpected") == "WARN"
    assert float_or_none("1.25") == 1.25
    assert float_or_none(None) is None
    assert int_or_none("2.6") == 3
    assert int_or_none("bad") is None
