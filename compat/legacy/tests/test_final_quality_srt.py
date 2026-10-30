from __future__ import annotations

from pathlib import Path

from qwen_asr.final_quality import validate_srt as legacy_validate_srt
from qwen_asr.final_quality_srt import parse_srt_timestamp, srt_legality_check, validate_srt
from qwen_asr.models import WorkPaths


def test_parse_srt_timestamp_rejects_invalid_minute_or_second() -> None:
    assert parse_srt_timestamp("00:01:02,003") == 62003
    assert parse_srt_timestamp("00:60:00,000") is None
    assert parse_srt_timestamp("00:00:60,000") is None
    assert parse_srt_timestamp("bad") is None


def test_validate_srt_detects_overlap_and_keeps_legacy_alias(tmp_path: Path) -> None:
    path = tmp_path / "bad.srt"
    path.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "hello\n\n"
        "3\n"
        "00:00:01,400 --> 00:00:01,800\n"
        "world\n",
        encoding="utf-8",
    )

    issues = validate_srt(path)

    assert any(item["type"] == "non_continuous_index" for item in issues)
    assert any(item["type"] == "overlap" and item["severity"] == "FAIL" for item in issues)
    assert legacy_validate_srt(path) == issues


def test_srt_legality_check_preserves_missing_optional_srt_skip_shape(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    result = srt_legality_check(paths, require_srt=False)

    assert result == {
        "name": "srt_legality",
        "status": "PASS",
        "skipped": True,
        "message": "未生成 SRT，跳过 SRT 合法性检查",
    }


def test_srt_legality_check_fails_when_required_srt_is_missing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    result = srt_legality_check(paths, require_srt=True)

    assert result["name"] == "srt_legality"
    assert result["status"] == "FAIL"
