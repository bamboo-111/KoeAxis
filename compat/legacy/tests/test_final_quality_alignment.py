from __future__ import annotations

from qwen_asr.final_quality import (
    _SimpleToken,
    _alignment_coverage,
    _alignment_health_check,
    _float_or_none,
    _one_ms_token_stats,
)
from qwen_asr.final_quality_alignment import (
    SimpleToken,
    alignment_coverage,
    alignment_health_check,
    float_or_none,
    one_ms_token_stats,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_alignment_health_check_skips_missing_manifest(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)

    assert alignment_health_check(paths) == {
        "name": "alignment_health",
        "status": "PASS",
        "skipped": True,
        "message": "未运行 align 阶段",
    }
    assert _alignment_health_check(paths)["status"] == "PASS"


def test_alignment_health_check_fails_failed_segment(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [{"segment_id": "s1", "status": "failed", "global_start_time": 0.0, "global_end_time": 1.0}],
    )

    check = alignment_health_check(paths)

    assert check["status"] == "FAIL"
    assert check["failed_count"] == 1
    assert check["failed_segment_ids"] == ["s1"]


def test_alignment_health_check_warns_for_one_ms_clusters(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "s1",
                "status": "completed",
                "global_start_time": 0.0,
                "global_end_time": 0.004,
                "tokens": [
                    {"text": "a", "start_time": 0.0, "end_time": 0.001},
                    {"text": "b", "start_time": 0.001, "end_time": 0.002},
                    {"text": "c", "start_time": 0.002, "end_time": 0.003},
                ],
            }
        ],
    )

    check = alignment_health_check(paths)

    assert check["status"] == "WARN"
    assert check["one_ms_cluster_count"] == 1


def test_alignment_health_check_warns_for_completed_coarse(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "s1",
                "status": "completed",
                "alignment_state": "completed_coarse",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "tokens": [],
            }
        ],
    )

    check = alignment_health_check(paths)

    assert check["status"] == "WARN"
    assert check["completed_exact_count"] == 0
    assert check["completed_coarse_count"] == 1


def test_alignment_health_excludes_failed_music_region(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [{"segment_id": "op", "status": "failed", "global_start_time": 1.0, "global_end_time": 2.0}],
    )
    report = paths.workdir / "reports" / "music.json"
    report.parent.mkdir(parents=True)
    write_json_atomic(report, {"intervals": {"op": {"start_ms": 500, "end_ms": 2500}}})

    check = alignment_health_check(paths)

    assert check["status"] == "PASS"
    assert check["failed_count"] == 0
    assert check["skipped_music_region_count"] == 1


def test_alignment_helper_compatibility_aliases() -> None:
    tokens = [
        SimpleToken(text="a", start_time=0.0, end_time=0.4),
        SimpleToken(text="b", start_time=0.4, end_time=0.401),
    ]
    legacy_tokens = [
        _SimpleToken(text="a", start_time=0.0, end_time=0.4),
        _SimpleToken(text="b", start_time=0.4, end_time=0.401),
    ]

    assert float_or_none("1.5") == 1.5
    assert _float_or_none("bad") is None
    assert alignment_coverage(tokens, 0.0, 1.0) == _alignment_coverage(legacy_tokens, 0.0, 1.0)
    assert one_ms_token_stats(tokens) == _one_ms_token_stats(legacy_tokens)
