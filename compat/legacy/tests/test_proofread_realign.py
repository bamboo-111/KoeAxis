from __future__ import annotations

import argparse
from pathlib import Path
import wave

from qwen_asr.models import AlignedSegment, AlignedToken, WorkPaths
from qwen_asr.proofread_realign import has_unrealigned_proofread_changes, run_proofread_realign_stage
from qwen_asr.storage import read_json, write_json_atomic


class FakeAligner:
    def __init__(self, *_args, **_kwargs) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.loaded = False

    def run_segment(self, transcript, cleanup: bool = False):  # noqa: ANN001, ARG002
        return AlignedSegment(
            segment_id=transcript.segment_id,
            audio_path=transcript.audio_path,
            global_start_time=transcript.global_start_time,
            global_end_time=transcript.global_end_time,
            text=transcript.text,
            language=transcript.language,
            tokens=[
                AlignedToken(text=transcript.text, start_time=1.05, end_time=1.55),
            ],
            status="completed",
        )


class FailingAligner(FakeAligner):
    def run_segment(self, transcript, cleanup: bool = False):  # noqa: ANN001, ARG002
        return AlignedSegment(
            segment_id=transcript.segment_id,
            audio_path=transcript.audio_path,
            global_start_time=transcript.global_start_time,
            global_end_time=transcript.global_end_time,
            text=transcript.text,
            status="failed",
            error="forced failure",
        )


class MismatchAligner(FakeAligner):
    def run_segment(self, transcript, cleanup: bool = False):  # noqa: ANN001, ARG002
        return AlignedSegment(
            segment_id=transcript.segment_id,
            audio_path=transcript.audio_path,
            global_start_time=transcript.global_start_time,
            global_end_time=transcript.global_end_time,
            text=transcript.text,
            language=transcript.language,
            tokens=[
                AlignedToken(text="\u3044\u3044\u3048", start_time=1.05, end_time=1.55),
            ],
            status="completed",
        )


class OverlapAligner(FakeAligner):
    def run_segment(self, transcript, cleanup: bool = False):  # noqa: ANN001, ARG002
        return AlignedSegment(
            segment_id=transcript.segment_id,
            audio_path=transcript.audio_path,
            global_start_time=transcript.global_start_time,
            global_end_time=transcript.global_end_time,
            text=transcript.text,
            language=transcript.language,
            tokens=[
                AlignedToken(text=transcript.text, start_time=1.05, end_time=1.90),
            ],
            status="completed",
        )


class ShortOverlapAligner(FakeAligner):
    def run_segment(self, transcript, cleanup: bool = False):  # noqa: ANN001, ARG002
        return AlignedSegment(
            segment_id=transcript.segment_id,
            audio_path=transcript.audio_path,
            global_start_time=transcript.global_start_time,
            global_end_time=transcript.global_end_time,
            text=transcript.text,
            language=transcript.language,
            tokens=[
                AlignedToken(text=transcript.text, start_time=0.90, end_time=1.53),
            ],
            status="completed",
        )


class LongCoverageAligner(FakeAligner):
    def run_segment(self, transcript, cleanup: bool = False):  # noqa: ANN001, ARG002
        return AlignedSegment(
            segment_id=transcript.segment_id,
            audio_path=transcript.audio_path,
            global_start_time=transcript.global_start_time,
            global_end_time=transcript.global_end_time,
            text=transcript.text,
            language=transcript.language,
            tokens=[
                AlignedToken(text=transcript.text, start_time=0.80, end_time=9.40),
            ],
            status="completed",
        )


class ExplodingAligner(FakeAligner):
    def load(self) -> None:
        raise AssertionError("qwen aligner should not be loaded")


def test_proofread_realign_updates_timing_and_clears_gate(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "0": {
                "start_time": 500,
                "end_time": 1050,
                "original_subtitle": "\u3042\u306e",
                "needs_realign": False,
                "realign_status": "completed",
            },
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=FakeAligner)

    assert report["status"] == "PASS"
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["start_time"] == 1050
    assert manifest["1"]["end_time"] == 1550
    assert manifest["1"]["needs_realign"] is False
    assert manifest["1"]["realign_status"] == "completed"
    assert manifest["1"]["realign_tokens"][0]["text"] == "\u306f\u3044"
    assert has_unrealigned_proofread_changes(paths) is False
    assert (tmp_path / "reports" / "proofread_realign.json").exists()


def test_proofread_realign_skips_punctuation_only_original_change(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u30ba\u30e0\u30eb\u30c9",
                "translated_subtitle": "\u7956\u59c6\u9c81\u5fb7",
                "needs_realign": True,
                "realign_status": "pending",
                "proofread_history": [
                    {
                        "source": "mimo-nearby-audio",
                        "changes": {
                            "original_subtitle": {
                                "before": "\u30ba\u30e0\u30eb\u30c9\u3001",
                                "after": "\u30ba\u30e0\u30eb\u30c9",
                            }
                        },
                    }
                ],
            }
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=ExplodingAligner)

    assert report["status"] == "PASS"
    assert report["candidate_count"] == 0
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["needs_realign"] is False
    assert manifest["1"]["realign_status"] == "completed"
    assert manifest["1"]["realign_method"] == "punctuation-only"
    assert has_unrealigned_proofread_changes(paths) is False


def test_proofread_realign_keeps_long_mixed_language_text_on_original_timing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 9000,
                "original_subtitle": (
                    "Cause I never see running free since my"
                    "\u4e16\u754c\u96e2\u308c\u3066\u611b\u3055\u308c\u305f\u304b\u3063\u305f"
                    " blue heart will shine tomorrow \u541b\u3060\u3051\u306e\u5149\u3060\u304b\u3089"
                ),
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=ExplodingAligner)

    assert report["status"] == "WARN"
    assert report["fallback_count"] == 1
    assert report["items"][0]["method"] == "mixed-language-original-timing"
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["start_time"] == 1000
    assert manifest["1"]["end_time"] == 9000
    assert manifest["1"]["needs_realign"] is False
    assert manifest["1"]["realign_method"] == "mixed-language-original-timing"
    assert manifest["1"]["realign_warning"] == "mixed-language-original-timing"
    assert has_unrealigned_proofread_changes(paths) is False


def test_proofread_realign_accepts_custom_manifest_and_outputs(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(paths.mimo_proofread_manifest, {})
    custom_manifest = tmp_path / "experiments" / "mimo-proofread-expanded" / "mimo_proofread_segments.json"
    custom_report = tmp_path / "reports" / "proofread_realign.expanded.json"
    custom_diagnostics = tmp_path / "diagnostics" / "proofread-realign-expanded"
    write_json_atomic(
        custom_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )
    args = _args()
    args.proofread_realign_manifest = str(custom_manifest)
    args.proofread_realign_report_output = str(custom_report)
    args.proofread_realign_diagnostics_dir = str(custom_diagnostics)

    report = run_proofread_realign_stage(args, paths, aligner_factory=FakeAligner)

    assert report["status"] == "PASS"
    assert report["manifest_path"] == str(custom_manifest)
    assert report["diagnostics_dir"] == str(custom_diagnostics)
    assert custom_report.exists()
    assert (custom_diagnostics / "1.wav").exists()
    manifest = read_json(custom_manifest)
    assert manifest["1"]["start_time"] == 1050
    assert manifest["1"]["realign_status"] == "completed"
    assert read_json(paths.mimo_proofread_manifest) == {}


def test_proofread_realign_falls_back_to_original_timing_when_alignment_fails(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "0": {
                "start_time": 500,
                "end_time": 1050,
                "original_subtitle": "\u3042\u306e",
                "needs_realign": False,
                "realign_status": "completed",
            },
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=FailingAligner)

    assert report["status"] == "WARN"
    assert report["fallback_count"] == 1
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["needs_realign"] is False
    assert manifest["1"]["realign_status"] == "completed"
    assert manifest["1"]["realign_method"] == "original-timing"
    assert manifest["1"]["realign_warning"] == "forced failure"
    assert manifest["1"]["start_time"] == 1000
    assert manifest["1"]["end_time"] == 1600


def test_proofread_realign_original_timing_primary_skips_aligner(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )
    args = _args()
    args.proofread_realign_primary = "original-timing"

    report = run_proofread_realign_stage(args, paths, aligner_factory=ExplodingAligner)

    assert report["status"] == "WARN"
    assert report["fallback_count"] == 1
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["start_time"] == 1000
    assert manifest["1"]["end_time"] == 1600
    assert manifest["1"]["needs_realign"] is False
    assert manifest["1"]["realign_status"] == "completed"
    assert manifest["1"]["realign_method"] == "original-timing"
    assert manifest["1"]["realign_warning"] == "explicit-original-timing-primary"


def test_proofread_realign_original_timing_primary_does_not_require_audio(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )
    args = _args()
    args.proofread_realign_primary = "original-timing"

    report = run_proofread_realign_stage(args, paths, aligner_factory=ExplodingAligner)

    assert report["status"] == "WARN"
    assert report["fallback_count"] == 1
    assert not paths.audio_path.exists()
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_status"] == "completed"


def test_proofread_realign_falls_back_when_qwen_content_mismatches(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=MismatchAligner)

    assert report["status"] == "WARN"
    assert report["fallback_count"] == 1
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_method"] == "original-timing"
    assert manifest["1"]["realign_warning"].startswith("qwen-content-mismatch")
    assert manifest["1"]["start_time"] == 1000
    assert manifest["1"]["end_time"] == 1600


def test_proofread_realign_clamps_severe_neighbor_overlap_when_candidate_remains_legal(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            },
            "2": {
                "start_time": 1650,
                "end_time": 2200,
                "original_subtitle": "\u3044\u3044\u3048",
                "needs_realign": False,
                "realign_status": "completed",
            },
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=OverlapAligner)

    assert report["status"] == "PASS"
    assert report["fallback_count"] == 0
    assert report["completed_count"] == 1
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_method"] == "qwen-clamped"
    assert manifest["1"]["realign_warning"] == "qwen timing clamped to neighbor boundaries: severe-neighbor-overlap"
    assert manifest["1"]["start_time"] == 1050
    assert manifest["1"]["end_time"] == 1650
    assert manifest["2"]["start_time"] == 1650
    assert manifest["2"]["end_time"] == 2200
    row = report["items"][0]
    assert row["method"] == "qwen-clamped"
    assert row["timing_guard"]["reason"] == "timing-clamped-to-neighbors"


def test_proofread_realign_falls_back_when_clamped_overlap_is_too_short(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            },
            "2": {
                "start_time": 1120,
                "end_time": 2200,
                "original_subtitle": "\u3044\u3044\u3048",
                "needs_realign": False,
                "realign_status": "completed",
            },
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=OverlapAligner)

    assert report["status"] == "WARN"
    assert report["fallback_count"] == 1
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_method"] == "original-timing"
    assert manifest["1"]["realign_warning"] == "severe-neighbor-overlap"
    assert manifest["1"]["start_time"] == 1000
    assert manifest["1"]["end_time"] == 1600
    assert manifest["2"]["start_time"] == 1120
    assert manifest["2"]["end_time"] == 2200


def test_proofread_realign_pads_clamped_overlap_to_readability_floor(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "0": {
                "start_time": 500,
                "end_time": 1050,
                "original_subtitle": "\u3042\u306e",
                "needs_realign": False,
                "realign_status": "completed",
            },
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            },
            "2": {
                "start_time": 2200,
                "end_time": 2600,
                "original_subtitle": "\u3044\u3044\u3048",
                "needs_realign": False,
                "realign_status": "completed",
            },
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=ShortOverlapAligner)

    assert report["status"] == "PASS"
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_method"] == "qwen-clamped"
    assert manifest["1"]["start_time"] == 1050
    assert manifest["1"]["end_time"] == 1550
    assert manifest["0"]["end_time"] == 1050
    assert manifest["2"]["start_time"] == 2200


def test_proofread_realign_clamps_overlong_token_range_to_original_window(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=12000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 9000,
                "original_subtitle": "\u306d\u3048\u306d\u3048",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )

    report = run_proofread_realign_stage(_args(), paths, aligner_factory=LongCoverageAligner)

    assert report["status"] == "PASS"
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_method"] == "qwen-clamped"
    assert manifest["1"]["realign_warning"] == "qwen timing clamped to original display window: overlong-duration"
    assert manifest["1"]["start_time"] == 1000
    assert manifest["1"]["end_time"] == 9000
    row = report["items"][0]
    assert row["timing_guard"]["duration_clamp"]["reason"] == "display-clamped-to-original-window"


def test_proofread_realign_uses_mfa_local_when_qwen_alignment_fails(tmp_path: Path, monkeypatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )
    monkeypatch.setattr(
        "qwen_asr.proofread_realign.detect_mfa_environment",
        lambda run_version_check=True: {"available": True, "command": ["mfa"], "root_dir": ""},
    )
    monkeypatch.setattr(
        "qwen_asr.proofread_realign.run_local_mfa_alignment_experiments",
        lambda *args, **kwargs: [
            {
                "status": "completed",
                "usable": True,
                "clip": "clip.wav",
                "lab": "clip.lab",
                "lab_text": "\u306f\u3044",
                "lab_text_source": "candidate",
                "elapsed_ms": 123,
                "word_quality": {"usable": True, "known_timed_count": 1},
                "global_word_ranges": [
                    {"start_ms": 1100, "end_ms": 1400, "text": "\u306f\u3044"},
                ],
            }
        ],
    )
    args = _args()
    args.proofread_realign_mfa_fallback = "local"

    report = run_proofread_realign_stage(args, paths, aligner_factory=FailingAligner)

    assert report["status"] == "PASS"
    assert report["mfa_completed_count"] == 1
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["start_time"] == 1100
    assert manifest["1"]["end_time"] == 1400
    assert manifest["1"]["realign_method"] == "mfa-local"
    assert manifest["1"]["realign_tokens"][0]["text"] == "\u306f\u3044"


def test_proofread_realign_records_unusable_mfa_before_original_timing_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )
    monkeypatch.setattr(
        "qwen_asr.proofread_realign.detect_mfa_environment",
        lambda run_version_check=True: {"available": True, "command": ["mfa"], "root_dir": ""},
    )
    monkeypatch.setattr(
        "qwen_asr.proofread_realign.run_local_mfa_alignment_experiments",
        lambda *args, **kwargs: [
            {
                "status": "completed",
                "usable": False,
                "reason": "mfa-output-unusable",
                "word_quality": {"usable": False, "unknown_count": 1},
                "global_word_ranges": [
                    {"start_ms": 1100, "end_ms": 1110, "text": "<unk>"},
                ],
            }
        ],
    )
    args = _args()
    args.proofread_realign_mfa_fallback = "local"

    report = run_proofread_realign_stage(args, paths, aligner_factory=FailingAligner)

    assert report["status"] == "WARN"
    assert report["fallback_count"] == 1
    assert report["mfa_unusable_count"] == 1
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_method"] == "original-timing"
    assert manifest["1"]["realign_mfa_status"] == "unusable"
    assert manifest["1"]["start_time"] == 1000
    assert manifest["1"]["end_time"] == 1600


def test_proofread_realign_rejects_mfa_when_recognized_text_differs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )
    monkeypatch.setattr(
        "qwen_asr.proofread_realign.detect_mfa_environment",
        lambda run_version_check=True: {"available": True, "command": ["mfa"], "root_dir": ""},
    )
    monkeypatch.setattr(
        "qwen_asr.proofread_realign.run_local_mfa_alignment_experiments",
        lambda *args, **kwargs: [
            {
                "status": "completed",
                "usable": True,
                "word_quality": {"usable": True, "known_timed_count": 1},
                "global_word_ranges": [
                    {"start_ms": 1100, "end_ms": 1400, "text": "\u3058\u3083\u306d"},
                ],
            }
        ],
    )
    args = _args()
    args.proofread_realign_mfa_fallback = "local"

    report = run_proofread_realign_stage(args, paths, aligner_factory=FailingAligner)

    assert report["status"] == "WARN"
    assert report["fallback_count"] == 1
    assert report["mfa_rejected_count"] == 1
    row = report["items"][0]
    assert row["mfa_status"] == "rejected"
    assert row["mfa_reason"] == "mfa-content-mismatch"
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_method"] == "original-timing"
    assert manifest["1"]["realign_mfa_status"] == "rejected"
    assert manifest["1"]["realign_mfa_reason"] == "mfa-content-mismatch"
    assert manifest["1"]["start_time"] == 1000
    assert manifest["1"]["end_time"] == 1600


def test_proofread_realign_mfa_primary_retries_original_timing_with_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=4000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": False,
                "realign_status": "completed",
                "realign_method": "original-timing",
            },
            "2": {
                "start_time": 2000,
                "end_time": 2600,
                "original_subtitle": "\u3044\u3044\u3048",
                "needs_realign": False,
                "realign_status": "completed",
                "realign_method": "original-timing",
            },
        },
    )
    monkeypatch.setattr(
        "qwen_asr.proofread_realign.detect_mfa_environment",
        lambda run_version_check=True: {"available": True, "command": ["mfa"], "root_dir": ""},
    )
    monkeypatch.setattr(
        "qwen_asr.proofread_realign.run_local_mfa_alignment_experiments",
        lambda *args, **kwargs: [
            {
                "status": "completed",
                "usable": True,
                "word_quality": {"usable": True, "known_timed_count": 1},
                "global_word_ranges": [
                    {"start_ms": 1100, "end_ms": 1400, "text": "\u306f\u3044"},
                ],
            }
        ],
    )
    args = _args()
    args.proofread_realign_primary = "mfa-local"
    args.proofread_realign_retry_method = "original-timing"
    args.proofread_realign_max_items = 1

    report = run_proofread_realign_stage(args, paths, aligner_factory=ExplodingAligner)

    assert report["status"] == "PASS"
    assert report["candidate_count"] == 2
    assert report["pending_count"] == 1
    assert report["mfa_completed_count"] == 1
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["realign_method"] == "mfa-local"
    assert manifest["1"]["start_time"] == 1100
    assert manifest["2"]["realign_method"] == "original-timing"


def test_proofread_realign_can_fail_alignment_in_strict_mode(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    _write_silent_wav(paths.audio_path, duration_ms=3000)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1600,
                "original_subtitle": "\u306f\u3044",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )

    args = _args()
    args.proofread_realign_fallback = "fail"
    report = run_proofread_realign_stage(args, paths, aligner_factory=FailingAligner)

    assert report["status"] == "FAIL"
    manifest = read_json(paths.mimo_proofread_manifest)
    assert manifest["1"]["needs_realign"] is True
    assert manifest["1"]["realign_status"] == "failed"
    assert manifest["1"]["realign_error"] == "forced failure"


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        proofread_realign_model="fake-aligner",
        proofread_realign_padding_ms=500,
        proofread_realign_language="Japanese",
        proofread_realign_fallback="original-timing",
        proofread_realign_mfa_fallback="off",
        proofread_realign_mfa_padding_ms=700,
        proofread_realign_mfa_min_content_score=0.70,
        proofread_realign_primary="qwen-first",
        proofread_realign_retry_method="none",
        proofread_realign_max_items=0,
        proofread_realign_manifest="",
        proofread_realign_diagnostics_dir="",
        proofread_realign_report_output="",
        dtype="fp16",
        device="cpu",
        attn_implementation=None,
        keep_raw_model_output=False,
        model_cache_dir=None,
        local_files_only=True,
    )


def _write_silent_wav(path: Path, *, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_rate = 16000
    frames = int(frame_rate * duration_ms / 1000)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(b"\x00\x00" * frames)
