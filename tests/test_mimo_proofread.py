from __future__ import annotations

import json
from pathlib import Path
import wave

import qwen_asr.mimo_proofread as mimo_proofread
from qwen_asr.mimo_proofread import (
    _apply_branch_updates,
    _ass_acceptance_guard,
    _build_manifest_suspect_report,
    _collect_stage1_suspects,
    _load_pipeline_inputs,
    _normalize_qa_item,
    _original_content_deletion_guard,
    _original_high_risk_replacement_guard,
    _original_no_ass_substantial_rewrite_guard,
    _process_stage1_text_task,
    _process_stage2_nearby_audio_batch_task,
    _request_suggestions_with_parse_retries,
    _safe_suggestion_value,
    _translation_shortening_guard,
    _translated_manifest_has_suspect_metadata,
    _write_two_stage_outputs,
    _write_nearby_audio_clip,
    MiMoConfig,
    SegmentTask,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_normalize_qa_item_accepts_legacy_suspected_original() -> None:
    normalized = _normalize_qa_item(
        {
            "id": "4",
            "asr_suspect": True,
            "suspected_original": "corrected source",
            "suggested_translation": "corrected translation",
            "confidence": 0.95,
        }
    )

    assert normalized["suggested_original"] == "corrected source"
    assert normalized["suggested_translation"] == "corrected translation"


def test_normalize_qa_item_drops_placeholder_suggestions() -> None:
    normalized = _normalize_qa_item(
        {
            "id": "4",
            "suggested_original": "None",
            "suggested_translation": "null",
            "confidence": 0.95,
        }
    )

    assert normalized["suggested_original"] == ""
    assert normalized["suggested_translation"] == ""


def test_safe_suggestion_rejects_ascii_translation_for_cjk_current() -> None:
    assert _safe_suggestion_value("I beg you", "\u4e0d\u884c\u7684", field="translated_subtitle") == ""
    assert _safe_suggestion_value("\u62dc\u6258\u4f60", "\u4e0d\u884c\u7684", field="translated_subtitle") == "\u62dc\u6258\u4f60"


def test_safe_suggestion_rejects_non_japanese_original_for_japanese_current() -> None:
    assert _safe_suggestion_value("None", "\u306f\u3044", field="original_subtitle") == ""
    assert _safe_suggestion_value("hello", "\u306f\u3044", field="original_subtitle") == ""
    assert _safe_suggestion_value("\u306f\u3044", "\u3042", field="original_subtitle") == "\u306f\u3044"


def test_ass_acceptance_guard_rejects_original_that_does_not_improve_reference_match() -> None:
    item = {
        "suspect_reason": (
            "ASS stage diff suspect: ass_stage_became_fail "
            "transition=aligned -> split issue=became-fail ass_index=117 ass_text=\u3060\u306a \u6b63\u89e3\u3060"
        )
    }

    decision = _ass_acceptance_guard(
        item,
        current_original="\u65e9\u3044\u3057",
        suggested_original="\u30d0\u30b9\u30bf\u30a2\u304c\u65e9\u3044\u3057",
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "ass-score-not-improved"


def test_ass_acceptance_guard_accepts_original_that_improves_reference_match() -> None:
    item = {
        "suspect_reason": (
            "ASS stage diff suspect: ass_stage_became_fail "
            "transition=aligned -> split issue=became-fail ass_index=48 ass_text=\u306f\u3044"
        )
    }

    decision = _ass_acceptance_guard(
        item,
        current_original="\u3042",
        suggested_original="\u306f\u3044",
    )

    assert decision["accepted"] is True
    assert decision["suggested_score"] > decision["current_score"]


def test_ass_acceptance_guard_rejects_long_reference_fragment_replacement() -> None:
    item = {
        "suspect_reason": (
            "ASS low score suspect: ass_index=79 "
            "ass_text=\u79c1 \u5965\u3055\u307e\u3092\u304a\u5b88\u308a\u3059\u308b\u3063\u3066\u7d04\u675f\u3057\u305f\u306e\u306b\u2026"
        )
    }

    decision = _ass_acceptance_guard(
        item,
        current_original="\u7d04\u675f\u3057\u306a\u3044\u5965\u69d8\u304c",
        suggested_original="\u7d04\u675f\u3057\u305f\u306e\u306b",
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "ass-long-reference-fragment-replacement"
    assert decision["suggested_score"] > decision["current_score"]


def test_original_content_deletion_guard_rejects_unique_text_without_ass_reference() -> None:
    decision = _original_content_deletion_guard(
        current_original="\u305d\u3093\u306a\u30ba\u30e0\u30eb\u30c9",
        suggested_original="\u30ba\u30e0\u30eb\u30c9\u3055\u3093\uff01",
        ass_guard={"accepted": True, "reason": "no-ass-reference"},
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "original-content-dropped-without-ass-reference"


def test_original_content_deletion_guard_allows_ass_backed_improvement() -> None:
    decision = _original_content_deletion_guard(
        current_original="\u305d\u3093\u306a\u30ba\u30e0\u30eb\u30c9",
        suggested_original="\u30ba\u30e0\u30eb\u30c9\u3055\u3093\uff01",
        ass_guard={"accepted": True, "reason": "ass-improved"},
    )

    assert decision["accepted"] is True
    assert decision["reason"] == "ass-reference-accepted"


def test_original_high_risk_replacement_guard_rejects_short_response_expansion_without_ass_reference() -> None:
    decision = _original_high_risk_replacement_guard(
        current_original="\u3046\u3093\u3002",
        suggested_original=(
            "\u6211\u3089\u304c\u5e1d\u5b50\u8ecd\u304c\u904a\u7267\u6c11\u306b\u6557\u308c\u305f\u3002"
            "\u8ecd\u306f\u58ca\u6ec5\u3057\u305f\u3002"
        ),
        ass_guard={"accepted": True, "reason": "no-ass-reference"},
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "protected-short-response-replaced-without-ass-reference"


def test_original_high_risk_replacement_guard_rejects_protected_short_replacement_without_ass_reference() -> None:
    decision = _original_high_risk_replacement_guard(
        current_original="\u3046\u3093\u3002",
        suggested_original="\u5965\u69d8\u3002",
        ass_guard={"accepted": True, "reason": "no-ass-reference"},
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "protected-short-response-replaced-without-ass-reference"


def test_original_no_ass_substantial_rewrite_guard_rejects_large_unbacked_rewrite() -> None:
    decision = _original_no_ass_substantial_rewrite_guard(
        current_original="\u6709\u540d",
        suggested_original="\u304a\u5024\u6bb5\u9ad8\u3044",
        ass_guard={"accepted": True, "reason": "no-ass-reference"},
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "no-ass-reference-original-change"


def test_original_no_ass_substantial_rewrite_guard_rejects_small_particle_fix() -> None:
    decision = _original_no_ass_substantial_rewrite_guard(
        current_original="\u3042\u306a\u305f\u305f\u3061\u306f",
        suggested_original="\u3042\u306a\u305f\u305f\u3061\u304c",
        ass_guard={"accepted": True, "reason": "no-ass-reference"},
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "no-ass-reference-original-change"


def test_translation_shortening_guard_rejects_abnormal_shortening() -> None:
    decision = _translation_shortening_guard(
        current_translation="\u8fd9\u662f\u4e00\u53e5\u5f88\u957f\u7684\u7ffb\u8bd1",
        suggested_translation="\u597d",
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "translation-abnormally-shortened"


def test_two_stage_report_sums_rejected_audio_suggestions(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.json"
    stage1 = tmp_path / "stage1.json"
    stage2 = tmp_path / "stage2.json"
    srt = tmp_path / "out.srt"

    _write_two_stage_outputs(
        manifest,
        report,
        stage1,
        stage2,
        srt,
        {"1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u306f\u3044"}},
        [{"status": "completed", "suspect_ids": ["1"]}],
        [{"status": "completed", "id": "1", "applied_count": 0, "rejected_count": 2}],
        started=None,
        translated={"1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u306f\u3044"}},
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["stage2_completed"] == 1
    assert payload["stage2_completed_batches"] == 1
    assert payload["audio_review_rejected_count"] == 2


def test_stage1_text_task_marks_suspects_but_does_not_apply_text_updates(monkeypatch) -> None:
    def fake_request(request, **_kwargs):
        request()
        return json.dumps(
            [
                {
                    "id": "1",
                    "error_type": "translation_error",
                    "original": "\u306f\u3044",
                    "translation": "\u662f",
                    "suggested_translation": "\u597d\u7684",
                    "asr_suspect": False,
                    "needs_audio_review": False,
                    "reason": "translation polish",
                    "confidence": 1.0,
                },
                {
                    "id": "2",
                    "error_type": "asr_suspect",
                    "original": "\u3042",
                    "translation": "\u554a",
                    "suggested_original": "\u306f\u3044",
                    "suggested_translation": "\u597d\u7684",
                    "asr_suspect": True,
                    "needs_audio_review": True,
                    "reason": "audio needed",
                    "confidence": 0.4,
                },
            ],
            ensure_ascii=False,
        ), {}, [
            {
                "id": "1",
                "error_type": "translation_error",
                "original": "\u306f\u3044",
                "translation": "\u662f",
                "suggested_translation": "\u597d\u7684",
                "asr_suspect": False,
                "needs_audio_review": False,
                "reason": "translation polish",
                "confidence": 1.0,
            },
            {
                "id": "2",
                "error_type": "asr_suspect",
                "original": "\u3042",
                "translation": "\u554a",
                "suggested_original": "\u306f\u3044",
                "suggested_translation": "\u597d\u7684",
                "asr_suspect": True,
                "needs_audio_review": True,
                "reason": "audio needed",
                "confidence": 0.4,
            },
        ]

    def fake_call(**_kwargs):
        return "[]", {}

    monkeypatch.setattr(mimo_proofread, "_request_suggestions_with_parse_retries", fake_request)
    monkeypatch.setattr(mimo_proofread, "_call_mimo_text_stage1_with_retries", fake_call)

    task = SegmentTask(
        index=1,
        total=1,
        segment={"segment_id": "segment_000001"},
        subtitle_entries={
            "1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"},
            "2": {"original_subtitle": "\u3042", "translated_subtitle": "\u554a"},
        },
        glossary_entries=[],
        audio_path=Path("unused.wav"),
    )

    result = _process_stage1_text_task(
        task=task,
        client=object(),
        config=_config(),
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        keep_raw=False,
        suspect_confidence_threshold=0.75,
        apply_confidence_threshold=0.8,
    )

    assert result.updates == {}
    assert result.report_item["applied_count"] == 0
    assert result.report_item["stage1_text_updates_disabled"] is True
    assert result.report_item["suspect_ids"] == ["2"]


def test_apply_branch_updates_changes_both_languages_and_keeps_evidence() -> None:
    branch = {
        "4": {
            "original_subtitle": "wrong source",
            "translated_subtitle": "wrong translation",
        }
    }

    applied = _apply_branch_updates(
        branch,
        {
            "4": {
                "original_subtitle": "correct source",
                "translated_subtitle": "correct translation",
                "__proofread_evidence": {
                    "confidence": 0.95,
                    "ass_guard": {"accepted": True, "reason": "ass-improved"},
                },
            }
        },
        source="mimo-nearby-audio",
    )

    assert applied == 1
    assert branch["4"]["original_subtitle"] == "correct source"
    assert branch["4"]["translated_subtitle"] == "correct translation"
    assert branch["4"]["needs_realign"] is True
    assert branch["4"]["realign_status"] == "pending"
    assert branch["4"]["proofread_history"] == [
        {
            "source": "mimo-nearby-audio",
            "changes": {
                "original_subtitle": {"before": "wrong source", "after": "correct source"},
                "translated_subtitle": {"before": "wrong translation", "after": "correct translation"},
            },
            "evidence": {
                "confidence": 0.95,
                "ass_guard": {"accepted": True, "reason": "ass-improved"},
            },
        }
    ]
    assert "__proofread_evidence" not in branch["4"]


def test_stage2_batch_records_applied_ass_guard_evidence(monkeypatch, tmp_path: Path) -> None:
    def fake_clip(**kwargs):
        clip_path = tmp_path / ("clip-" + kwargs["subtitle_id"] + ".wav")
        clip_path.write_bytes(b"fake")
        return clip_path, {"duration_s": 1.0}

    def fake_request(request, **_kwargs):
        request()
        return json.dumps(
            [
                {
                    "id": "1",
                    "error_type": "asr_suspect",
                    "original": "\u3042",
                    "translation": "\u554a",
                    "suggested_original": "\u306f\u3044",
                    "suggested_translation": "\u597d\u7684",
                    "asr_suspect": True,
                    "needs_audio_review": False,
                    "reason": "audio clear",
                    "confidence": 0.95,
                }
            ],
            ensure_ascii=False,
        ), {}, [
            {
                "id": "1",
                "error_type": "asr_suspect",
                "original": "\u3042",
                "translation": "\u554a",
                "suggested_original": "\u306f\u3044",
                "suggested_translation": "\u597d\u7684",
                "asr_suspect": True,
                "needs_audio_review": False,
                "reason": "audio clear",
                "confidence": 0.95,
            }
        ]

    def fake_call(**_kwargs):
        return "[]", {}

    monkeypatch.setattr(mimo_proofread, "_write_nearby_audio_clip", fake_clip)
    monkeypatch.setattr(mimo_proofread, "_request_suggestions_with_parse_retries", fake_request)
    monkeypatch.setattr(mimo_proofread, "_call_mimo_nearby_audio_with_retries", fake_call)

    translated = {
        "1": {
            "start_time": 1000,
            "end_time": 1200,
            "original_subtitle": "\u3042",
            "translated_subtitle": "\u554a",
            "suspect_reason": "ASS stage diff suspect: ass_text=\u306f\u3044",
        }
    }
    segment = {
        "segment_id": "s1",
        "audio_path": str(tmp_path / "s1.wav"),
        "source_audio_path": "",
        "global_start_time": 0.0,
        "global_end_time": 2.0,
    }

    [result] = _process_stage2_nearby_audio_batch_task(
        target_ids=["1"],
        client=object(),
        config=_config(),
        segments=[segment],
        translated=translated,
        glossary=[],
        clips_dir=tmp_path,
        context_subtitles=0,
        padding_s=0.0,
        max_glossary_entries=0,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        keep_raw=False,
        apply_confidence_threshold=0.9,
    )

    evidence = result.updates["1"]["__proofread_evidence"]
    assert evidence["ass_guard"]["accepted"] is True
    assert evidence["ass_guard"]["suggested_score"] > evidence["ass_guard"]["current_score"]
    assert result.report_item["applied_evidence"]["1"]["ass_guard"]["reason"] in {
        "ass-high-score",
        "ass-improved",
    }


def test_stage2_batch_rejects_original_content_deletion_without_ass_reference(monkeypatch, tmp_path: Path) -> None:
    def fake_clip(**kwargs):
        clip_path = tmp_path / ("clip-" + kwargs["subtitle_id"] + ".wav")
        clip_path.write_bytes(b"fake")
        return clip_path, {"duration_s": 1.0}

    def fake_request(request, **_kwargs):
        request()
        suggestion = {
            "id": "1",
            "error_type": "asr_suspect",
            "original": "\u305d\u3093\u306a\u30ba\u30e0\u30eb\u30c9",
            "translation": "\u90a3\u6837\u7684\u7956\u7a46\u5c14\u5fb7",
            "suggested_original": "\u30ba\u30e0\u30eb\u30c9\u3055\u3093\uff01",
            "suggested_translation": "\u7956\u7a46\u5c14\u5fb7\u5148\u751f",
            "asr_suspect": True,
            "needs_audio_review": False,
            "reason": "audio clear",
            "confidence": 0.95,
        }
        return json.dumps([suggestion], ensure_ascii=False), {}, [suggestion]

    def fake_call(**_kwargs):
        return "[]", {}

    monkeypatch.setattr(mimo_proofread, "_write_nearby_audio_clip", fake_clip)
    monkeypatch.setattr(mimo_proofread, "_request_suggestions_with_parse_retries", fake_request)
    monkeypatch.setattr(mimo_proofread, "_call_mimo_nearby_audio_with_retries", fake_call)

    translated = {
        "1": {
            "start_time": 1000,
            "end_time": 1200,
            "original_subtitle": "\u305d\u3093\u306a\u30ba\u30e0\u30eb\u30c9",
            "translated_subtitle": "\u90a3\u6837\u7684\u7956\u7a46\u5c14\u5fb7",
            "suspect_reason": "manifest suspect without ass_text",
        }
    }
    segment = {
        "segment_id": "s1",
        "audio_path": str(tmp_path / "s1.wav"),
        "source_audio_path": "",
        "global_start_time": 0.0,
        "global_end_time": 2.0,
    }

    [result] = _process_stage2_nearby_audio_batch_task(
        target_ids=["1"],
        client=object(),
        config=_config(),
        segments=[segment],
        translated=translated,
        glossary=[],
        clips_dir=tmp_path,
        context_subtitles=0,
        padding_s=0.0,
        max_glossary_entries=0,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        keep_raw=False,
        apply_confidence_threshold=0.9,
    )

    assert result.updates == {}
    assert result.report_item["rejections"][0]["reason"] == "original-content-dropped-without-ass-reference"


def test_load_pipeline_inputs_supports_flat_workdir(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.segments_manifest, [{"segment_id": "segment_000001"}])
    write_json_atomic(paths.translated_manifest, {"1": {"original_subtitle": "source"}})

    segments, translated = _load_pipeline_inputs(tmp_path)

    assert segments == [{"segment_id": "segment_000001"}]
    assert translated == {"1": {"original_subtitle": "source"}}


def test_load_pipeline_inputs_rejects_translated_manifest_missing_split_keys(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.segments_manifest, [{"segment_id": "segment_000001"}])
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": ""},
            "2": {"original_subtitle": "\u3046\u3093", "translated_subtitle": ""},
        },
    )
    write_json_atomic(
        paths.translated_manifest,
        {"1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"}},
    )

    try:
        _load_pipeline_inputs(tmp_path)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected incomplete translated manifest to be rejected")

    assert "translated_segments.json is incomplete" in message
    assert "missing_keys=2" in message


def test_load_pipeline_inputs_rejects_blank_translations_for_split_keys(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.segments_manifest, [{"segment_id": "segment_000001"}])
    write_json_atomic(
        paths.split_manifest,
        {"1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": ""}},
    )
    write_json_atomic(
        paths.translated_manifest,
        {"1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": ""}},
    )

    try:
        _load_pipeline_inputs(tmp_path)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected blank translated manifest to be rejected")

    assert "blank_translation_keys=1" in message


def test_manifest_suspect_metadata_routes_audio_review_candidates() -> None:
    translated = {
        "1": {
            "original_subtitle": "ok",
            "translated_subtitle": "ok",
            "asr_suspect": False,
            "needs_audio_review": False,
            "suspect_types": [],
            "suspect_confidence": 0.99,
        },
        "2": {
            "original_subtitle": "bad",
            "translated_subtitle": "bad",
            "asr_suspect": True,
            "needs_audio_review": True,
            "suspect_types": ["fragment"],
            "suspect_reason": "fragment",
            "suspect_confidence": 0.3,
        },
    }

    assert _translated_manifest_has_suspect_metadata(translated)
    report = _build_manifest_suspect_report(translated, confidence_threshold=0.75)

    assert _collect_stage1_suspects(report) == ["2"]
    assert report[0]["source"] == "translation-manifest"


def test_manifest_suspect_metadata_skips_realign_only_timing_shifted() -> None:
    translated = {
        "1": {
            "original_subtitle": "hai",
            "translated_subtitle": "yes",
            "asr_suspect": False,
            "needs_audio_review": False,
            "needs_realign": True,
            "realign_status": "pending",
            "suspect_types": ["ass_short_dialogue_timing_shifted", "ass_fail_score"],
            "suspect_confidence": 1.0,
        },
        "2": {
            "original_subtitle": "bad",
            "translated_subtitle": "bad",
            "asr_suspect": False,
            "needs_audio_review": False,
            "suspect_types": ["ass_short_dialogue_missing", "ass_fail_score"],
            "suspect_confidence": 1.0,
        },
    }

    report = _build_manifest_suspect_report(translated, confidence_threshold=0.75)

    assert _collect_stage1_suspects(report) == ["2"]


def test_request_suggestions_retries_invalid_json(monkeypatch) -> None:
    responses = iter([("", {"attempt": 1}), ("[]", {"attempt": 2})])
    monkeypatch.setattr(mimo_proofread.time, "sleep", lambda _seconds: None)

    content, usage, suggestions = _request_suggestions_with_parse_retries(
        lambda: next(responses),
        max_retries=2,
        base_delay=0.0,
        max_delay=0.0,
    )

    assert content == "[]"
    assert usage == {"attempt": 2}
    assert suggestions == []


def test_request_suggestions_parses_unclosed_json_fence() -> None:
    content = '```json\n[{"id": "125", "suggested_translation": "横。"}]'

    parsed = mimo_proofread._parse_suggestions(content)

    assert parsed == [{"id": "125", "suggested_translation": "横。"}]


def test_nearby_clip_uses_full_source_audio_across_segment_boundary(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    segment_path = tmp_path / "segment.wav"
    clips_dir = tmp_path / "clips"
    for path, seconds in ((source_path, 10), (segment_path, 2)):
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(1000)
            target.writeframes(b"\x00\x00" * (seconds * 1000))
    segment = {
        "audio_path": str(segment_path),
        "source_audio_path": str(source_path),
        "global_start_time": 4.0,
    }
    entries = {"1": {"start_time": 5500, "end_time": 7500}}

    _, meta = _write_nearby_audio_clip(
        subtitle_id="1",
        segment=segment,
        entries=entries,
        audio_path=source_path,
        clips_dir=clips_dir,
        padding_s=0.5,
    )

    assert meta["start_s"] == 5.0
    assert meta["end_s"] == 8.0
    assert meta["duration_s"] == 3.0


def test_stage2_batch_failure_retries_individual_targets(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_clip(**kwargs):
        clip_path = tmp_path / ("clip-" + kwargs["subtitle_id"] + ".wav")
        clip_path.write_bytes(b"fake")
        return clip_path, {"duration_s": 1.0}

    def fake_request(request, **_kwargs):
        request()
        ids = list(mimo_proofread._LAST_TEST_TARGET_IDS)
        calls.append(ids)
        if len(ids) > 1:
            raise ValueError("invalid batch json")
        return "[]", {}, []

    def fake_call(**kwargs):
        mimo_proofread._LAST_TEST_TARGET_IDS = list(kwargs["target_ids"])
        return "[]", {}

    monkeypatch.setattr(mimo_proofread, "_write_nearby_audio_clip", fake_clip)
    monkeypatch.setattr(mimo_proofread, "_request_suggestions_with_parse_retries", fake_request)
    monkeypatch.setattr(mimo_proofread, "_call_mimo_nearby_audio_with_retries", fake_call)
    monkeypatch.setattr(mimo_proofread, "_LAST_TEST_TARGET_IDS", [], raising=False)

    translated = {
        "1": {"start_time": 1000, "end_time": 1200, "original_subtitle": "bad", "translated_subtitle": "bad"},
        "2": {"start_time": 1300, "end_time": 1500, "original_subtitle": "bad", "translated_subtitle": "bad"},
    }
    segment = {
        "segment_id": "s1",
        "audio_path": str(tmp_path / "s1.wav"),
        "source_audio_path": "",
        "global_start_time": 0.0,
        "global_end_time": 2.0,
    }
    config = MiMoConfig(
        base_url="",
        api_key="",
        model="",
        timeout=1.0,
        max_tokens=128,
        temperature=0.0,
        disable_thinking=True,
        extra_body=None,
        compact_output=False,
    )

    results = _process_stage2_nearby_audio_batch_task(
        target_ids=["1", "2"],
        client=object(),
        config=config,
        segments=[segment],
        translated=translated,
        glossary=[],
        clips_dir=tmp_path,
        context_subtitles=0,
        padding_s=0.0,
        max_glossary_entries=0,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        keep_raw=False,
        apply_confidence_threshold=0.75,
    )

    assert calls == [["1", "2"], ["1"], ["2"]]
    assert [item.report_item["status"] for item in results] == ["completed", "completed"]
    assert all(item.report_item["fallback_from_batch"] == ["1", "2"] for item in results)


def _config() -> MiMoConfig:
    return MiMoConfig(
        base_url="",
        api_key="",
        model="",
        timeout=1.0,
        max_tokens=128,
        temperature=0.0,
        disable_thinking=True,
        extra_body=None,
        compact_output=False,
    )
