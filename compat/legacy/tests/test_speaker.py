from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from optimizer.asr_data import ASRDataSeg
from qwen_asr.speaker.assign import SpeakerAssigner, map_tokens_to_cues, vote_labels
from qwen_asr.speaker.voiceprint import (
    SpeakerTrack,
    detect_members,
    frame_scores,
    load_enrollment,
    parse_speakers,
)

MEMBERS = ["理名", "朱李"]
A = [0.7, 0.3]  # clearly 理名
B = [0.3, 0.7]  # clearly 朱李
AMBIG = [0.52, 0.48]  # overlap / boundary: not sure


def _assigner(tokens: list[tuple[str, int, int, list[float]]], min_margin: float = 0.1) -> SpeakerAssigner:
    return SpeakerAssigner(
        members=list(MEMBERS),
        token_texts=[t[0] for t in tokens],
        token_times=[(t[1], t[2]) for t in tokens],
        token_scores=np.array([t[3] for t in tokens], np.float32),
        min_margin=min_margin,
    )


def _new_cue(text: str, start: int, end: int) -> ASRDataSeg:
    return ASRDataSeg(text, start, end)


def test_map_tokens_to_cues_ignores_punctuation_and_width() -> None:
    where = map_tokens_to_cues(["今日", "は", "ＡＢ", "です"], ["今日は、", "AB です。"])
    assert where == [(0, 0), (0, 2), (1, 0), (1, 3)]


def test_vote_labels_merges_unconfident_run_into_neighbour() -> None:
    scores = np.array([A, A, AMBIG, A, A], np.float32)
    assert vote_labels(scores, 0.1) == [0, 0, 0, 0, 0]
    scores = np.array([A, A, B, B], np.float32)
    assert vote_labels(scores, 0.1) == [0, 0, 1, 1]


def test_split_cues_cuts_at_confident_speaker_change_and_keeps_text() -> None:
    assigner = _assigner([
        ("無理", 0, 400, A), ("じゃ", 400, 600, A), ("ねえ", 600, 900, A),
        ("そう", 1000, 1300, B), ("無理", 1300, 1600, B), ("じゃ", 1600, 1800, B), ("ねえ", 1800, 2100, B),
    ])
    cue = ASRDataSeg("無理じゃねえ、そう無理じゃねえ。", 0, 2200)

    result, cuts = assigner.split_cues([cue], _new_cue)

    assert cuts == 1
    assert [(c.text, c.start_time, c.end_time) for c in result] == [
        ("無理じゃねえ、", 0, 900),
        ("そう無理じゃねえ。", 1000, 2200),
    ]
    labels = assigner.label_cues(result)
    assert [label.speaker for label in labels] == ["理名", "朱李"]
    assert all(label.margin >= 0.1 for label in labels)


def test_split_cues_does_not_cut_on_overlap_or_single_member() -> None:
    tokens = [("あの", 0, 300, A), ("うん", 300, 500, AMBIG), ("ね", 500, 700, A)]
    cue = ASRDataSeg("あのうんね", 0, 700)
    result, cuts = _assigner(tokens).split_cues([cue], _new_cue)
    assert cuts == 0 and result == [cue]

    single = SpeakerAssigner(["理名"], ["あの", "ね"], [(0, 300), (300, 600)], np.array([[0.7], [0.6]], np.float32))
    assert single.split_cues([cue], _new_cue) == ([cue], 0)


def test_label_cues_unknown_without_tokens() -> None:
    assigner = _assigner([("はい", 0, 300, A)])
    labels = assigner.label_cues([ASRDataSeg("はい", 0, 300), ASRDataSeg("♪", 400, 900)])
    assert labels[0].speaker == "理名"
    assert labels[1].speaker == ""


def test_from_track_scores_tokens_over_their_time_span() -> None:
    scores = np.zeros((300, 2), np.float32)
    scores[:100] = A
    scores[100:] = B
    track = SpeakerTrack(list(MEMBERS), scores)
    tokens = [ASRDataSeg("あ", 0, 500), ASRDataSeg("い", 1500, 2500)]
    assigner = SpeakerAssigner.from_track(track, tokens)
    assert assigner.token_scores.argmax(1).tolist() == [0, 1]


def test_frame_scores_average_covering_windows() -> None:
    # windows of 0.04 s every 0.02 s -> frames covered by one or two windows
    window_scores = np.array([[1.0], [3.0], [5.0]])
    result = frame_scores(window_scores, 8, win=0.04, hop=0.02)
    assert result[:, 0].tolist() == pytest.approx([1, 1, 2, 2, 4, 4, 5, 5])


def test_detect_members_drops_absent_member() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    scores = np.full((n, 3), 0.3) + rng.normal(0, 0.02, (n, 3))
    scores[: n // 2, 0] = 0.7
    scores[n // 2 :, 2] = 0.7
    present, stats = detect_members(scores, ["理名", "夕莉", "朱李"])
    assert present == ["理名", "朱李"]
    assert stats["夕莉"]["share"] == 0.0
    # nothing confident at all: keep everyone rather than guess
    flat = np.full((n, 3), 0.3)
    assert detect_members(flat, ["理名", "夕莉", "朱李"])[0] == ["理名", "夕莉", "朱李"]


def test_speaker_track_cache_is_keyed(tmp_path) -> None:
    track = SpeakerTrack(list(MEMBERS), np.array([A, B], np.float32), info={"detected": MEMBERS})
    path = tmp_path / "speaker_scores.npz"
    track.save(path, {"audio_size": 1})
    loaded = SpeakerTrack.load(path, {"audio_size": 1})
    assert loaded is not None and loaded.members == MEMBERS
    assert np.allclose(loaded.scores, track.scores, atol=1e-3)
    assert loaded.info == {"detected": MEMBERS}
    assert SpeakerTrack.load(path, {"audio_size": 2}) is None
    assert SpeakerTrack.load(tmp_path / "missing.npz") is None


def test_load_enrollment_normalises_centroids(tmp_path) -> None:
    path = tmp_path / "enroll.json"
    path.write_text(json.dumps({"members": {"理名": [3, 4], "朱李": [0, 2]}, "win": 0.75, "hop": 0.2}), "utf-8")
    enrollment = load_enrollment(path)
    assert enrollment.members == ["理名", "朱李"]
    assert np.allclose(np.linalg.norm(enrollment.centroids, axis=1), 1)
    assert enrollment.digest
    assert enrollment.subset(["朱李"]).members == ["朱李"]


def test_parse_speakers() -> None:
    assert parse_speakers("理名, 朱李") == ["理名", "朱李"]
    assert parse_speakers("理名，朱李") == ["理名", "朱李"]
    assert parse_speakers("") is None
    assert parse_speakers(None) is None


def _labelled(text: str, start: int, end: int, speaker: str, margin: float) -> ASRDataSeg:
    segment = ASRDataSeg(text, start, end)
    segment.speaker = speaker
    segment.speaker_margin = margin
    return segment


def test_semantic_merge_never_joins_confidently_different_speakers() -> None:
    from qwen_asr.semantic_boundary_merge import _merge_batch

    source = [_labelled("あれって", 0, 500, "理名", 0.3), _labelled("なんだっけ", 500, 1000, "朱李", 0.3)]
    assert [s.text for s in _merge_batch(source, [[1, 2]], max_chars=36)] == ["あれって", "なんだっけ"]

    unsure = [_labelled("あれって", 0, 500, "理名", 0.3), _labelled("なんだっけ", 500, 1000, "朱李", 0.02)]
    merged = _merge_batch(unsure, [[1, 2]], max_chars=36)
    assert [s.text for s in merged] == ["あれってなんだっけ"]
    assert merged[0].speaker == "理名"


def test_semantic_merge_payload_carries_speakers(monkeypatch) -> None:
    import qwen_asr.semantic_boundary_merge as merge

    prompts = []

    def fake_call_llm(messages, **_):
        prompts.append(messages[1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"groups":[[1],[2]}'))])

    monkeypatch.setattr(merge, "call_llm", fake_call_llm)
    batch = [_labelled("あれって", 0, 500, "理名", 0.3), _labelled("うん", 500, 800, "朱李", 0.05)]
    groups = merge._request_groups(  # noqa: SLF001
        batch, model="m", base_url="http://x", api_key="k", max_chars=36, disable_thinking=True,
        llm_extra_body=None, timeout=1.0,
    )
    assert groups == [[1], [2]]  # the dropped closing bracket is repaired
    assert merge.SPEAKER_NOTE in prompts[0]
    payload = json.loads(prompts[0].split("\n", 2)[2])
    assert [item["speaker"] for item in payload] == ["理名", "?"]

    prompts.clear()
    merge._request_groups(  # noqa: SLF001
        [ASRDataSeg("あ", 0, 1), ASRDataSeg("い", 1, 2)], model="m", base_url="http://x", api_key="k",
        max_chars=36, disable_thinking=True, llm_extra_body=None, timeout=1.0,
    )
    assert merge.SPEAKER_NOTE not in prompts[0] and '"speaker"' not in prompts[0]


def test_translator_sends_speakers_as_hint(monkeypatch) -> None:
    from optimizer.translator import SubtitleTranslator

    contents = []

    def fake_call_llm(**kwargs):
        contents.append(kwargs["messages"][1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"1": {"translation": "你好", "confidence": 0.9}, "2": {"translation": "嗯", "confidence": 0.9}}'
        ))])

    monkeypatch.setattr("optimizer.translator.call_llm", fake_call_llm)
    translator = SubtitleTranslator(thread_num=1, batch_num=2, model="m", base_url="http://example.invalid/v1",
                                    api_key="k", target_language="简体中文")
    first, second = ASRDataSeg("こんにちは", 0, 500), ASRDataSeg("うん", 500, 800)
    first.speaker, second.speaker = "理名", ""
    try:
        translator._translate_chunk([first, second])  # noqa: SLF001
    finally:
        translator.stop()
    assert contents[0].endswith('<speakers>{"1": "理名"}</speakers>')
    assert first.translated_text == "你好"


def test_attach_manifest_speakers_matches_by_time_and_text() -> None:
    from qwen_asr.optimizer_bridge_stages import attach_manifest_speakers

    payload = {
        "1": {"start_time": 0, "end_time": 500, "original_subtitle": "こんにちは", "speaker": "理名"},
        "2": {"start_time": 500, "end_time": 800, "original_subtitle": "うん", "speaker": ""},
    }
    segments = [ASRDataSeg("こんにちは", 0, 500), ASRDataSeg("うん", 500, 800)]
    assert attach_manifest_speakers(payload, segments)
    assert [s.speaker for s in segments] == ["理名", ""]
    assert not attach_manifest_speakers({"1": {"start_time": 0, "end_time": 1, "original_subtitle": "a"}}, segments)


def test_prepare_speaker_track_never_fails_align(tmp_path, monkeypatch) -> None:
    import argparse

    import qwen_asr.speaker.voiceprint as voiceprint
    from qwen_asr.commands.align import prepare_speaker_track
    from qwen_asr.models import WorkPaths

    work_paths = WorkPaths.from_workdir(tmp_path / "workspaces" / "ep")
    work_paths.audio_path.parent.mkdir(parents=True)
    work_paths.audio_path.write_bytes(b"RIFF")

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("ReDimNet weights not found")

    monkeypatch.setattr(voiceprint, "ensure_speaker_track", boom)
    prepare_speaker_track(argparse.Namespace(speaker_enroll="enroll.json"), work_paths)  # logs, does not raise
    prepare_speaker_track(argparse.Namespace(), work_paths)  # disabled


def test_build_speaker_ass_uses_member_styles_and_flags_unsure_lines() -> None:
    from qwen_asr.speaker.ass_export import SPEAKER_UNSURE, build_speaker_ass
    from qwen_asr.speaker.assign import CueSpeaker

    config = {
        "script_info": {"PlayResX": "640", "PlayResY": "480"},
        "styles": {"理名": "Style: 1.理名,方正大黑_GBK,75,&H00FFFFFF", "朱李": "Style: 5.朱李,方正大黑_GBK,75,&H00FFFFFF"},
    }
    segments = [ASRDataSeg("こんにちは", 0, 1500, "你好"), ASRDataSeg("うん", 1500, 2000, "嗯"),
                ASRDataSeg("{♪}", 2000, 3000, "")]
    labels = [CueSpeaker("理名", 0.3), CueSpeaker("朱李", 0.02), CueSpeaker("", 0.0)]

    text = build_speaker_ass(segments, labels, members=["理名", "朱李"], ass_config=config, title="ep")

    assert text.startswith("\ufeff[Script Info]\r\n")
    assert "\r\r\n" not in text
    assert "PlayResX: 640" in text
    rows = text.split("\r\n")
    assert "Style: 1.理名,方正大黑_GBK,75,&H00FFFFFF" in rows and "Style: 5.朱李,方正大黑_GBK,75,&H00FFFFFF" in rows
    assert any(r.startswith("Style: Default,") for r in rows)
    dialogue = [r for r in rows if r.startswith("Dialogue:")]
    assert dialogue[0] == r"Dialogue: 0,0:00:00.00,0:00:01.50,1.理名,,0,0,0,,こんにちは\N你好"
    assert dialogue[1] == rf"Dialogue: 0,0:00:01.50,0:00:02.00,5.朱李,{SPEAKER_UNSURE},0,0,0,,うん\N嗯"
    assert dialogue[2].startswith(f"Dialogue: 0,0:00:02.00,0:00:03.00,Default,{SPEAKER_UNSURE},")
    assert "{" not in dialogue[2].split(",,", 1)[1]  # braces would start an override block


def test_build_speaker_ass_lists_only_present_members_with_fallback_style() -> None:
    from qwen_asr.speaker.ass_export import build_speaker_ass
    from qwen_asr.speaker.assign import CueSpeaker

    text = build_speaker_ass([ASRDataSeg("a", 0, 10)], [CueSpeaker("夕莉", 0.4)], members=["夕莉"], ass_config={})
    styles = [r for r in text.split("\r\n") if r.startswith("Style:")]
    assert len(styles) == 1 and styles[0].startswith("Style: 1.夕莉,")


def _export_args(media, export_path, **extra):
    import argparse

    from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT

    return argparse.Namespace(
        force=False, format="srt", source="transcript", max_subtitle_duration=6.0, min_subtitle_duration=1.0,
        max_chars_per_line_zh=18, max_chars_per_line_en=42, max_lines=2, pause_split_seconds=0.8,
        coarse_subtitles=False, optimizer_root=str(DEFAULT_OPTIMIZER_ROOT), export_mode="custom",
        export_path=str(export_path), media_path=str(media), **extra,
    )


def _export_workspace(tmp_path):
    from qwen_asr.models import WorkPaths
    from qwen_asr.storage import write_json_atomic

    media = tmp_path / "demo.mp3"
    media.write_bytes(b"media")
    paths = WorkPaths.from_workdir(tmp_path / "workspaces" / "0001-demo")
    write_json_atomic(paths.transcript_manifest, [{
        "segment_id": "segment_000001", "audio_path": "segment.wav", "global_start_time": 0.0,
        "global_end_time": 1.0, "text": "こんにちは", "status": "completed",
    }])
    return media, paths


def test_export_writes_speaker_ass_next_to_srt(tmp_path, monkeypatch) -> None:
    import qwen_asr.commands.export as export
    import qwen_asr.optimizer_bridge_adapter as adapter
    import qwen_asr.optimizer_bridge_stages as bridge
    from qwen_asr.commands.stages import cmd_export

    media, paths = _export_workspace(tmp_path)
    enroll = tmp_path / "enroll.json"
    enroll.write_text(json.dumps({"members": {"理名": [1, 0], "朱李": [0, 1]},
                                  "ass": {"styles": {"理名": "Style: 1.理名,X,75"}}}), "utf-8")
    fake = _assigner([("こんにちは", 0, 1000, A)])
    monkeypatch.setattr(bridge, "load_speaker_assigner", lambda *_a, **_k: fake)
    monkeypatch.setattr(adapter, "aligned_manifest_to_asr_data", lambda *_a, **_k: SimpleNamespace(segments=[]))
    out = tmp_path / "exports" / "ep.srt"

    assert cmd_export(_export_args(media, out, speaker_enroll=str(enroll)), paths) == 0

    ass = out.with_suffix(".ass")
    assert out.exists() and ass.exists()
    assert ",1.理名,," in ass.read_text("utf-8-sig")
    assert export.speaker_ass_path(paths).exists()

    # without --speaker-enroll the stale cached ASS is removed and nothing new is copied
    ass.unlink()
    assert cmd_export(_export_args(media, out), paths) == 0
    assert not export.speaker_ass_path(paths).exists() and not ass.exists()


def test_export_without_speaker_track_still_writes_srt(tmp_path, monkeypatch) -> None:
    import qwen_asr.optimizer_bridge_stages as bridge
    from qwen_asr.commands.stages import cmd_export

    media, paths = _export_workspace(tmp_path)
    monkeypatch.setattr(bridge, "load_speaker_assigner", lambda *_a, **_k: None)
    out = tmp_path / "exports" / "ep.srt"
    assert cmd_export(_export_args(media, out, speaker_enroll="missing.json"), paths) == 0
    assert out.exists() and not out.with_suffix(".ass").exists()
