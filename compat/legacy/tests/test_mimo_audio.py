from __future__ import annotations

from pathlib import Path
import wave

from qwen_asr.mimo_audio import (
    build_nearby_audio_batches,
    filter_glossary,
    nearby_entries_for_subtitle_id,
    nearby_entries_for_subtitle_ids,
    segment_for_subtitle_id,
    subtitle_entries_for_segment,
    write_nearby_audio_clip,
)


def test_subtitle_entries_for_segment_selects_overlapping_entries() -> None:
    translated = {
        "1": {"start_time": 0, "end_time": 900},
        "2": {"start_time": 900, "end_time": 1500},
        "3": {"start_time": 2100, "end_time": 2500},
        "meta": "ignored",
    }

    entries = subtitle_entries_for_segment(
        translated,
        {"global_start_time": 1.0, "global_end_time": 2.0},
    )

    assert list(entries) == ["2"]


def test_filter_glossary_puts_standing_terms_before_text_matches() -> None:
    glossary = [
        {"group": "names", "source": "Alice", "target": "爱丽丝", "note": ""},
        {"group": "misc", "source": "Unused", "target": "未使用", "note": ""},
        {"group": "places", "source": "Cafe", "target": "咖啡馆", "note": ""},
        {"group": "show_terms", "source": "Guild", "target": "公会", "note": ""},
    ]
    entries = {
        "1": {"original_subtitle": "Meet Alice", "translated_subtitle": "见爱丽丝"},
        "2": {"original_subtitle": "Other", "translated_subtitle": "其他"},
    }

    selected = filter_glossary(glossary, entries, limit=3)

    assert [item["source"] for item in selected] == ["Alice", "Guild"]


def test_segment_for_subtitle_id_uses_subtitle_start_time() -> None:
    segments = [
        {"segment_id": "a", "global_start_time": 0.0, "global_end_time": 1.0},
        {"segment_id": "b", "global_start_time": 1.0, "global_end_time": 2.0},
    ]

    segment = segment_for_subtitle_id("7", segments, {"7": {"start_time": 1200}})

    assert segment["segment_id"] == "b"


def test_nearby_entries_for_single_and_batch_ids() -> None:
    translated = {str(index): {"start_time": index * 100, "end_time": index * 100 + 50} for index in range(1, 8)}

    assert list(nearby_entries_for_subtitle_id("4", translated, 1)) == ["3", "4", "5"]
    assert list(nearby_entries_for_subtitle_ids(["3", "5"], translated, 1)) == ["2", "3", "4", "5", "6"]


def test_build_nearby_audio_batches_groups_by_segment_gap_and_size() -> None:
    translated = {
        "1": {"start_time": 1000, "end_time": 1200},
        "2": {"start_time": 1300, "end_time": 1500},
        "3": {"start_time": 2600, "end_time": 2800},
        "4": {"start_time": 4100, "end_time": 4300},
    }
    segments = [
        {"segment_id": "s1", "global_start_time": 0.0, "global_end_time": 3.0},
        {"segment_id": "s2", "global_start_time": 3.0, "global_end_time": 5.0},
    ]

    batches = build_nearby_audio_batches(
        subtitle_ids=["1", "2", "3", "4"],
        segments=segments,
        translated=translated,
        context_subtitles=0,
        batch_size=2,
        max_gap_s=0.5,
    )

    assert batches == [["1", "2"], ["3"], ["4"]]


def test_write_nearby_audio_clip_uses_full_source_audio_across_segment_boundary(tmp_path: Path) -> None:
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

    _, meta = write_nearby_audio_clip(
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


def test_filter_glossary_sends_names_and_show_terms_but_not_unmatched_usage_notes() -> None:
    glossary = [
        {"group": "通用日语", "source": "ちゃん", "target": "小XX", "note": ""},
        {"group": "通用日语", "source": "トゲナシトゲアリ", "target": "无刺有刺", "note": ""},
        {"group": "日文", "source": "ディレクター", "target": "导演", "note": ""},
        {"group": "日文", "source": "絵コンテ", "target": "分镜脚本", "note": "将剧本分解为镜头画面的视觉蓝图" * 2},
        {"group": "声优/乐手名（中文）", "source": "リナ", "target": "理名", "note": "主唱；乐队：无刺有刺"},
    ]
    entries = {"1": {"original_subtitle": "絵コンテを描いた", "translated_subtitle": "画了分镜"}}

    selected = filter_glossary(glossary, entries, limit=80)

    assert [item["source"] for item in selected] == ["トゲナシトゲアリ", "リナ", "絵コンテ"]
    assert all("group" not in item for item in selected)
    assert selected[2]["note"].endswith("…") and len(selected[2]["note"]) == 25
    assert selected[1]["note"] == "主唱；乐队：无刺有刺"


def test_filter_glossary_matches_on_chinese_target_too() -> None:
    glossary = [{"group": "日文", "source": "ディレクター", "target": "导演", "note": ""}]
    selected = filter_glossary(glossary, {"1": {"original_subtitle": "監督", "translated_subtitle": "导演说"}}, 80)
    assert [item["source"] for item in selected] == ["ディレクター"]
