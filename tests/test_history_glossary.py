from __future__ import annotations

from pathlib import Path
import json

from qwen_asr.history_glossary import (
    MatchResult,
    _parse_history_llm_extra_body,
    export_review_ass,
    extract_glossary_entries,
    extract_glossary_entries_with_llm,
    match_dialogues_to_asr,
    parse_ass_dialogues,
)
from qwen_asr.models import AlignedSegment, AlignedToken
from optimizer.asr_data import ASRDataSeg


def test_parse_ass_dialogues_reads_dialogue_lines(tmp_path: Path) -> None:
    ass_path = tmp_path / "sample.ass"
    ass_path.write_text(
        "[Script Info]\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,你好\\N世界\n"
        "Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,{\\i1}第二句\n",
        encoding="utf-8",
    )

    dialogues = parse_ass_dialogues(ass_path)

    assert len(dialogues) == 2
    assert dialogues[0].start_ms == 1000
    assert dialogues[0].end_ms == 2500
    assert dialogues[0].text == "你好 世界"
    assert dialogues[1].text == "第二句"


def test_parse_ass_dialogues_deduplicates_same_timed_text(tmp_path: Path) -> None:
    ass_path = tmp_path / "dedupe.ass"
    ass_path.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,A,,0,0,0,,同一句\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,B,,0,0,0,,同一句\n",
        encoding="utf-8",
    )

    dialogues = parse_ass_dialogues(ass_path)

    assert len(dialogues) == 1


def test_match_dialogues_to_asr_scores_split_and_token_fallback() -> None:
    split_segments = [
        ASRDataSeg("トゲナシトゲアリのトゲラジ", 900, 2600),
        ASRDataSeg("コエログみなさん新生活慣れましたか", 4900, 8100),
    ]
    aligned = [
        AlignedSegment(
            segment_id="segment_1",
            audio_path="a.wav",
            global_start_time=0.9,
            global_end_time=8.1,
            text="dummy",
            tokens=[
                AlignedToken(text="トゲナシ", start_time=0.9, end_time=1.4),
                AlignedToken(text="トゲアリ", start_time=1.4, end_time=1.9),
            ],
        )
    ]
    dialogues = parse_ass_dialogues_from_lines(
        [
            "Dialogue: 0,0:00:01.00,0:00:02.60,Default,,0,0,0,,无刺有刺的刺刺广播",
            "Dialogue: 0,0:00:09.00,0:00:10.00,Default,,0,0,0,,没有匹配",
        ]
    )

    matches = match_dialogues_to_asr(
        episode_id="15",
        media_path=Path("demo.mp3"),
        ass_path=Path("demo.ass"),
        dialogues=dialogues,
        split_segments=split_segments,
        aligned_segments=aligned,
        min_match_score=0.72,
    )

    assert matches[0].source_kind == "split"
    assert matches[0].score > 0.72
    assert matches[0].level == "high"
    assert matches[1].level == "low"


def test_export_review_ass_contains_styles_and_reason(tmp_path: Path) -> None:
    ass_path = tmp_path / "review.ass"
    export_review_ass(
        ass_path,
        [
            MatchResult(
                episode_id="15",
                media_path="demo.mp3",
                ass_path="demo.ass",
                ass_start_ms=1000,
                ass_end_ms=2000,
                ass_text="中文句子",
                source_text="日本語候補",
                source_kind="split",
                source_start_ms=1000,
                source_end_ms=2100,
                matched_segment_count=3,
                score=0.42,
                level="low",
                time_overlap_score=0.2,
                boundary_score=0.3,
                length_ratio_score=0.4,
                merge_penalty=0.16,
                token_coverage_score=0.5,
                reasons=["time weak", "merged 3 splits"],
            )
        ],
    )

    content = ass_path.read_text(encoding="utf-8")
    assert "Style: ReviewLow" in content
    assert "Style: ReviewNote" in content
    assert "score=0.42 level=low" in content


def test_extract_glossary_entries_writes_readable_xlsx(tmp_path: Path) -> None:
    matches = [
        MatchResult(
            episode_id="15",
            media_path="demo.mp3",
            ass_path="demo.ass",
            ass_start_ms=1000,
            ass_end_ms=2000,
            ass_text="无刺有刺的刺刺广播",
            source_text="トゲナシトゲアリのトゲラジ",
            source_kind="split",
            source_start_ms=1000,
            source_end_ms=2000,
            matched_segment_count=1,
            score=0.91,
            level="high",
            time_overlap_score=0.9,
            boundary_score=0.9,
            length_ratio_score=0.9,
            merge_penalty=0.0,
            token_coverage_score=0.9,
            reasons=["ok"],
        ),
        MatchResult(
            episode_id="16",
            media_path="demo2.mp3",
            ass_path="demo2.ass",
            ass_start_ms=1200,
            ass_end_ms=2200,
            ass_text="无刺有刺的刺刺广播",
            source_text="トゲナシトゲアリのトゲラジ",
            source_kind="split",
            source_start_ms=1200,
            source_end_ms=2200,
            matched_segment_count=1,
            score=0.89,
            level="high",
            time_overlap_score=0.88,
            boundary_score=0.85,
            length_ratio_score=0.9,
            merge_penalty=0.0,
            token_coverage_score=0.92,
            reasons=["ok"],
        ),
    ]

    entries = extract_glossary_entries(matches, min_match_score=0.72, min_term_frequency=2)

    assert len(entries) == 1
    assert entries[0].source == "トゲナシトゲアリのトゲラジ"
    assert entries[0].target == "无刺有刺的刺刺广播"


def test_extract_glossary_entries_filters_sentence_like_candidates() -> None:
    matches = [
        MatchResult(
            episode_id="15",
            media_path="demo.mp3",
            ass_path="demo.ass",
            ass_start_ms=1000,
            ass_end_ms=6000,
            ass_text="我们无刺有刺不仅仅在「Girls Band Cry」剧中登场",
            source_text="アニメガールズバンドくらいの劇場に登場するバンドだけでなく",
            source_kind="split",
            source_start_ms=1000,
            source_end_ms=5900,
            matched_segment_count=2,
            score=0.93,
            level="high",
            time_overlap_score=0.95,
            boundary_score=0.9,
            length_ratio_score=0.8,
            merge_penalty=0.08,
            token_coverage_score=0.95,
            reasons=["ok"],
        )
    ]

    entries = extract_glossary_entries(matches, min_match_score=0.72, min_term_frequency=1)

    assert entries == []


def test_extract_glossary_entries_keeps_keyword_like_short_terms() -> None:
    matches = [
        MatchResult(
            episode_id="15",
            media_path="demo.mp3",
            ass_path="demo.ass",
            ass_start_ms=1000,
            ass_end_ms=2000,
            ass_text="「声log」",
            source_text="コイログ",
            source_kind="split",
            source_start_ms=1000,
            source_end_ms=2000,
            matched_segment_count=1,
            score=0.90,
            level="high",
            time_overlap_score=0.9,
            boundary_score=0.9,
            length_ratio_score=0.9,
            merge_penalty=0.0,
            token_coverage_score=0.9,
            reasons=["ok"],
        )
    ]

    entries = extract_glossary_entries(matches, min_match_score=0.72, min_term_frequency=1)

    assert len(entries) == 1
    assert entries[0].group == "show_terms"
    assert entries[0].source == "コイログ"
    assert "curated=priority" in entries[0].note


def test_extract_glossary_entries_can_be_empty() -> None:
    entries = extract_glossary_entries([], min_match_score=0.72, min_term_frequency=2)

    assert entries == []


def test_extract_glossary_entries_allows_curated_singletons() -> None:
    matches = [
        MatchResult(
            episode_id="15",
            media_path="demo.mp3",
            ass_path="demo.ass",
            ass_start_ms=1000,
            ass_end_ms=2000,
            ass_text="从川崎走向世界",
            source_text="川崎から世界へ",
            source_kind="split",
            source_start_ms=1000,
            source_end_ms=2000,
            matched_segment_count=1,
            score=0.88,
            level="high",
            time_overlap_score=0.9,
            boundary_score=0.9,
            length_ratio_score=0.9,
            merge_penalty=0.0,
            token_coverage_score=0.9,
            reasons=["ok"],
        )
    ]

    entries = extract_glossary_entries(matches, min_match_score=0.72, min_term_frequency=2)

    assert len(entries) == 1
    assert entries[0].source == "川崎から世界へ"


def test_extract_glossary_entries_with_llm_uses_json_response(monkeypatch) -> None:
    calls = {}

    class Message:
        content = json.dumps(
            {
                "entries": [
                    {
                        "group": "show_terms",
                        "source": "コイログ",
                        "target": "声log",
                        "note": "栏目名",
                    }
                ]
            },
            ensure_ascii=False,
        )

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    def fake_call_llm(**kwargs):
        calls["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr("qwen_asr.history_glossary.call_llm", fake_call_llm)

    entries = extract_glossary_entries_with_llm(
        [
            MatchResult(
                episode_id="15",
                media_path="demo.mp3",
                ass_path="demo.ass",
                ass_start_ms=1000,
                ass_end_ms=2000,
                ass_text="「声log」",
                source_text="コイログ",
                source_kind="split",
                source_start_ms=1000,
                source_end_ms=2000,
                matched_segment_count=1,
                score=0.90,
                level="high",
                time_overlap_score=0.9,
                boundary_score=0.9,
                length_ratio_score=0.9,
                merge_penalty=0.0,
                token_coverage_score=0.9,
                reasons=["ok"],
            )
        ],
        min_match_score=0.72,
        llm_model="model",
        base_url="http://localhost:8000/v1",
        api_key="key",
        disable_thinking=True,
        llm_extra_body_json=None,
        timeout=120.0,
    )

    assert len(entries) == 1
    assert entries[0].source == "コイログ"
    assert calls["kwargs"]["require_json"] is True


def test_extract_glossary_entries_with_llm_filters_contextual_role_phrase(monkeypatch) -> None:
    class Message:
        content = json.dumps(
            {
                "entries": [
                    {
                        "group": "fixed_phrases",
                        "source": "監督朱李",
                        "target": "导演朱李",
                        "note": "称呼",
                    },
                    {
                        "group": "names",
                        "source": "しゅり",
                        "target": "朱李",
                        "note": "姓名",
                    },
                ]
            },
            ensure_ascii=False,
        )

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    def fake_call_llm(**kwargs):
        return Response()

    monkeypatch.setattr("qwen_asr.history_glossary.call_llm", fake_call_llm)

    entries = extract_glossary_entries_with_llm(
        [
            MatchResult(
                episode_id="15",
                media_path="demo.mp3",
                ass_path="demo.ass",
                ass_start_ms=1000,
                ass_end_ms=2000,
                ass_text="导演朱李",
                source_text="監督朱李",
                source_kind="split",
                source_start_ms=1000,
                source_end_ms=2000,
                matched_segment_count=1,
                score=0.90,
                level="high",
                time_overlap_score=0.9,
                boundary_score=0.9,
                length_ratio_score=0.9,
                merge_penalty=0.0,
                token_coverage_score=0.9,
                reasons=["ok"],
            )
        ],
        min_match_score=0.72,
        llm_model="model",
        base_url="http://localhost:8000/v1",
        api_key="key",
        disable_thinking=True,
        llm_extra_body_json=None,
        timeout=120.0,
    )

    assert len(entries) == 1
    assert entries[0].source == "しゅり"
    assert entries[0].target == "朱李"


def test_parse_history_llm_extra_body_accepts_python_literal_dict() -> None:
    parsed = _parse_history_llm_extra_body("{'top_p': 0.2, 'max_tokens': 800}")

    assert parsed == {"top_p": 0.2, "max_tokens": 800}


def parse_ass_dialogues_from_lines(lines: list[str]):
    return [
        dialogue
        for dialogue in parse_ass_dialogues_from_text(
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            + "\n".join(lines)
        )
    ]


def parse_ass_dialogues_from_text(content: str):
    path = Path(__file__).with_name("inline.ass")
    path.write_text(content, encoding="utf-8")
    try:
        return parse_ass_dialogues(path)
    finally:
        path.unlink(missing_ok=True)
