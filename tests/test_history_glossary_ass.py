from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from qwen_asr.history_glossary import _ass_time_to_ms as legacy_ass_time_to_ms
from qwen_asr.history_glossary import _clean_ass_text as legacy_clean_ass_text
from qwen_asr.history_glossary import parse_ass_dialogues as legacy_parse_ass_dialogues
from qwen_asr.history_glossary_ass import (
    ass_time_to_ms,
    clean_ass_text,
    export_review_ass,
    ms_to_ass_time,
    parse_ass_dialogues,
)


@dataclass(frozen=True)
class Dialogue:
    start_ms: int
    end_ms: int
    style: str
    text: str


def test_parse_ass_dialogues_dedupes_and_cleans_formatting(tmp_path: Path) -> None:
    ass = tmp_path / "sample.ass"
    ass.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.60,Default,,0,0,0,,{\\i1}无刺\\N有刺\\h广播\n"
        "Dialogue: 0,0:00:01.00,0:00:02.60,Default,,0,0,0,,{\\i1}无刺\\N有刺\\h广播\n",
        encoding="utf-8",
    )

    dialogues = parse_ass_dialogues(ass, Dialogue)
    legacy_dialogues = legacy_parse_ass_dialogues(ass)

    assert dialogues == [Dialogue(start_ms=1000, end_ms=2600, style="Default", text="无刺 有刺 广播")]
    assert [(item.start_ms, item.end_ms, item.style, item.text) for item in legacy_dialogues] == [
        (1000, 2600, "Default", "无刺 有刺 广播")
    ]


def test_ass_text_and_time_helpers_keep_legacy_aliases() -> None:
    assert clean_ass_text("{\\pos(1,2)}A\\hB\\NC") == "A B C"
    assert legacy_clean_ass_text("{\\pos(1,2)}A\\hB\\NC") == "A B C"
    assert ass_time_to_ms("1:02:03.45") == 3723450
    assert legacy_ass_time_to_ms("1:02:03.45") == 3723450
    assert ms_to_ass_time(3723450) == "1:02:03.45"


def test_export_review_ass_writes_styles_and_skips_high_matches(tmp_path: Path) -> None:
    output = tmp_path / "review.ass"
    matches = [
        SimpleNamespace(
            episode_id="1",
            ass_start_ms=1000,
            ass_end_ms=2000,
            level="high",
            ass_text="skip",
            source_text="skip",
            score=0.99,
            reasons=[],
        ),
        SimpleNamespace(
            episode_id="1",
            ass_start_ms=3000,
            ass_end_ms=3600,
            level="low",
            ass_text="无刺有刺",
            source_text="トゲトゲ",
            score=0.42,
            reasons=["boundary"],
        ),
    ]

    export_review_ass(output, matches, ensure_directory=lambda path: path.mkdir(parents=True, exist_ok=True))

    text = output.read_text(encoding="utf-8")
    assert "Style: ReviewLow" in text
    assert "Style: ReviewNote" in text
    assert "score=0.42 level=low" in text
    assert "skip" not in text
