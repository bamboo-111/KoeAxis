from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

ASS_DIALOGUE_PATTERN = re.compile(
    r"^Dialogue:\s*(?P<layer>[^,]*),(?P<start>[^,]*),(?P<end>[^,]*),(?P<style>[^,]*),"
    r"(?P<name>[^,]*),(?P<margin_l>[^,]*),(?P<margin_r>[^,]*),(?P<margin_v>[^,]*),"
    r"(?P<effect>[^,]*),(?P<text>.*)$"
)


def parse_ass_dialogues(path: Path, dialogue_type: type[Any]) -> list[Any]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    dialogues: list[Any] = []
    seen: set[tuple[int, int, str]] = set()
    for line in text.splitlines():
        match = ASS_DIALOGUE_PATTERN.match(line.strip())
        if not match:
            continue
        subtitle_text = clean_ass_text(match.group("text"))
        if not subtitle_text:
            continue
        start_ms = ass_time_to_ms(match.group("start"))
        end_ms = ass_time_to_ms(match.group("end"))
        key = (start_ms, end_ms, subtitle_text)
        if key in seen:
            continue
        seen.add(key)
        dialogues.append(
            dialogue_type(
                start_ms=start_ms,
                end_ms=end_ms,
                style=match.group("style").strip(),
                text=subtitle_text,
            )
        )
    return dialogues


def export_review_ass(path: Path, matches: Iterable[Any], *, ensure_directory: Any) -> None:
    ensure_directory(path.parent)
    body = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Match,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,2,1,2,40,40,26,1",
        "Style: ReviewLow,Arial,36,&H0000E5FF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,2,1,2,40,40,26,1",
        "Style: ReviewNote,Arial,22,&H00A0A0A0,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,1,0,8,40,40,40,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for item in sorted(matches, key=lambda row: (int(row.episode_id), row.ass_start_ms, row.ass_end_ms)):
        if item.level == "high":
            continue
        style = "ReviewLow" if item.level == "low" else "Match"
        text = f"[#{item.episode_id}] {item.ass_text}\\N{item.source_text or '(no source match)'}"
        note = f"score={item.score:.2f} level={item.level} reason={' / '.join(item.reasons[:4])}"
        body.append(
            "Dialogue: 0,{start},{end},{style},,0,0,0,,{text}".format(
                start=ms_to_ass_time(item.ass_start_ms),
                end=ms_to_ass_time(item.ass_end_ms),
                style=style,
                text=escape_ass_text(text),
            )
        )
        note_end = min(item.ass_end_ms + 800, item.ass_end_ms + max(400, item.ass_end_ms - item.ass_start_ms))
        body.append(
            "Dialogue: 0,{start},{end},ReviewNote,,0,0,0,,{text}".format(
                start=ms_to_ass_time(item.ass_start_ms),
                end=ms_to_ass_time(note_end),
                text=escape_ass_text(note),
            )
        )
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def clean_ass_text(text: str) -> str:
    cleaned = re.sub(r"\{[^}]*\}", "", text)
    cleaned = cleaned.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def escape_ass_text(text: str) -> str:
    escaped = text.replace("{", "\\{").replace("}", "\\}")
    return escaped.replace("\n", "\\N")


def ass_time_to_ms(value: str) -> int:
    hours, minutes, seconds_cs = value.strip().split(":")
    seconds, centiseconds = seconds_cs.split(".")
    total = (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000
    return total + int(centiseconds) * 10


def ms_to_ass_time(value: int) -> str:
    total = max(0, int(value))
    centiseconds = (total % 1000) // 10
    total_seconds = total // 1000
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
