"""ASS export with one style per speaker, the way the studio's proofread subtitles mark who is talking.

Studio files (#15/#16/#18) use one style per member named "<number>.<member>" ("1.理名", "2.夕莉", "5.朱李") that only
differ in colour, list only the members of the episode, and leave the Name field empty except for notes such as
"听不清". Here every cue gets the style of its speaker; when the voice decision is not confident the most likely
speaker's style is still used and the Name field says SPEAKER_UNSURE so the line can be checked; cues without any
scored word (music, jingles without ASR tokens) use the neutral "Default" style.
"""
from __future__ import annotations

from typing import Any, Sequence

try:  # same text cleanup as the SRT export
    from optimizer.asr_data import clean_subtitle_text
except ImportError:  # pragma: no cover
    def clean_subtitle_text(text: str) -> str:
        return text

SPEAKER_UNSURE = "说话人待确认"
# Cue-level margin below which the line is flagged. Episode 30 gold (110 single-member cues): margin < 0.05 was right
# 8/12 times, 0.05-0.1 12/13, >= 0.1 84/85 — flagging below 0.05 marks the risky lines without crying wolf.
UNSURE_MARGIN = 0.05
DEFAULT_STYLE_NAME = "Default"
STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
    "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
    "MarginV, Encoding"
)
# Same font/size/layout as the studio member styles, neutral colours.
FALLBACK_STYLE = "{name},方正大黑_GBK,75,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,5,1.2,2,10,10,10,1"
EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


def _style_name(style_line: str) -> str:
    return style_line.split(":", 1)[1].split(",", 1)[0].strip()


def member_styles(members: Sequence[str], configured: dict[str, str]) -> dict[str, str]:
    """member -> full "Style: ..." line; members without a configured style get a neutral "<n>.<member>" style."""
    out = {}
    for index, member in enumerate(members, 1):
        line = configured.get(member)
        out[member] = line if line else "Style: " + FALLBACK_STYLE.format(name=f"{index}.{member}")
    return out


def _ass_time(ms: int) -> str:
    cs = max(int(round(ms / 10)), 0)
    hours, rest = divmod(cs, 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{cs:02}"


def _ass_text(text: str) -> str:
    # braces start override blocks in ASS; newlines become hard line breaks
    lines = [line.strip() for line in str(text or "").replace("\r", "").split("\n")]
    joined = r"\N".join(line for line in lines if line)
    return joined.replace("{", "｛").replace("}", "｝")


def build_speaker_ass(
    segments: Sequence[Any],
    labels: Sequence[Any],
    *,
    members: Sequence[str],
    ass_config: dict[str, Any] | None = None,
    unsure_margin: float = UNSURE_MARGIN,
    title: str = "",
) -> str:
    """segments: .text / .translated_text / .start_time / .end_time (ms); labels: .speaker / .margin per segment."""
    config = ass_config or {}
    styles = member_styles(members, dict(config.get("styles") or {}))
    info = {"WrapStyle": "0", "PlayResX": "640", "PlayResY": "480", "ScaledBorderAndShadow": "yes"}
    info.update({str(k): str(v) for k, v in (config.get("script_info") or {}).items()})
    events, used_default = [], False
    for segment, label in zip(segments, labels, strict=True):
        speaker = getattr(label, "speaker", "") or ""
        if speaker in styles:
            style = _style_name(styles[speaker])
            name = "" if float(getattr(label, "margin", 0.0)) >= unsure_margin else SPEAKER_UNSURE
        else:
            style, name, used_default = DEFAULT_STYLE_NAME, SPEAKER_UNSURE, True
        parts = (clean_subtitle_text(str(segment.text or "")), clean_subtitle_text(str(getattr(segment, "translated_text", "") or "")))
        text = "\n".join(part for part in parts if part.strip())
        events.append(
            f"Dialogue: 0,{_ass_time(int(segment.start_time))},{_ass_time(int(segment.end_time))},{style},{name},"
            f"0,0,0,,{_ass_text(text)}"
        )
    style_lines = [styles[m] for m in members]
    if used_default:
        style_lines.append("Style: " + FALLBACK_STYLE.format(name=DEFAULT_STYLE_NAME))
    head = ["[Script Info]", f"Title: {title or 'KoeAxis'}", "ScriptType: v4.00+"]
    head += [f"{key}: {value}" for key, value in info.items()]
    body = ["", "[V4+ Styles]", STYLE_FORMAT, *style_lines, "", "[Events]", EVENT_FORMAT, *events]
    return "\ufeff" + "\r\n".join(head + body) + "\r\n"
