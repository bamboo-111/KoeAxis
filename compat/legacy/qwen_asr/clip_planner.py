"""Plan and cut highlight clips from a full-episode subtitle.

Workflow: an LLM scores candidate clips on a numbered transcript (cue ranges only, never times),
the user picks candidates from the rendered list, and the picked ranges are cut losslessly with
ffmpeg. Cut points follow the editor's hand-cut habit measured on 8 LosslessCut clips: start 1.0s
before the first cue (but not inside the previous cue), end 1.0s after the last cue (~0.4s mean error).
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PRE_ROLL_S = 1.0
DEFAULT_POST_ROLL_S = 1.0
MIN_LEAD_S = 0.15
MIN_SLICED_CUE_S = 0.3
MIN_CLIP_S = 20.0
MAX_CLIP_S = 20 * 60.0
_FILENAME_BAD = re.compile(r'[\\/:*?"<>|\r\n\t]+')
_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


@dataclass(frozen=True)
class Cue:
    index: int  # 1-based, as in the SRT file
    start: float
    end: float
    text: str
    translation: str = ""


@dataclass(frozen=True)
class ClipCandidate:
    number: int  # 1-based id shown to the user
    stars: int
    start_cue: int
    end_cue: int
    start: float
    end: float
    content: str
    title_full: str
    title_short: str
    reason: str
    note: str = ""
    verify: str = ""  # words to confirm by listening before trusting the highlight

    @property
    def duration(self) -> float:
        return self.end - self.start


class ClipPlanError(ValueError):
    pass


def _seconds(match: re.Match[str]) -> float:
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip())
    for block in blocks:
        lines = [line.strip() for line in block.split("\n")]
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        times = _TIME.findall(lines[timing])
        if len(times) != 2:
            continue
        start = _seconds(_TIME.search(lines[timing]))  # type: ignore[arg-type]
        end_match = list(_TIME.finditer(lines[timing]))[1]
        body = [line for line in lines[timing + 1:] if line]
        if not body:
            continue
        cues.append(Cue(len(cues) + 1, start, _seconds(end_match), body[0], " / ".join(body[1:])))
    return cues


def format_clock(seconds: float) -> str:
    seconds = round(max(0.0, seconds), 1)  # round first: 59.96s must not print as 00:60.0
    minutes, rest = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{rest:04.1f}"
    return f"{minutes:02d}:{rest:04.1f}"


def transcript_for_prompt(cues: list[Cue]) -> str:
    return "\n".join(
        f"#{cue.index} [{format_clock(cue.start)}] {cue.text}" + (f" | {cue.translation}" if cue.translation else "")
        for cue in cues
    )


def build_messages(cues: list[Cue], *, prompt: str, title: str) -> list[dict[str, str]]:
    user = f"节目：{title}\n共 {len(cues)} 条字幕。\n\n{transcript_for_prompt(cues)}"
    return [{"role": "system", "content": prompt}, {"role": "user", "content": user}]


def extract_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = fenced + [text[text.find("{"): text.rfind("}") + 1]] if "{" in text else fenced
    for raw in candidates:
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ClipPlanError("LLM response does not contain a JSON object")


def cut_window(
    cues: list[Cue],
    start_cue: int,
    end_cue: int,
    *,
    pre_roll: float = DEFAULT_PRE_ROLL_S,
    post_roll: float = DEFAULT_POST_ROLL_S,
    media_duration: float | None = None,
) -> tuple[float, float]:
    """Cut times for an inclusive 1-based cue range.

    The pre-roll never reaches back into the previous cue (its last words would open the clip);
    when the previous cue runs right up to the first one, cut MIN_LEAD_S before the first cue.
    The post-roll is not clamped: a little of the next sentence is the editor's fade-out room.
    """
    if not 1 <= start_cue <= end_cue <= len(cues):
        raise ClipPlanError(f"cue range {start_cue}-{end_cue} outside 1-{len(cues)}")
    first = cues[start_cue - 1]
    start = first.start - pre_roll
    if start_cue > 1:
        start = max(start, min(cues[start_cue - 2].end, first.start - MIN_LEAD_S))
    start = max(0.0, start)
    end = cues[end_cue - 1].end + post_roll
    if media_duration is not None:
        end = min(end, media_duration)
    return round(start, 3), round(end, 3)


def _text(value: Any) -> str:
    return str(value or "").strip()


def build_plan(
    response: dict[str, Any],
    cues: list[Cue],
    *,
    pre_roll: float = DEFAULT_PRE_ROLL_S,
    post_roll: float = DEFAULT_POST_ROLL_S,
    media_duration: float | None = None,
) -> dict[str, Any]:
    """Validate the LLM answer and attach cut times. Invalid candidates are dropped with a warning."""
    raw = response.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise ClipPlanError("LLM response has no candidates")
    kept: list[ClipCandidate] = []
    index_map: dict[int, int] = {}
    warnings: list[str] = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            warnings.append(f"candidate {position}: not an object")
            continue
        try:
            start_cue, end_cue = int(item["start_cue"]), int(item["end_cue"])
            start, end = cut_window(cues, start_cue, end_cue, pre_roll=pre_roll, post_roll=post_roll,
                                    media_duration=media_duration)
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"candidate {position}: {exc}")
            continue
        if not MIN_CLIP_S <= end - start <= MAX_CLIP_S:
            warnings.append(f"candidate {position}: duration {end - start:.0f}s out of range")
            continue
        try:
            stars = min(5, max(1, int(item.get("stars", 3))))
        except (TypeError, ValueError):
            stars = 3
        title_short = safe_filename(_text(item.get("title_short")) or _text(item.get("title_full")) or f"片段{position + 1}")
        index_map[position] = len(kept)
        kept.append(ClipCandidate(
            number=len(kept) + 1, stars=stars, start_cue=start_cue, end_cue=end_cue, start=start, end=end,
            content=_text(item.get("content")), title_full=_text(item.get("title_full")) or title_short,
            title_short=title_short, reason=_text(item.get("reason")), note=_text(item.get("note")),
            verify=_text(item.get("verify")),
        ))
    if not kept:
        raise ClipPlanError("no valid candidates: " + "; ".join(warnings))

    def picks(value: Any) -> list[dict[str, Any]]:
        out = []
        for entry in value if isinstance(value, list) else [value]:
            if not isinstance(entry, dict):
                continue
            try:
                mapped = index_map[int(entry.get("index"))]
            except (KeyError, TypeError, ValueError):
                continue
            out.append({"number": kept[mapped].number, "why": _text(entry.get("why"))})
        return out

    return {
        "core_judgment": _text(response.get("core_judgment")),
        "positioning": _text(response.get("positioning")),
        "candidates": [candidate.__dict__ for candidate in kept],
        "top3": picks(response.get("top3"))[:3],
        "single_pick": (picks(response.get("single_pick")) or [None])[0],
        "pre_roll_s": pre_roll,
        "post_roll_s": post_roll,
        "warnings": warnings,
    }


def candidates_from_plan(plan: dict[str, Any]) -> list[ClipCandidate]:
    return [ClipCandidate(**item) for item in plan["candidates"]]


def safe_filename(name: str, limit: int = 60) -> str:
    cleaned = _FILENAME_BAD.sub(" ", name).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:limit].rstrip() or "片段"


def _cell(value: str) -> str:
    return value.replace("|", "／").replace("\n", " ")


def render_markdown(plan: dict[str, Any], cues: list[Cue], *, title: str) -> str:
    candidates = candidates_from_plan(plan)
    by_number = {candidate.number: candidate for candidate in candidates}
    overlaps = overlap_notes(candidates)
    lines = [f"# {title} 切片候选", "", plan.get("core_judgment") or "", "",
             "| 编号 | 推荐度 | 时间 | 时长 | 内容 | 标题建议 | 推荐理由 |",
             "| ---: | ------ | ---: | ---: | ---- | -------- | -------- |"]
    for c in candidates:
        stars = "★" * c.stars + "☆" * (5 - c.stars)
        titles = f"{_cell(c.title_full)}<br>短：{_cell(c.title_short)}"
        reason = _cell(c.reason) + (f"<br>注：{_cell(c.note)}" if c.note else "")
        if c.verify:
            reason += f"<br>**需听音确认**：{_cell(c.verify)}"
        if c.number in overlaps:
            reason += "<br>与 " + "、".join(f"#{n}" for n in overlaps[c.number]) + " 重叠"
        lines.append(f"| {c.number} | {stars} | {format_clock(c.start)}–{format_clock(c.end)} | "
                     f"{c.duration / 60:.1f} 分 | {_cell(c.content)} | {titles} | {reason} |")
    lines += ["", "## 我会优先剪的前三个", ""]
    for rank, pick in enumerate(plan.get("top3") or [], 1):
        c = by_number.get(pick["number"])
        if c:
            lines.append(f"{rank}. #{c.number} {c.title_short}：{pick['why']}")
    lines += ["", "## 这一期的定位", "", plan.get("positioning") or ""]
    single = plan.get("single_pick")
    if single and single["number"] in by_number:
        c = by_number[single["number"]]
        lines += ["", "## 如果只看一个片段", "",
                  f"#{c.number} {c.title_full}（{format_clock(c.start)}–{format_clock(c.end)}）：{single['why']}"]
    lines += ["", "## 切点核对（首尾字幕）", ""]
    for c in candidates:
        first, last = cues[c.start_cue - 1], cues[c.end_cue - 1]
        lines.append(f"- #{c.number}（字幕 {c.start_cue}–{c.end_cue}）开头「{first.text}」… 结尾「{last.text}」")
    if plan.get("warnings"):
        lines += ["", "## 被丢弃的候选", ""] + [f"- {w}" for w in plan["warnings"]]
    return "\n".join(lines) + "\n"


def clip_basename(episode_label: str, candidate: ClipCandidate) -> str:
    return safe_filename(f"#{episode_label}-{candidate.number:02d} {candidate.title_short}", limit=80)


def ffmpeg_cut_command(media: Path, target: Path, start: float, end: float) -> list[str]:
    """Lossless stream copy, like LosslessCut: input seek + duration."""
    return ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(media), "-t", f"{end - start:.3f}",
            "-map", "0:a:0", "-c", "copy", "-map_metadata", "-1", str(target)]


def cut_clip(media: Path, target: Path, start: float, end: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(ffmpeg_cut_command(media, target, start, end), check=True)


def slice_cues(cues: list[Cue], start: float, end: float) -> list[Cue]:
    """Cues overlapping [start, end], shifted to clip time; cues cut by the clip end are dropped."""
    out: list[Cue] = []
    for cue in cues:
        if cue.end <= start or cue.start >= end:
            continue
        if cue.end > end + 0.05:
            continue  # the next sentence heard in the fade-out tail: keep the audio, drop the subtitle
        s, e = max(cue.start, start) - start, min(cue.end, end) - start
        clipped = cue.start < start
        if e - s < (MIN_SLICED_CUE_S if clipped else 0.05):
            continue  # a sliver of the previous sentence cut off by the clip start
        out.append(Cue(len(out) + 1, round(s, 3), round(e, 3), cue.text, cue.translation))
    return out


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_srt(cues: list[Cue]) -> str:
    blocks = []
    for cue in cues:
        body = cue.text + (("\n" + cue.translation.replace(" / ", "\n")) if cue.translation else "")
        blocks.append(f"{cue.index}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{body}\n")
    return "\n".join(blocks)


def media_duration_seconds(media: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(media)],
                             capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


REFINE_BEFORE = 30  # cues shown before the current start
REFINE_AFTER = 8
REFINE_END_BEFORE = 8
REFINE_END_AFTER = 20  # cues shown after the current end


def _window_lines(cues: list[Cue], lo: int, hi: int, marks: dict[int, str]) -> list[str]:
    lo, hi = max(1, lo), min(len(cues), hi)
    out = []
    for cue in cues[lo - 1:hi]:
        mark = marks.get(cue.index, "")
        line = f"#{cue.index} [{format_clock(cue.start)}] {cue.text}" + (f" | {cue.translation}" if cue.translation else "")
        out.append(f"{mark:>8} {line}" if mark else f"         {line}")
    return out


def refine_windows(cues: list[Cue], candidate: ClipCandidate) -> tuple[tuple[int, int], tuple[int, int]]:
    start_window = (max(1, candidate.start_cue - REFINE_BEFORE), min(len(cues), candidate.start_cue + REFINE_AFTER))
    end_window = (max(1, candidate.end_cue - REFINE_END_BEFORE), min(len(cues), candidate.end_cue + REFINE_END_AFTER))
    return start_window, end_window


def build_refine_messages(cues: list[Cue], candidates: list[ClipCandidate], *, prompt: str) -> list[dict[str, str]]:
    parts = []
    for c in candidates:
        (s_lo, s_hi), (e_lo, e_hi) = refine_windows(cues, c)
        marks = {c.start_cue: ">>START", c.end_cue: ">>END"}
        parts.append(f"## 切片 {c.number}：{c.title_short}（当前 {c.start_cue}–{c.end_cue}）\n{c.content}\n"
                     f"### 起点附近\n" + "\n".join(_window_lines(cues, s_lo, s_hi, marks)) + "\n"
                     "### 终点附近\n" + "\n".join(_window_lines(cues, e_lo, e_hi, marks)))
    return [{"role": "system", "content": prompt}, {"role": "user", "content": "\n\n".join(parts)}]


def apply_refinement(
    plan: dict[str, Any],
    response: dict[str, Any],
    cues: list[Cue],
    *,
    media_duration: float | None = None,
) -> dict[str, Any]:
    """Move cue ranges as suggested, but only inside the windows the model was shown."""
    pre, post = plan.get("pre_roll_s", DEFAULT_PRE_ROLL_S), plan.get("post_roll_s", DEFAULT_POST_ROLL_S)
    suggestions = {}
    for item in response.get("items") or []:
        try:
            suggestions[int(item["number"])] = (int(item["start_cue"]), int(item["end_cue"]), _text(item.get("why")))
        except (KeyError, TypeError, ValueError):
            continue
    updated = []
    changes = []
    for c in candidates_from_plan(plan):
        new = suggestions.get(c.number)
        if new:
            (s_lo, s_hi), (e_lo, e_hi) = refine_windows(cues, c)
            start_cue = new[0] if s_lo <= new[0] <= s_hi else c.start_cue
            end_cue = new[1] if e_lo <= new[1] <= e_hi else c.end_cue
            if start_cue <= end_cue and (start_cue, end_cue) != (c.start_cue, c.end_cue):
                start, end = cut_window(cues, start_cue, end_cue, pre_roll=pre, post_roll=post,
                                        media_duration=media_duration)
                if MIN_CLIP_S <= end - start <= MAX_CLIP_S:
                    changes.append({"number": c.number, "from": [c.start_cue, c.end_cue],
                                    "to": [start_cue, end_cue], "why": new[2]})
                    c = ClipCandidate(**{**c.__dict__, "start_cue": start_cue, "end_cue": end_cue,
                                         "start": start, "end": end})
        updated.append(c)
    return {**plan, "candidates": [c.__dict__ for c in updated], "boundary_changes": changes}


def overlap_notes(candidates: list[ClipCandidate], min_overlap_s: float = 5.0) -> dict[int, list[int]]:
    notes: dict[int, list[int]] = {}
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if min(a.end, b.end) - max(a.start, b.start) >= min_overlap_s:
                notes.setdefault(a.number, []).append(b.number)
                notes.setdefault(b.number, []).append(a.number)
    return notes


def parse_adjust(spec: str) -> tuple[int, int, int]:
    """'2:315-411' -> (2, 315, 411)."""
    match = re.fullmatch(r"\s*(\d+)\s*[:：]\s*(\d+)\s*[-–~]\s*(\d+)\s*", spec)
    if not match:
        raise ClipPlanError(f"bad adjust spec {spec!r}, expected N:START-END")
    number, start_cue, end_cue = (int(group) for group in match.groups())
    return number, start_cue, end_cue


def adjust_candidate(
    candidate: ClipCandidate,
    cues: list[Cue],
    start_cue: int,
    end_cue: int,
    *,
    pre_roll: float = DEFAULT_PRE_ROLL_S,
    post_roll: float = DEFAULT_POST_ROLL_S,
    media_duration: float | None = None,
) -> ClipCandidate:
    start, end = cut_window(cues, start_cue, end_cue, pre_roll=pre_roll, post_roll=post_roll,
                            media_duration=media_duration)
    return ClipCandidate(**{**candidate.__dict__, "start_cue": start_cue, "end_cue": end_cue, "start": start, "end": end})


def recompute_times(plan: dict[str, Any], cues: list[Cue], *, media_duration: float | None = None) -> dict[str, Any]:
    """Re-derive cut times from the stored cue ranges with the current cut rule."""
    pre, post = plan.get("pre_roll_s", DEFAULT_PRE_ROLL_S), plan.get("post_roll_s", DEFAULT_POST_ROLL_S)
    updated = [adjust_candidate(c, cues, c.start_cue, c.end_cue, pre_roll=pre, post_roll=post,
                                media_duration=media_duration).__dict__ for c in candidates_from_plan(plan)]
    return {**plan, "candidates": updated}
