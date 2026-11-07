from __future__ import annotations

import argparse
import statistics
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from qwen_asr.history_glossary import AssDialogue, parse_ass_dialogues
from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT, load_specific_asr_data
from qwen_asr.storage import ensure_directory, read_json, write_json_atomic


DEFAULT_INCLUDE_STYLES = ("Text - JP", "Text - JP - UP")
DEFAULT_EXCLUDE_STYLE_PREFIXES = ("OP", "ED")
SHORT_DIALOGUE_MAX_NORMALIZED_CHARS = 4
OVERLONG_MATCH_MIN_RATIO = 2.5
OVERLONG_MATCH_MIN_EXTRA_CHARS = 6
SHORT_DIALOGUE_WIDE_MATCH_THRESHOLD = 0.75


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    text: str
    start_ms: int
    end_ms: int
    key: str = ""


def cmd_ass_quality(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    ass_path = Path(args.ass).resolve()
    source = str(args.source)
    report = build_ass_quality_report(
        work_paths,
        ass_path=ass_path,
        source=source,
        include_styles=_csv_tuple(getattr(args, "include_styles", None)) or DEFAULT_INCLUDE_STYLES,
        exclude_style_prefixes=_csv_tuple(getattr(args, "exclude_style_prefixes", None)) or DEFAULT_EXCLUDE_STYLE_PREFIXES,
        optimizer_root=Path(getattr(args, "optimizer_root", DEFAULT_OPTIMIZER_ROOT)),
        offset_ms=getattr(args, "offset_ms", None),
        window_ms=int(getattr(args, "window_ms", 1200)),
        diagnostic_window_ms=int(getattr(args, "diagnostic_window_ms", 8000)),
        low_score_threshold=float(getattr(args, "low_score_threshold", 0.45)),
        fail_score_threshold=float(getattr(args, "fail_score_threshold", 0.20)),
        max_cases=int(getattr(args, "max_cases", 30)),
    )
    output = Path(getattr(args, "output", "") or work_paths.workdir / "reports" / f"ass_quality.{source}.json")
    write_json_atomic(output, report)
    markdown_output = str(getattr(args, "markdown_output", "") or "").strip()
    if markdown_output:
        path = Path(markdown_output)
        ensure_directory(path.parent)
        path.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"ASS \u57fa\u51c6\u8bc4\u4f30{_zh_status(str(report['status']))}\uff1a{output}")
    return 0 if report["status"] != "FAIL" else 1


def build_ass_quality_report(
    work_paths: WorkPaths,
    *,
    ass_path: Path,
    source: str,
    include_styles: tuple[str, ...] = DEFAULT_INCLUDE_STYLES,
    exclude_style_prefixes: tuple[str, ...] = DEFAULT_EXCLUDE_STYLE_PREFIXES,
    optimizer_root: Path = DEFAULT_OPTIMIZER_ROOT,
    offset_ms: int | None = None,
    window_ms: int = 1200,
    diagnostic_window_ms: int = 8000,
    low_score_threshold: float = 0.45,
    fail_score_threshold: float = 0.20,
    max_cases: int = 30,
) -> dict[str, Any]:
    dialogues = select_reference_dialogues(
        parse_ass_dialogues(ass_path),
        include_styles=include_styles,
        exclude_style_prefixes=exclude_style_prefixes,
    )
    cues = load_source_cues(
        work_paths,
        source=source,
        optimizer_root=optimizer_root,
    )
    if not dialogues:
        raise RuntimeError("ASS reference has no selected dialogue lines.")
    if not cues:
        raise RuntimeError(f"No subtitle cues available for source: {source}")

    if offset_ms is None:
        offset_ms = estimate_global_offset_ms(dialogues, cues)
    return evaluate_ass_quality(
        ass_path=ass_path,
        source=source,
        dialogues=dialogues,
        cues=cues,
        offset_ms=int(offset_ms),
        window_ms=window_ms,
        diagnostic_window_ms=diagnostic_window_ms,
        low_score_threshold=low_score_threshold,
        fail_score_threshold=fail_score_threshold,
        max_cases=max_cases,
    )


def select_reference_dialogues(
    dialogues: list[AssDialogue],
    *,
    include_styles: tuple[str, ...] = DEFAULT_INCLUDE_STYLES,
    exclude_style_prefixes: tuple[str, ...] = DEFAULT_EXCLUDE_STYLE_PREFIXES,
) -> list[AssDialogue]:
    selected: list[AssDialogue] = []
    seen: set[tuple[int, int, str]] = set()
    for item in dialogues:
        style = item.style.strip()
        if any(style.startswith(prefix) for prefix in exclude_style_prefixes):
            continue
        if include_styles and style not in include_styles:
            continue
        text = item.text.strip()
        if not normalize_for_match(text):
            continue
        key = (item.start_ms, item.end_ms, text)
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return sorted(selected, key=lambda item: (item.start_ms, item.end_ms))


def load_source_cues(work_paths: WorkPaths, *, source: str, optimizer_root: Path = DEFAULT_OPTIMIZER_ROOT) -> list[SubtitleCue]:
    if source == "export":
        return parse_srt_cues(work_paths.subtitles_srt)
    if source == "aligned":
        return load_aligned_segment_cues(work_paths)
    data = load_specific_asr_data(work_paths, source=source, optimizer_root=optimizer_root)
    if data is None:
        return []
    return [
        SubtitleCue(
            text=str(getattr(segment, "text", "") or ""),
            start_ms=int(getattr(segment, "start_time", 0) or 0),
            end_ms=int(getattr(segment, "end_time", 0) or 0),
            key=str(index),
        )
        for index, segment in enumerate(getattr(data, "segments", []), 1)
        if str(getattr(segment, "text", "") or "").strip()
    ]


def load_aligned_segment_cues(work_paths: WorkPaths) -> list[SubtitleCue]:
    payload = read_json(work_paths.aligned_manifest, default=[])
    cues: list[SubtitleCue] = []
    if not isinstance(payload, list):
        return cues
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        start_ms = int(round(float(item.get("global_start_time", 0.0) or 0.0) * 1000))
        end_ms = int(round(float(item.get("global_end_time", 0.0) or 0.0) * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        cues.append(SubtitleCue(text=text, start_ms=start_ms, end_ms=end_ms, key=str(item.get("segment_id", ""))))
    return cues


def parse_srt_cues(path: Path) -> list[SubtitleCue]:
    if not path.exists():
        return []
    blocks = path.read_text(encoding="utf-8-sig").strip().split("\n\n")
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [line.rstrip("\r") for line in block.splitlines()]
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = lines[1].split(" --> ", 1)
        cues.append(SubtitleCue(text=lines[2], start_ms=_srt_ms(start), end_ms=_srt_ms(end), key=lines[0].strip()))
    return cues


def estimate_global_offset_ms(dialogues: list[AssDialogue], cues: list[SubtitleCue]) -> int:
    anchors: list[tuple[int, float]] = []
    for dialogue in dialogues:
        reference = normalize_for_match(dialogue.text)
        if len(reference) < 6:
            continue
        for cue in cues:
            candidate = normalize_for_match(cue.text)
            if not candidate:
                continue
            score = partial_ratio(reference, candidate)
            if score >= 0.92:
                anchors.append((cue.start_ms - dialogue.start_ms, score))
    if not anchors:
        return 0
    bucket_ms = 500
    buckets: dict[int, list[tuple[int, float]]] = {}
    for offset, score in anchors:
        bucket = round(offset / bucket_ms)
        buckets.setdefault(bucket, []).append((offset, score))
    ranked_buckets = sorted(
        buckets.items(),
        key=lambda item: (len(item[1]), sum(score for _offset, score in item[1])),
        reverse=True,
    )[:5]
    candidates: list[int] = []
    for bucket, _values in ranked_buckets:
        neighboring = [
            pair
            for other_bucket, values in buckets.items()
            if abs(other_bucket - bucket) <= 1
            for pair in values
        ]
        values = [round(offset / 10) * 10 for offset, _score in neighboring]
        candidate = int(statistics.median(values))
        if candidate not in candidates:
            candidates.append(candidate)
    return max(candidates, key=lambda offset: _offset_window_score(dialogues, cues, offset))


def evaluate_ass_quality(
    *,
    ass_path: Path,
    source: str,
    dialogues: list[AssDialogue],
    cues: list[SubtitleCue],
    offset_ms: int,
    window_ms: int,
    diagnostic_window_ms: int,
    low_score_threshold: float,
    fail_score_threshold: float,
    max_cases: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for index, dialogue in enumerate(dialogues, 1):
        target_start = dialogue.start_ms + offset_ms
        target_end = dialogue.end_ms + offset_ms
        nearby = [
            cue for cue in cues
            if min(cue.end_ms, target_end + window_ms) > max(cue.start_ms, target_start - window_ms)
        ]
        diagnostic_nearby = [
            cue for cue in cues
            if min(cue.end_ms, target_end + diagnostic_window_ms) > max(cue.start_ms, target_start - diagnostic_window_ms)
        ]
        best = _best_match(dialogue.text, nearby)
        diagnostic_best = _best_match(dialogue.text, diagnostic_nearby)
        score = best["score"]
        reference_chars = len(normalize_for_match(dialogue.text))
        matched_chars = len(normalize_for_match(best["text"]))
        diagnostics = _row_diagnostics(
            reference_chars=reference_chars,
            matched_chars=matched_chars,
            score=score,
            low_score_threshold=low_score_threshold,
        )
        short_dialogue_wide = _short_dialogue_wide_result(
            reference_chars=reference_chars,
            score=score,
            low_score_threshold=low_score_threshold,
            diagnostic_best=diagnostic_best,
            target_start=target_start,
            target_end=target_end,
        )
        diagnostics.extend(short_dialogue_wide["diagnostics"])
        scores.append(score)
        rows.append(
            {
                "index": index,
                "ass_start_ms": dialogue.start_ms,
                "ass_end_ms": dialogue.end_ms,
                "target_start_ms": target_start,
                "target_end_ms": target_end,
                "ass_text": dialogue.text,
                "matched_text": best["text"],
                "matched_key": best["key"],
                "matched_start_ms": best["start_ms"],
                "matched_end_ms": best["end_ms"],
                "ass_normalized_chars": reference_chars,
                "matched_normalized_chars": matched_chars,
                "match_length_ratio": round(matched_chars / max(reference_chars, 1), 6),
                "diagnostic_matched_text": diagnostic_best["text"],
                "diagnostic_matched_key": diagnostic_best["key"],
                "diagnostic_matched_start_ms": diagnostic_best["start_ms"],
                "diagnostic_matched_end_ms": diagnostic_best["end_ms"],
                "diagnostic_score": round(float(diagnostic_best["score"]), 6),
                "diagnostic_distance_ms": short_dialogue_wide["distance_ms"],
                "score": round(score, 6),
                "level": _level(score, low_score_threshold, fail_score_threshold),
                "diagnostics": diagnostics,
            }
        )
    low_cases = [row for row in rows if row["score"] < low_score_threshold]
    fail_cases = [row for row in rows if row["score"] < fail_score_threshold]
    short_dialogue_misses = [
        row
        for row in rows
        if "short-dialogue-low-score" in row["diagnostics"]
    ]
    overlong_matches = [
        row
        for row in rows
        if "overlong-match" in row["diagnostics"]
    ]
    timing_shifted_short_dialogues = [
        row
        for row in rows
        if "short-dialogue-timing-shifted" in row["diagnostics"]
    ]
    missing_short_dialogues = [
        row
        for row in rows
        if "short-dialogue-missing" in row["diagnostics"]
    ]
    status = "FAIL" if fail_cases else ("WARN" if low_cases else "PASS")
    return {
        "status": status,
        "ass_path": str(ass_path),
        "source": source,
        "offset_ms": offset_ms,
        "selected_dialogue_count": len(dialogues),
        "source_cue_count": len(cues),
        "thresholds": {
            "low_score": low_score_threshold,
            "fail_score": fail_score_threshold,
            "diagnostic_window_ms": diagnostic_window_ms,
            "short_dialogue_wide_match": SHORT_DIALOGUE_WIDE_MATCH_THRESHOLD,
        },
        "summary": {
            "mean": round(statistics.mean(scores), 6) if scores else 0.0,
            "median": round(statistics.median(scores), 6) if scores else 0.0,
            "score_ge_075": sum(score >= 0.75 for score in scores),
            "score_ge_045": sum(score >= low_score_threshold for score in scores),
            "score_lt_045": len(low_cases),
            "score_lt_020": len(fail_cases),
            "short_dialogue_low_score": len(short_dialogue_misses),
            "overlong_match": len(overlong_matches),
            "short_dialogue_timing_shifted": len(timing_shifted_short_dialogues),
            "short_dialogue_missing": len(missing_short_dialogues),
        },
        "rows": rows,
        "worst_cases": sorted(rows, key=lambda row: row["score"])[:max_cases],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# ASS \u57fa\u51c6\u8bc4\u4f30\u62a5\u544a",
        "",
        f"- \u72b6\u6001\uff1a{_zh_status(str(report['status']))}",
        f"- ASS\uff1a{report['ass_path']}",
        f"- \u8bc4\u4f30\u6765\u6e90\uff1a{_zh_source(str(report['source']))}",
        f"- \u91c7\u7528\u5168\u5c40\u504f\u79fb\uff1a{report['offset_ms']} ms",
        f"- \u6b63\u7247\u65e5\u8bed\u5bf9\u767d\u6570\uff1a{report['selected_dialogue_count']}",
        f"- \u88ab\u8bc4\u4f30\u5b57\u5e55\u6570\uff1a{report['source_cue_count']}",
        f"- \u5e73\u5747\u5206\uff1a{summary['mean']}",
        f"- \u4e2d\u4f4d\u5206\uff1a{summary['median']}",
        f"- \u4f4e\u4e8e 0.45\uff1a{summary['score_lt_045']}",
        f"- \u4f4e\u4e8e 0.20\uff1a{summary['score_lt_020']}",
        f"- \u77ed\u5bf9\u767d\u4f4e\u5206\uff1a{summary.get('short_dialogue_low_score', 0)}",
        f"- \u77ed\u5bf9\u767d\u7591\u4f3c\u9519\u65f6\uff1a{summary.get('short_dialogue_timing_shifted', 0)}",
        f"- \u77ed\u5bf9\u767d\u7591\u4f3c\u7f3a\u5931\uff1a{summary.get('short_dialogue_missing', 0)}",
        f"- \u8fc7\u957f\u5339\u914d\uff1a{summary.get('overlong_match', 0)}",
        "",
        "## \u6700\u4f4e\u5206\u6837\u672c",
        "",
    ]
    for item in report["worst_cases"]:
        lines.extend(
            [
                f"### {item['index']}  \u5206\u6570 {item['score']}",
                "",
                f"- ASS \u65f6\u95f4\uff1a{item['ass_start_ms']} - {item['ass_end_ms']}",
                f"- \u5339\u914d\u65f6\u95f4\uff1a{item['matched_start_ms']} - {item['matched_end_ms']}",
                f"- ASS\uff1a{item['ass_text']}",
                f"- \u5339\u914d\uff1a{item['matched_text']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _row_diagnostics(
    *,
    reference_chars: int,
    matched_chars: int,
    score: float,
    low_score_threshold: float,
) -> list[str]:
    diagnostics: list[str] = []
    if reference_chars <= SHORT_DIALOGUE_MAX_NORMALIZED_CHARS and score < low_score_threshold:
        diagnostics.append("short-dialogue-low-score")
    if (
        score >= low_score_threshold
        and reference_chars > 0
        and matched_chars - reference_chars >= OVERLONG_MATCH_MIN_EXTRA_CHARS
        and matched_chars / reference_chars >= OVERLONG_MATCH_MIN_RATIO
    ):
        diagnostics.append("overlong-match")
    return diagnostics


def _short_dialogue_wide_result(
    *,
    reference_chars: int,
    score: float,
    low_score_threshold: float,
    diagnostic_best: dict[str, Any],
    target_start: int,
    target_end: int,
) -> dict[str, Any]:
    if reference_chars > SHORT_DIALOGUE_MAX_NORMALIZED_CHARS or score >= low_score_threshold:
        return {"diagnostics": [], "distance_ms": 0}

    diagnostic_score = float(diagnostic_best.get("score", 0.0) or 0.0)
    diagnostic_chars = len(normalize_for_match(str(diagnostic_best.get("text", "") or "")))
    start_ms = int(diagnostic_best.get("start_ms", 0) or 0)
    end_ms = int(diagnostic_best.get("end_ms", 0) or 0)
    distance_ms = 0
    if end_ms and end_ms < target_start:
        distance_ms = target_start - end_ms
    elif start_ms and start_ms > target_end:
        distance_ms = start_ms - target_end

    if (
        diagnostic_score >= SHORT_DIALOGUE_WIDE_MATCH_THRESHOLD
        and diagnostic_chars >= reference_chars
        and distance_ms > 0
    ):
        return {
            "diagnostics": ["short-dialogue-timing-shifted"],
            "distance_ms": distance_ms,
        }
    return {
        "diagnostics": ["short-dialogue-missing"],
        "distance_ms": distance_ms,
    }


def _zh_status(status: str) -> str:
    return {
        "PASS": "\u901a\u8fc7",
        "WARN": "\u8b66\u544a",
        "FAIL": "\u5931\u8d25",
    }.get(status, status)


def _zh_source(source: str) -> str:
    return {
        "transcript": "\u539f\u59cb\u8bc6\u522b\u7a3f",
        "aligned": "\u5bf9\u9f50\u540e\u7a3f",
        "split": "\u65ad\u53e5\u540e\u7a3f",
        "translated": "\u7ffb\u8bd1\u540e\u7a3f",
        "mimo": "\u97f3\u9891\u590d\u6838\u540e\u7a3f",
        "normalized": "\u65f6\u95f4\u89c4\u8303\u5316\u540e\u7a3f",
        "export": "\u5bfc\u51fa\u5b57\u5e55",
    }.get(source, source)


def normalize_for_match(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(character for character in value if character.isalnum() or _is_japanese(character))


def _normalize_kana_equivalent(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(
        _katakana_to_hiragana(character)
        for character in value
        if character.isalnum() or _is_japanese(character)
    )


def partial_ratio(reference: str, candidate: str) -> float:
    reference = normalize_for_match(reference)
    candidate = normalize_for_match(candidate)
    if not reference or not candidate:
        return 0.0
    if len(reference) > len(candidate):
        reference, candidate = candidate, reference
    if len(candidate) <= len(reference) * 1.3:
        return SequenceMatcher(None, reference, candidate, autojunk=False).ratio()
    window_min = max(1, round(len(reference) * 0.7))
    window_max = min(len(candidate), round(len(reference) * 1.3))
    best = 0.0
    for size in range(window_min, window_max + 1):
        for start in range(0, len(candidate) - size + 1):
            score = SequenceMatcher(None, reference, candidate[start : start + size], autojunk=False).ratio()
            if score > best:
                best = score
    return best


def ass_match_score(reference: str, candidate: str) -> float:
    normalized_reference = normalize_for_match(reference)
    normalized_candidate = normalize_for_match(candidate)
    if not normalized_reference or not normalized_candidate:
        return 0.0
    kana_reference = _normalize_kana_equivalent(reference)
    kana_candidate = _normalize_kana_equivalent(candidate)
    if kana_reference == kana_candidate:
        return 1.0
    if len(normalized_candidate) < max(1, round(len(normalized_reference) * 0.7)):
        return SequenceMatcher(None, normalized_reference, normalized_candidate, autojunk=False).ratio()
    if (
        len(normalized_reference) <= SHORT_DIALOGUE_MAX_NORMALIZED_CHARS
        and len(normalized_candidate) > len(normalized_reference) + OVERLONG_MATCH_MIN_EXTRA_CHARS
    ):
        return SequenceMatcher(None, normalized_reference, normalized_candidate, autojunk=False).ratio()
    return partial_ratio(normalized_reference, normalized_candidate)


def _katakana_to_hiragana(character: str) -> str:
    code = ord(character)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return character


def _best_match(reference: str, cues: list[SubtitleCue]) -> dict[str, Any]:
    best = {"score": 0.0, "text": "", "key": "", "start_ms": 0, "end_ms": 0}
    for cue in cues:
        score = ass_match_score(reference, cue.text)
        if not best["key"] or score > best["score"]:
            best = {"score": score, "text": cue.text, "key": cue.key, "start_ms": cue.start_ms, "end_ms": cue.end_ms}
    return best


def _offset_window_score(dialogues: list[AssDialogue], cues: list[SubtitleCue], offset_ms: int) -> tuple[float, int]:
    scores: list[float] = []
    window_ms = 1200
    for dialogue in dialogues:
        reference = normalize_for_match(dialogue.text)
        if len(reference) < 4:
            continue
        target_start = dialogue.start_ms + offset_ms
        target_end = dialogue.end_ms + offset_ms
        nearby = [
            cue for cue in cues
            if min(cue.end_ms, target_end + window_ms) > max(cue.start_ms, target_start - window_ms)
        ]
        if nearby:
            scores.append(max(partial_ratio(reference, cue.text) for cue in nearby))
        else:
            scores.append(0.0)
    if not scores:
        return (0.0, 0)
    return (statistics.mean(scores), sum(score >= 0.45 for score in scores))


def _level(score: float, low_score_threshold: float, fail_score_threshold: float) -> str:
    if score < fail_score_threshold:
        return "fail"
    if score < low_score_threshold:
        return "low"
    if score < 0.75:
        return "warn"
    return "ok"


def _csv_tuple(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _srt_ms(value: str) -> int:
    hours, minutes, rest = value.strip().split(":")
    seconds, milliseconds = rest.replace(".", ",").split(",")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(milliseconds)


def _is_japanese(character: str) -> bool:
    code = ord(character)
    return 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xFF66 <= code <= 0xFF9D
