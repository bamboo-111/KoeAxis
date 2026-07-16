from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json, write_json_atomic


SHORT_RESPONSES = {
    "はい", "え", "うん", "ううん", "いいえ", "駄目", "だめ", "違う", "痛い",
}
MIN_CONTENT_RETENTION = 0.90
MIN_SEGMENT_RETENTION = 0.60
SENTENCE_UNIT_RE = re.compile(r"[^。！？!?…]+[。！？!?…]*")


@dataclass(frozen=True, slots=True)
class Cue:
    text: str
    start_ms: int
    end_ms: int
    key: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def normalized(self) -> str:
        return normalize_japanese(self.text)


def normalize_japanese(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in value if char.isalnum() or _is_japanese(char))


def evaluate_content_conservation(
    work_paths: WorkPaths,
    *,
    include_export: bool = False,
) -> dict[str, Any]:
    stages: list[tuple[str, list[Cue]]] = [("transcript", _load_transcript(work_paths.transcript_manifest))]
    aligned = _load_aligned(work_paths)
    split = _load_subtitle_manifest(work_paths.split_manifest)
    proofread = _load_subtitle_manifest(work_paths.mimo_proofread_manifest)
    if aligned:
        stages.append(("align", aligned))
    if split:
        stages.append(("split", split))
    if proofread:
        stages.append(("proofread", proofread))
    if include_export and work_paths.subtitles_srt.exists():
        stages.append(("export", _load_srt(work_paths.subtitles_srt)))

    comparisons: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for (source_name, source), (target_name, target) in zip(stages, stages[1:], strict=False):
        comparison = _compare_stages(source_name, source, target_name, target)
        comparisons.append(comparison)
        issues.extend(comparison["issues"])

    status = "FAIL" if any(item["severity"] == "FAIL" for item in issues) else (
        "WARN" if issues else "PASS"
    )
    report = {
        "status": status,
        "reference_stage": "transcript",
        "checked_stages": [name for name, _items in stages],
        "comparisons": comparisons,
        "issues": issues,
        "summary": {
            "fail_count": sum(item["severity"] == "FAIL" for item in issues),
            "warn_count": sum(item["severity"] == "WARN" for item in issues),
        },
    }
    work_paths.content_quality_report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(work_paths.content_quality_report, report)
    return report


def _compare_stages(source_name: str, source: list[Cue], target_name: str, target: list[Cue]) -> dict[str, Any]:
    source_text = "".join(item.normalized for item in source)
    target_text = "".join(item.normalized for item in target)
    matched = sum(block.size for block in SequenceMatcher(None, source_text, target_text, autojunk=False).get_matching_blocks())
    retention = matched / max(1, len(source_text))
    issues: list[dict[str, Any]] = []
    seen_issues: set[tuple[Any, ...]] = set()
    if source_text and retention < MIN_CONTENT_RETENTION:
        _append_issue(
            issues,
            seen_issues,
            _issue("FAIL", "content_retention", source_name, target_name, f"内容保留率 {retention:.3f} 低于 {MIN_CONTENT_RETENTION:.2f}"),
        )

    source_units = [unit for item in source for unit in _semantic_units(item)]
    source_counter = Counter(item.normalized for item in source_units if item.normalized)
    source_duplicates = Counter(text for text, _start, _end in _adjacent_boundary_duplicates(source))
    seen_target_duplicates: Counter[str] = Counter()
    for text, start_ms, end_ms in _adjacent_boundary_duplicates(target):
        seen_target_duplicates[text] += 1
        if seen_target_duplicates[text] <= source_duplicates[text]:
            continue
        _append_issue(
            issues,
            seen_issues,
            _issue("WARN", "introduced_duplicate", source_name, target_name, f"相邻字幕边界出现重复文本：{text}", text=text, start_ms=start_ms, end_ms=end_ms),
        )

    for item in source_units:
        text = item.normalized
        if not text:
            continue
        candidates = [candidate for candidate in target if _time_related(item, candidate)]
        local_text = "".join(candidate.normalized for candidate in candidates)
        if _is_short_response(text) and text not in local_text:
            if text in target_text:
                _append_issue(
                    issues,
                    seen_issues,
                    _issue("WARN", "short_response_timing_shifted", source_name, target_name, f"短应答文本仍存在但时间位置偏移：{item.text}", text=item.text, start_ms=item.start_ms, end_ms=item.end_ms),
                )
            else:
                severity, kind = _proofread_missing_text_issue(source_name, target_name, candidates, "missing_short_response")
                _append_issue(
                    issues,
                    seen_issues,
                    _issue(severity, kind, source_name, target_name, f"短应答在后续阶段消失：{item.text}", text=item.text, start_ms=item.start_ms, end_ms=item.end_ms),
                )
        elif (
            len(text) <= 12
            and source_counter[text] == 1
            and text not in target_text
            and _locally_disappeared(text, local_text)
        ):
            severity, kind = _proofread_missing_text_issue(source_name, target_name, candidates, "missing_unique_text")
            _append_issue(
                issues,
                seen_issues,
                _issue(severity, kind, source_name, target_name, f"独有文本在后续阶段消失：{item.text}", text=item.text, start_ms=item.start_ms, end_ms=item.end_ms),
            )

    if source_name == "transcript" and target_name == "align":
        target_by_key = {item.key: item for item in target if item.key}
        for item in source:
            candidate = target_by_key.get(item.key)
            if not candidate or not item.normalized:
                continue
            ratio = len(candidate.normalized) / len(item.normalized)
            if ratio < MIN_SEGMENT_RETENTION:
                _append_issue(
                    issues,
                    seen_issues,
                    _issue("FAIL", "alignment_fallback_too_short", source_name, target_name, f"对齐结果明显短于原识别：{ratio:.3f}", text=item.text, start_ms=item.start_ms, end_ms=item.end_ms),
                )

    return {
        "source": source_name,
        "target": target_name,
        "source_characters": len(source_text),
        "target_characters": len(target_text),
        "content_retention": round(retention, 6),
        "issues": issues,
    }


def _issue(severity: str, kind: str, source: str, target: str, message: str, **details: Any) -> dict[str, Any]:
    return {"severity": severity, "type": kind, "source": source, "target": target, "message": message, **details}


def _append_issue(issues: list[dict[str, Any]], seen: set[tuple[Any, ...]], issue: dict[str, Any]) -> None:
    key = (
        issue.get("severity"),
        issue.get("type"),
        issue.get("source"),
        issue.get("target"),
        issue.get("text", ""),
        issue.get("start_ms", ""),
        issue.get("end_ms", ""),
    )
    if key in seen:
        return
    seen.add(key)
    issues.append(issue)


def _proofread_missing_text_issue(
    source_name: str,
    target_name: str,
    candidates: list[Cue],
    fallback_kind: str,
) -> tuple[str, str]:
    if source_name == "split" and target_name == "proofread" and _has_audio_proofread_evidence(candidates):
        return "WARN", "proofread_audio_evidence_changed_text"
    return "FAIL", fallback_kind


def _has_audio_proofread_evidence(cues: list[Cue]) -> bool:
    for cue in cues:
        metadata = cue.metadata
        if not isinstance(metadata, dict):
            continue
        history = metadata.get("proofread_history", [])
        if not isinstance(history, list):
            continue
        for entry in history:
            if not isinstance(entry, dict):
                continue
            source = str(entry.get("source", ""))
            evidence = entry.get("evidence")
            changes = entry.get("changes")
            if not source.startswith("mimo-") or not isinstance(evidence, dict) or not evidence:
                continue
            if isinstance(changes, dict) and "original_subtitle" in changes:
                return True
    return False


def _is_short_response(text: str) -> bool:
    return text in SHORT_RESPONSES or (len(text) <= 3 and text in {"はい", "え", "うん", "ううん"})


def _semantic_units(cue: Cue) -> list[Cue]:
    parts = [match.group(0).strip() for match in SENTENCE_UNIT_RE.finditer(cue.text)]
    parts = [part for part in parts if normalize_japanese(part)]
    if len(parts) <= 1:
        return [cue]
    weights = [max(1, len(normalize_japanese(part))) for part in parts]
    total = max(1, sum(weights))
    duration = max(1, cue.end_ms - cue.start_ms)
    result: list[Cue] = []
    consumed = 0
    cursor = cue.start_ms
    for index, (part, weight) in enumerate(zip(parts, weights, strict=True)):
        consumed += weight
        if index == len(parts) - 1:
            end_ms = cue.end_ms
        else:
            end_ms = cue.start_ms + round(duration * consumed / total)
            remaining = len(parts) - index - 1
            end_ms = max(cursor + 1, min(cue.end_ms - remaining, end_ms))
        result.append(Cue(part.rstrip("。！？!?…"), cursor, end_ms, cue.key, cue.metadata))
        cursor = end_ms
    return result


def _locally_disappeared(source_text: str, local_text: str) -> bool:
    if not local_text:
        return True
    matched = sum(
        block.size
        for block in SequenceMatcher(None, source_text, local_text, autojunk=False).get_matching_blocks()
    )
    return len(local_text) < len(source_text) * 0.5 and matched < len(source_text) * 0.25


def _adjacent_boundary_duplicates(cues: list[Cue]) -> list[tuple[str, int, int]]:
    duplicates: list[tuple[str, int, int]] = []
    for previous, current in zip(cues, cues[1:], strict=False):
        if current.start_ms - previous.end_ms > 500:
            continue
        left = previous.normalized
        right = current.normalized
        if not left or not right:
            continue
        overlap = ""
        if left == right and len(left) >= 2:
            overlap = left
        else:
            maximum = min(len(left), len(right), 20)
            minimum = 2 if current.start_ms < previous.end_ms else 3
            for size in range(maximum, minimum - 1, -1):
                if left[-size:] == right[:size]:
                    overlap = right[:size]
                    break
        if overlap:
            duplicates.append((overlap, previous.start_ms, current.end_ms))
    return duplicates


def _time_related(source: Cue, target: Cue) -> bool:
    margin = 500
    return target.end_ms >= source.start_ms - margin and target.start_ms <= source.end_ms + margin


def _load_transcript(path: Path) -> list[Cue]:
    payload = read_json(path, default=[])
    if not isinstance(payload, list):
        return []
    return [Cue(str(item.get("text", "")), _seconds_ms(item.get("global_start_time")), _seconds_ms(item.get("global_end_time")), str(item.get("segment_id", "")), item) for item in payload if isinstance(item, dict) and item.get("status", "completed") == "completed"]


def _load_aligned(work_paths: WorkPaths) -> list[Cue]:
    actual_source = _load_aligned_split_source(work_paths)
    if actual_source:
        return actual_source
    payload = read_json(work_paths.aligned_manifest, default=[])
    if not isinstance(payload, list):
        return []
    result = []
    for item in payload:
        if not isinstance(item, dict) or item.get("status", "completed") != "completed":
            continue
        tokens = item.get("tokens", [])
        text = str(item.get("text", "")) or "".join(str(token.get("text", "")) for token in tokens if isinstance(token, dict))
        result.append(Cue(text, _seconds_ms(item.get("global_start_time")), _seconds_ms(item.get("global_end_time")), str(item.get("segment_id", "")), item))
    return result


def _load_aligned_split_source(work_paths: WorkPaths) -> list[Cue]:
    if not work_paths.aligned_manifest.exists():
        return []
    try:
        from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT, _load_optimizer_types, aligned_manifest_to_asr_data

        ASRData, ASRDataSeg, _SubtitleSplitter = _load_optimizer_types(DEFAULT_OPTIMIZER_ROOT)
        asr_data = aligned_manifest_to_asr_data(work_paths, ASRData, ASRDataSeg)
    except Exception:
        return []
    return [
        Cue(
            str(segment.text),
            int(segment.start_time),
            int(segment.end_time),
            str(index),
        )
        for index, segment in enumerate(getattr(asr_data, "segments", []), 1)
        if str(getattr(segment, "text", "")).strip()
    ]


def _load_subtitle_manifest(path: Path) -> list[Cue]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return []
    return [Cue(str(item.get("original_subtitle", "")), int(item.get("start_time", 0) or 0), int(item.get("end_time", 0) or 0), str(key), item) for key, item in payload.items() if isinstance(item, dict)]


def _load_srt(path: Path) -> list[Cue]:
    blocks = re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip())
    result: list[Cue] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = lines[1].split(" --> ", 1)
        result.append(Cue(lines[2], _srt_ms(start), _srt_ms(end), lines[0].strip()))
    return result


def _seconds_ms(value: Any) -> int:
    return int(round(float(value or 0) * 1000))


def _srt_ms(value: str) -> int:
    hours, minutes, rest = value.strip().split(":")
    seconds, milliseconds = rest.replace(".", ",").split(",")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(milliseconds)


def _is_japanese(char: str) -> bool:
    code = ord(char)
    return 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xFF66 <= code <= 0xFF9D
