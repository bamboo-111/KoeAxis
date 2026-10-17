"""Program memory: per-show fixes for misrecognitions that recur every episode.

A rules file (JSON) lists regex replacements applied inside one subtitle cue, so timing and
cue count never change. A translation rule may require the (already fixed) source line to match
``when_source``; that keeps e.g. "红色的竞赛" untouched unless the Japanese line really names the
concert. ``flags`` only report cues that still need a human (cross-cue fragments and the like).

Rules file layout::

    {
      "rules": [
        {"id": "...", "field": "source" | "translation", "pattern": "...", "replace": "...",
         "when_source": "optional regex"}
      ],
      "flags": [
        {"id": "...", "message": "...", "source": "optional regex", "translation": "optional regex",
         "translation_not": "optional regex"}
      ]
    }

All listed conditions of a flag must hold for the cue to be reported.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qwen_asr.storage import read_json, write_json_atomic

if TYPE_CHECKING:
    from qwen_asr.models import WorkPaths

LOGGER = logging.getLogger(__name__)

FIELDS = ("source", "translation")


@dataclass(frozen=True)
class MemoryRule:
    rule_id: str
    field: str
    pattern: re.Pattern[str]
    replace: str
    when_source: re.Pattern[str] | None = None


@dataclass(frozen=True)
class MemoryFlag:
    flag_id: str
    message: str
    source: re.Pattern[str] | None = None
    translation: re.Pattern[str] | None = None
    translation_not: re.Pattern[str] | None = None

    def matches(self, source: str, translation: str) -> bool:
        if self.source is None and self.translation is None and self.translation_not is None:
            return False
        if self.source is not None and not self.source.search(source):
            return False
        if self.translation is not None and not self.translation.search(translation):
            return False
        if self.translation_not is not None and self.translation_not.search(translation):
            return False
        return True


@dataclass(frozen=True)
class ProgramMemory:
    rules: tuple[MemoryRule, ...] = ()
    flags: tuple[MemoryFlag, ...] = ()


@dataclass
class CueResult:
    source: str
    translation: str
    hits: list[dict[str, str]] = field(default_factory=list)
    flags: list[dict[str, str]] = field(default_factory=list)


def _compile(value: Any, where: str) -> re.Pattern[str] | None:
    if value in (None, ""):
        return None
    try:
        return re.compile(str(value))
    except re.error as exc:
        raise ValueError(f"invalid regex in {where}: {exc}") from exc


def parse_program_memory(data: dict[str, Any]) -> ProgramMemory:
    rules = []
    for index, raw in enumerate(data.get("rules") or []):
        rule_id = str(raw.get("id") or f"rule-{index + 1}")
        field_name = str(raw.get("field") or "")
        if field_name not in FIELDS:
            raise ValueError(f"rule {rule_id}: field must be one of {FIELDS}")
        pattern = _compile(raw.get("pattern"), f"rule {rule_id}")
        if pattern is None:
            raise ValueError(f"rule {rule_id}: pattern is required")
        rules.append(
            MemoryRule(
                rule_id=rule_id,
                field=field_name,
                pattern=pattern,
                replace=str(raw.get("replace", "")),
                when_source=_compile(raw.get("when_source"), f"rule {rule_id}"),
            )
        )
    flags = []
    for index, raw in enumerate(data.get("flags") or []):
        flag_id = str(raw.get("id") or f"flag-{index + 1}")
        flags.append(
            MemoryFlag(
                flag_id=flag_id,
                message=str(raw.get("message") or flag_id),
                source=_compile(raw.get("source"), f"flag {flag_id}"),
                translation=_compile(raw.get("translation"), f"flag {flag_id}"),
                translation_not=_compile(raw.get("translation_not"), f"flag {flag_id}"),
            )
        )
    return ProgramMemory(rules=tuple(rules), flags=tuple(flags))


def load_program_memory(path: str | Path) -> ProgramMemory:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"program memory file must contain a JSON object: {path}")
    return parse_program_memory(data)


def apply_to_cue(source: str, translation: str, memory: ProgramMemory) -> CueResult:
    """Source rules run first so translation rules can check the corrected source line."""
    result = CueResult(source=source, translation=translation)
    for wanted in FIELDS:
        for rule in memory.rules:
            if rule.field != wanted:
                continue
            if rule.when_source is not None and not rule.when_source.search(result.source):
                continue
            before = getattr(result, wanted)
            after = rule.pattern.sub(rule.replace, before)
            if after != before:
                setattr(result, wanted, after)
                result.hits.append({"rule": rule.rule_id, "field": wanted, "before": before, "after": after})
    for flag in memory.flags:
        if flag.matches(result.source, result.translation):
            result.flags.append({"flag": flag.flag_id, "message": flag.message})
    return result


def apply_to_manifest(manifest: dict[str, Any], memory: ProgramMemory) -> dict[str, Any]:
    """Fix ``original_subtitle``/``translated_subtitle`` in a proofread manifest in place; return a report."""
    changes: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for key, item in manifest.items():
        if not isinstance(item, dict):
            continue
        result = apply_to_cue(
            str(item.get("original_subtitle") or ""),
            str(item.get("translated_subtitle") or ""),
            memory,
        )
        if result.hits:
            item["original_subtitle"] = result.source
            item["translated_subtitle"] = result.translation
            changes.append({"id": str(key), "hits": result.hits})
        if result.flags:
            flagged.append({"id": str(key), "source": result.source, "translation": result.translation,
                            "flags": result.flags})
    return build_report(changes, flagged)


_SRT_TIMING = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")


def apply_to_srt_text(text: str, memory: ProgramMemory) -> tuple[str, dict[str, Any]]:
    """Fix a bilingual SRT (line 1 source, line 2 translation). Line endings are preserved."""
    newline = "\r\n" if "\r\n" in text else "\n"
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip("\r\n"))
    changes: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    out_blocks = []
    for block in blocks:
        lines = re.split(r"\r?\n", block)
        if len(lines) >= 3 and _SRT_TIMING.match(lines[1]):
            cue_id = lines[0].strip()
            source = lines[2]
            translation = "\n".join(lines[3:])
            result = apply_to_cue(source, translation, memory)
            if result.hits:
                changes.append({"id": cue_id, "hits": result.hits})
                lines = lines[:2] + [result.source] + (result.translation.split("\n") if len(lines) > 3 else [])
            if result.flags:
                flagged.append({"id": cue_id, "source": result.source, "translation": result.translation,
                                "flags": result.flags})
        out_blocks.append(newline.join(lines))
    return (newline * 2).join(out_blocks) + newline, build_report(changes, flagged)


def build_report(changes: list[dict[str, Any]], flagged: list[dict[str, Any]]) -> dict[str, Any]:
    rule_counts: dict[str, int] = {}
    for change in changes:
        for hit in change["hits"]:
            rule_counts[hit["rule"]] = rule_counts.get(hit["rule"], 0) + 1
    return {
        "changed_cue_count": len(changes),
        "flagged_cue_count": len(flagged),
        "rule_hit_counts": rule_counts,
        "changes": changes,
        "flagged": flagged,
    }


def apply_to_work_paths(work_paths: WorkPaths, memory_file: Path) -> dict[str, Any]:
    """Fix ``normalized_segments.json`` (and rewrite ``subtitles.normalized.srt``); write reports/program_memory.json.

    Runs right after normalize, whose cues are whole display sentences: the proofread manifest is
    still word fragments (共済を on its own, 赤色の in the previous cue), so rules see far less there.
    ``source_reverts`` records, per normalized cue, the Japanese line before and after the fix; the
    normalize content check undoes exactly those cues, so only these deliberate edits are exempt.
    A previously exported subtitle file is deleted when text changes: it no longer matches and the
    export stage rewrites it. Rules are idempotent, so re-running is harmless.
    """
    memory = load_program_memory(memory_file)
    path = work_paths.normalized_manifest
    manifest = read_json(path, default=None) if path.exists() else None
    report: dict[str, Any] = {"memory_file": str(memory_file), "manifest": None, **build_report([], [])}
    if isinstance(manifest, dict) and manifest:
        report.update(apply_to_manifest(manifest, memory))
        report["manifest"] = str(path)
        if report["changed_cue_count"]:
            write_json_atomic(path, manifest)
            _rewrite_normalized_srt(work_paths, manifest)
            for stale in (work_paths.subtitles_srt, work_paths.subtitles_vtt):
                stale.unlink(missing_ok=True)
    previous = read_json(_report_path(work_paths), default={}) if _report_path(work_paths).exists() else {}
    report["source_reverts"] = _merge_source_reverts(previous, report, manifest if isinstance(manifest, dict) else {})
    write_json_atomic(_report_path(work_paths), report)
    return report


def source_reverts(work_paths: WorkPaths) -> dict[str, dict[str, str]]:
    """Normalized cue id -> {"before", "after"} Japanese text program memory changed."""
    path = _report_path(work_paths)
    if not path.exists():
        return {}
    report = read_json(path, default={})
    reverts = report.get("source_reverts") if isinstance(report, dict) else None
    if not isinstance(reverts, dict):
        return {}
    return {
        str(key): {"before": str(value["before"]), "after": str(value["after"])}
        for key, value in reverts.items()
        if isinstance(value, dict) and "before" in value and "after" in value
    }


def _report_path(work_paths: WorkPaths) -> Path:
    return work_paths.workdir / "reports" / "program_memory.json"


def _merge_source_reverts(
    previous: dict[str, Any], report: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Keep earlier records whose cue still shows the fixed text (an idempotent re-run changes nothing)."""
    reverts: dict[str, dict[str, str]] = {}
    for key, value in (previous.get("source_reverts") or {}).items():
        item = manifest.get(str(key))
        if isinstance(value, dict) and isinstance(item, dict) and item.get("original_subtitle") == value.get("after"):
            reverts[str(key)] = {"before": str(value["before"]), "after": str(value["after"])}
    for change in report.get("changes", []):
        source_hits = [hit for hit in change.get("hits", []) if hit.get("field") == "source"]
        if source_hits:
            key = str(change["id"])
            before = reverts.get(key, {}).get("before", source_hits[0]["before"])
            reverts[key] = {"before": before, "after": source_hits[-1]["after"]}
    return reverts


def _rewrite_normalized_srt(work_paths: WorkPaths, manifest: dict[str, Any]) -> None:
    from optimizer.asr_data import ASRData

    work_paths.normalized_srt.write_text(ASRData.from_json(manifest).to_srt(), encoding="utf-8")


def apply_if_requested(args: Any, work_paths: WorkPaths) -> dict[str, Any] | None:
    """Run right after normalize (CLI stage and PipelineRunner) when --program-memory-file is set."""
    memory_file = getattr(args, "program_memory_file", None)
    if not memory_file:
        return None
    report = apply_to_work_paths(work_paths, Path(memory_file))
    LOGGER.info(
        "Program memory fixed %d cue(s), flagged %d for review in %s",
        report["changed_cue_count"],
        report["flagged_cue_count"],
        report.get("manifest") or "no subtitle manifest",
    )
    return report
