from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_asr.alignment_state import derive_alignment_state
from qwen_asr.models import WorkPaths
from qwen_asr.stages import STAGE_DEFINITIONS, get_stage_definition
from qwen_asr.storage import read_json


class ArtifactState:
    def __init__(self, work_paths: WorkPaths) -> None:
        self.work_paths = work_paths

    def is_complete(self, stage: str) -> bool:
        method = getattr(self, f"_is_{stage.replace('-', '_')}_complete", None)
        if method is not None:
            return bool(method())
        definition = get_stage_definition(stage)
        return bool(definition.output_attrs) and all(self._path(attr).exists() for attr in definition.output_attrs)

    def missing_inputs(self, stage: str) -> list[str]:
        definition = get_stage_definition(stage)
        missing = [attr for attr in definition.input_attrs if not self._path(attr).exists()]
        for group in definition.any_input_groups:
            if not any(self._path(attr).exists() for attr in group):
                missing.append("|".join(group))
        return missing

    def downstream_outputs(self, stage: str) -> list[Path]:
        paths: list[Path] = []
        for downstream in get_stage_definition(stage).downstream_stages:
            definition = get_stage_definition(downstream)
            paths.extend(self._path(attr) for attr in definition.force_delete_attrs)
        return paths

    def downstream_existing_outputs(self, stage: str) -> list[Path]:
        return [path for path in self.downstream_outputs(stage) if path.exists()]

    def is_outdated(self, stage: str) -> bool:
        definition = get_stage_definition(stage)
        inputs = [self._path(attr) for attr in definition.input_attrs]
        for group in definition.any_input_groups:
            inputs.extend(self._path(attr) for attr in group if self._path(attr).exists())
        outputs = [self._path(attr) for attr in definition.output_attrs if self._path(attr).exists()]
        if stage in {"quality-gate", "normalize", "export"} and outputs and _review_draft_is_dirty(self.work_paths):
            return True
        if not inputs or not outputs:
            return False
        existing_inputs = [path for path in inputs if path.exists()]
        if not existing_inputs:
            return False
        oldest_output = min(path.stat().st_mtime for path in outputs)
        newest_input = max(path.stat().st_mtime for path in existing_inputs)
        return newest_input > oldest_output

    def force_delete_paths(self, stage: str) -> list[Path]:
        definition = get_stage_definition(stage)
        return [self._path(attr) for attr in definition.force_delete_attrs]

    def delete_stage_outputs(self, stage: str) -> None:
        for path in self.force_delete_paths(stage):
            path.unlink(missing_ok=True)

    def delete_downstream_outputs(self, stage: str) -> None:
        for path in self.downstream_existing_outputs(stage):
            path.unlink(missing_ok=True)

    def _path(self, attr: str) -> Path:
        return getattr(self.work_paths, attr)

    def _is_prepare_complete(self) -> bool:
        return self.work_paths.audio_path.exists() and _json_list_count(self.work_paths.segments_manifest) > 0

    def _is_transcribe_complete(self) -> bool:
        segments = _read_list(self.work_paths.segments_manifest)
        transcripts = _read_list(self.work_paths.transcript_manifest)
        if not segments or not transcripts:
            return False
        completed = {
            str(item.get("segment_id"))
            for item in transcripts
            if item.get("status", "completed") == "completed" and str(item.get("text", "")).strip()
        }
        return all(str(item.get("segment_id")) in completed for item in segments)

    def _is_correct_complete(self) -> bool:
        transcripts = _read_list(self.work_paths.transcript_manifest)
        report = _read_list(self.work_paths.corrected_manifest)
        eligible = [
            item for item in transcripts
            if item.get("status", "completed") == "completed" and str(item.get("text", "")).strip()
        ]
        completed = [item for item in report if item.get("status", "completed") == "completed"]
        return bool(eligible) and len(completed) >= len(eligible)

    def _is_align_complete(self) -> bool:
        transcripts = _read_list(self.work_paths.transcript_manifest)
        aligned = _read_list(self.work_paths.aligned_manifest)
        eligible = [
            str(item.get("segment_id"))
            for item in transcripts
            if item.get("status", "completed") == "completed" and str(item.get("text", "")).strip()
        ]
        completed = {
            str(item.get("segment_id"))
            for item in aligned
            if derive_alignment_state(item) in {"completed_exact", "completed_coarse"}
        }
        return bool(eligible) and all(segment_id in completed for segment_id in eligible)

    def _is_split_complete(self) -> bool:
        return _json_dict_count(self.work_paths.split_manifest) > 0

    def _is_translate_complete(self) -> bool:
        payload = read_json(self.work_paths.translated_manifest, default={})
        if not isinstance(payload, dict) or not payload:
            return False
        split_payload = read_json(self.work_paths.split_manifest, default={})
        if isinstance(split_payload, dict) and split_payload:
            expected_keys = {str(key) for key in split_payload.keys()}
            if not expected_keys.issubset({str(key) for key in payload.keys()}):
                return False
            valid_items = [payload.get(key) for key in expected_keys]
        else:
            valid_items = [item for item in payload.values() if isinstance(item, dict)]
        if not valid_items or not all(isinstance(item, dict) for item in valid_items):
            return False
        return all(str(item.get("translated_subtitle", "")).strip() for item in valid_items)

    def _is_normalize_complete(self) -> bool:
        return _quality_gate_allows_formal_outputs(self.work_paths) and _json_dict_count(self.work_paths.normalized_manifest) > 0

    def _is_mimo_proofread_complete(self) -> bool:
        if not self.work_paths.mimo_proofread_manifest.exists() or not self.work_paths.mimo_proofread_report.exists():
            return False
        report = read_json(self.work_paths.mimo_proofread_report, default=None)
        if isinstance(report, list):
            return bool(report) and all(item.get("status") == "completed" for item in report if isinstance(item, dict))
        if not isinstance(report, dict):
            return False
        if report.get("mode") == "two-stage-nearby":
            if int(report.get("stage1_failed", 0) or 0) > 0:
                return False
            if int(report.get("stage2_failed", 0) or 0) > 0:
                return False
            candidate_count = int(report.get("audio_review_candidate_count", report.get("stage1_suspect_count", 0)) or 0)
            completed = int(report.get("stage2_completed", 0) or 0)
            return completed >= candidate_count
        return True

    def _is_export_complete(self) -> bool:
        return _quality_gate_allows_formal_outputs(self.work_paths) and (
            self.work_paths.subtitles_srt.exists() or self.work_paths.subtitles_vtt.exists()
        )

    def _is_quality_gate_complete(self) -> bool:
        if not self.work_paths.final_quality_report.exists():
            return False
        report = read_json(self.work_paths.final_quality_report, default={})
        return isinstance(report, dict) and str(report.get("status", "")).upper() != "FAIL"


def stage_statuses(work_paths: WorkPaths) -> dict[str, dict[str, Any]]:
    state = ArtifactState(work_paths)
    return {
        stage: {
            "complete": state.is_complete(stage),
            "missing_inputs": state.missing_inputs(stage),
            "outdated": state.is_outdated(stage),
        }
        for stage in STAGE_DEFINITIONS
    }


def _read_list(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default=[])
    return payload if isinstance(payload, list) else []


def _json_list_count(path: Path) -> int:
    return len(_read_list(path))


def _json_dict_count(path: Path) -> int:
    payload = read_json(path, default={})
    return len(payload) if isinstance(payload, dict) else 0


def _quality_gate_allows_formal_outputs(work_paths: WorkPaths) -> bool:
    if not work_paths.final_quality_report.exists():
        return True
    report = read_json(work_paths.final_quality_report, default={})
    if not isinstance(report, dict):
        return False
    return str(report.get("status", "")).upper() != "FAIL"


def _review_draft_is_dirty(work_paths: WorkPaths) -> bool:
    path = work_paths.workdir / "drafts" / "web-review.json"
    try:
        payload = read_json(path, default={})
    except (OSError, UnicodeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") == 1 and bool(payload.get("dirty"))
