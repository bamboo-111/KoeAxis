from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_asr.models import WorkPaths
from qwen_asr.stages import STAGE_DEFINITIONS, get_stage_definition
from qwen_asr.storage import read_json


class ArtifactState:
    def __init__(self, work_paths: WorkPaths) -> None:
        self.work_paths = work_paths

    def is_complete(self, stage: str) -> bool:
        method = getattr(self, f"_is_{stage}_complete", None)
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
            if item.get("status", "completed") == "completed"
        }
        return bool(eligible) and all(segment_id in completed for segment_id in eligible)

    def _is_split_complete(self) -> bool:
        return _json_dict_count(self.work_paths.split_manifest) > 0

    def _is_translate_complete(self) -> bool:
        payload = read_json(self.work_paths.translated_manifest, default={})
        if not isinstance(payload, dict) or not payload:
            return False
        return all(str(item.get("translated_subtitle", "")).strip() for item in payload.values() if isinstance(item, dict))

    def _is_normalize_complete(self) -> bool:
        return _json_dict_count(self.work_paths.normalized_manifest) > 0

    def _is_mimo_proofread_complete(self) -> bool:
        return self.work_paths.mimo_proofread_manifest.exists() and self.work_paths.mimo_proofread_report.exists()

    def _is_export_complete(self) -> bool:
        return self.work_paths.subtitles_srt.exists() or self.work_paths.subtitles_vtt.exists()


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
