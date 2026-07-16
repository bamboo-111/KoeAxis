from __future__ import annotations

from qwen_asr.artifact_state import ArtifactState
from qwen_asr.final_quality_common import fail, passed, warn
from qwen_asr.models import WorkPaths


def stage_checkpoint_check(work_paths: WorkPaths) -> dict[str, object]:
    state = ArtifactState(work_paths)
    failed: list[str] = []
    warned: list[str] = []
    for stage in (
        "transcribe",
        "align",
        "split",
        "translate",
        "mimo-proofread",
        "proofread-realign",
        "normalize",
        "export",
    ):
        definition_complete = state.is_complete(stage)
        if has_checkpoint_artifact(work_paths, stage) and not definition_complete:
            failed.append(stage)
        elif state.is_outdated(stage) and definition_complete:
            warned.append(stage)
    if failed:
        return fail("stage_checkpoint", f"存在中断或未完成 checkpoint：{', '.join(failed)}", stages=failed)
    if warned:
        return warn("stage_checkpoint", f"存在可能过期的阶段产物：{', '.join(warned)}", stages=warned)
    return passed("stage_checkpoint", "阶段 checkpoint 完整性通过")


def has_checkpoint_artifact(work_paths: WorkPaths, stage: str) -> bool:
    if stage == "transcribe":
        return work_paths.transcript_checkpoint_path.exists() or work_paths.transcript_events_path.exists()
    if stage == "align":
        return work_paths.aligned_checkpoint_path.exists() or work_paths.aligned_events_path.exists()
    if stage == "split":
        return work_paths.split_manifest.exists() or work_paths.split_srt.exists()
    if stage == "translate":
        return work_paths.translated_manifest.exists() or work_paths.translated_srt.exists()
    if stage == "mimo-proofread":
        return work_paths.mimo_proofread_report.exists()
    if stage == "proofread-realign":
        return (work_paths.workdir / "reports" / "proofread_realign.json").exists()
    if stage == "normalize":
        return work_paths.normalized_manifest.exists() or work_paths.normalized_srt.exists()
    if stage == "export":
        return work_paths.subtitles_srt.exists() or work_paths.subtitles_vtt.exists()
    return False
