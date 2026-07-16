from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


STAGE_ORDER: tuple[str, ...] = (
    "prepare",
    "transcribe",
    "correct",
    "align",
    "split",
    "translate",
    "mimo-proofread",
    "proofread-realign",
    "quality-gate",
    "normalize",
    "export",
)


class StageStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: str
    status: StageStatus
    return_code: int = 0
    summary: str = ""
    done: int | None = None
    total: int | None = None
    current: str = ""


@dataclass(frozen=True, slots=True)
class StageDefinition:
    name: str
    input_attrs: tuple[str, ...] = ()
    any_input_groups: tuple[tuple[str, ...], ...] = ()
    output_attrs: tuple[str, ...] = ()
    force_delete_attrs: tuple[str, ...] = ()
    downstream_stages: tuple[str, ...] = ()
    optional: bool = False


STAGE_DEFINITIONS: dict[str, StageDefinition] = {
    "prepare": StageDefinition(
        name="prepare",
        output_attrs=("audio_path", "segments_manifest"),
        force_delete_attrs=("audio_path", "segments_manifest"),
        downstream_stages=("transcribe", "correct", "align", "split", "translate", "mimo-proofread", "proofread-realign", "quality-gate", "normalize", "export"),
    ),
    "transcribe": StageDefinition(
        name="transcribe",
        input_attrs=("segments_manifest",),
        output_attrs=("transcript_manifest", "transcript_text"),
        force_delete_attrs=("transcript_manifest", "raw_transcript_manifest", "corrected_manifest", "transcript_text"),
        downstream_stages=("correct", "align", "split", "translate", "mimo-proofread", "proofread-realign", "quality-gate", "normalize", "export"),
    ),
    "correct": StageDefinition(
        name="correct",
        input_attrs=("transcript_manifest",),
        output_attrs=("corrected_manifest",),
        force_delete_attrs=("corrected_manifest",),
        downstream_stages=("align", "split", "translate", "mimo-proofread", "proofread-realign", "quality-gate", "normalize", "export"),
        optional=True,
    ),
    "align": StageDefinition(
        name="align",
        input_attrs=("transcript_manifest",),
        output_attrs=("aligned_manifest",),
        force_delete_attrs=("aligned_manifest",),
        downstream_stages=("split", "translate", "mimo-proofread", "proofread-realign", "quality-gate", "normalize", "export"),
    ),
    "split": StageDefinition(
        name="split",
        input_attrs=("aligned_manifest",),
        output_attrs=("split_manifest", "split_srt"),
        force_delete_attrs=("split_manifest", "split_srt"),
        downstream_stages=("translate", "mimo-proofread", "proofread-realign", "quality-gate", "normalize", "export"),
    ),
    "translate": StageDefinition(
        name="translate",
        input_attrs=("split_manifest",),
        output_attrs=("translated_manifest", "translated_srt"),
        force_delete_attrs=("translated_manifest", "translated_srt"),
        downstream_stages=("mimo-proofread", "proofread-realign", "quality-gate", "normalize", "export"),
    ),
    "mimo-proofread": StageDefinition(
        name="mimo-proofread",
        input_attrs=("translated_manifest", "segments_manifest"),
        output_attrs=("mimo_proofread_manifest", "mimo_proofread_report", "mimo_proofread_srt"),
        force_delete_attrs=("mimo_proofread_manifest", "mimo_proofread_report", "mimo_proofread_srt"),
        downstream_stages=("proofread-realign", "quality-gate", "normalize", "export"),
        optional=True,
    ),
    "proofread-realign": StageDefinition(
        name="proofread-realign",
        input_attrs=("mimo_proofread_manifest",),
        output_attrs=("mimo_proofread_manifest",),
        downstream_stages=("quality-gate", "normalize", "export"),
        optional=True,
    ),
    "quality-gate": StageDefinition(
        name="quality-gate",
        any_input_groups=(("transcript_manifest", "split_manifest", "translated_manifest", "mimo_proofread_manifest"),),
        output_attrs=("final_quality_report",),
        force_delete_attrs=("final_quality_report",),
        downstream_stages=("normalize", "export"),
        optional=True,
    ),
    "normalize": StageDefinition(
        name="normalize",
        any_input_groups=(("mimo_proofread_manifest", "translated_manifest", "split_manifest", "transcript_manifest"),),
        output_attrs=("normalized_manifest", "normalized_srt"),
        force_delete_attrs=("normalized_manifest", "normalized_srt"),
        downstream_stages=("export",),
    ),
    "export": StageDefinition(
        name="export",
        any_input_groups=(("normalized_manifest", "mimo_proofread_manifest", "translated_manifest", "split_manifest", "aligned_manifest", "transcript_manifest"),),
        output_attrs=("subtitles_srt", "subtitles_vtt"),
        force_delete_attrs=("subtitles_srt", "subtitles_vtt"),
    ),
}


def get_stage_definition(stage: str) -> StageDefinition:
    try:
        return STAGE_DEFINITIONS[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown stage: {stage}") from exc


def downstream_stages(stage: str) -> tuple[str, ...]:
    return get_stage_definition(stage).downstream_stages


def stage_names_for_run(
    *,
    with_correct: bool = False,
    with_align: bool = False,
    with_split: bool = False,
    with_translate: bool = False,
    with_mimo_proofread: bool = False,
    with_normalize: bool = False,
) -> tuple[str, ...]:
    stages = ["prepare", "transcribe"]
    if with_correct:
        stages.append("correct")
    if with_align:
        stages.append("align")
    if with_split:
        stages.append("split")
    if with_translate:
        stages.append("translate")
    if with_mimo_proofread:
        stages.append("mimo-proofread")
        stages.append("proofread-realign")
    stages.append("quality-gate")
    if with_normalize:
        stages.append("normalize")
    stages.append("export")
    return tuple(stages)
