from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return dataclass_to_dict(value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    raw = asdict(instance)
    return {key: _serialize_value(value) for key, value in raw.items()}


@dataclass(slots=True)
class WorkPaths:
    workdir: Path
    audio_path: Path
    segments_dir: Path
    segments_manifest: Path
    transcript_manifest: Path
    transcript_events_path: Path
    transcript_checkpoint_path: Path
    raw_transcript_manifest: Path
    corrected_manifest: Path
    aligned_manifest: Path
    aligned_events_path: Path
    aligned_checkpoint_path: Path
    split_manifest: Path
    split_srt: Path
    translated_manifest: Path
    translated_srt: Path
    mimo_proofread_dir: Path
    mimo_proofread_manifest: Path
    mimo_proofread_report: Path
    mimo_proofread_srt: Path
    normalized_manifest: Path
    normalized_srt: Path
    transcript_text: Path
    subtitles_srt: Path
    subtitles_vtt: Path
    transcribe_profile_path: Path
    progress_path: Path
    logs_dir: Path
    project_metadata: Path

    @classmethod
    def from_workdir(cls, workdir: Path) -> "WorkPaths":
        workdir = workdir.resolve()
        if workdir.parent.name == "workspaces":
            return cls(
                workdir=workdir,
                audio_path=workdir / "audio" / "source.wav",
                segments_dir=workdir / "audio" / "segments",
                segments_manifest=workdir / "manifests" / "segments.json",
                transcript_manifest=workdir / "manifests" / "transcript_segments.json",
                transcript_events_path=workdir / "manifests" / "transcript_events.jsonl",
                transcript_checkpoint_path=workdir / "manifests" / "transcript_checkpoint.json",
                raw_transcript_manifest=workdir / "manifests" / "transcript_segments.raw.json",
                corrected_manifest=workdir / "manifests" / "corrected_segments.json",
                aligned_manifest=workdir / "manifests" / "aligned_segments.json",
                aligned_events_path=workdir / "manifests" / "aligned_events.jsonl",
                aligned_checkpoint_path=workdir / "manifests" / "aligned_checkpoint.json",
                split_manifest=workdir / "manifests" / "split_segments.json",
                split_srt=workdir / "drafts" / "subtitles.split.srt",
                translated_manifest=workdir / "manifests" / "translated_segments.json",
                translated_srt=workdir / "drafts" / "subtitles.translated.srt",
                mimo_proofread_dir=workdir / "experiments" / "mimo-proofread",
                mimo_proofread_manifest=workdir / "experiments" / "mimo-proofread" / "mimo_proofread_segments.json",
                mimo_proofread_report=workdir / "experiments" / "mimo-proofread" / "mimo_proofread_report.json",
                mimo_proofread_srt=workdir / "experiments" / "mimo-proofread" / "subtitles.mimo-proofread.srt",
                normalized_manifest=workdir / "manifests" / "normalized_segments.json",
                normalized_srt=workdir / "drafts" / "subtitles.normalized.srt",
                transcript_text=workdir / "drafts" / "transcript.txt",
                subtitles_srt=workdir / "export-cache" / "subtitles.srt",
                subtitles_vtt=workdir / "export-cache" / "subtitles.vtt",
                transcribe_profile_path=workdir / "transcribe.profile.json",
                progress_path=workdir / "progress.json",
                logs_dir=workdir / "logs",
                project_metadata=workdir / "project.json",
            )
        return cls(
            workdir=workdir,
            audio_path=workdir / "audio.wav",
            segments_dir=workdir / "segments",
            segments_manifest=workdir / "segments.json",
            transcript_manifest=workdir / "transcript_segments.json",
            transcript_events_path=workdir / "transcript_events.jsonl",
            transcript_checkpoint_path=workdir / "transcript_checkpoint.json",
            raw_transcript_manifest=workdir / "transcript_segments.raw.json",
            corrected_manifest=workdir / "corrected_segments.json",
            aligned_manifest=workdir / "aligned_segments.json",
            aligned_events_path=workdir / "aligned_events.jsonl",
            aligned_checkpoint_path=workdir / "aligned_checkpoint.json",
            split_manifest=workdir / "split_segments.json",
            split_srt=workdir / "subtitles.split.srt",
            translated_manifest=workdir / "translated_segments.json",
            translated_srt=workdir / "subtitles.translated.srt",
            mimo_proofread_dir=workdir / "experiments" / "mimo-proofread",
            mimo_proofread_manifest=workdir / "experiments" / "mimo-proofread" / "mimo_proofread_segments.json",
            mimo_proofread_report=workdir / "experiments" / "mimo-proofread" / "mimo_proofread_report.json",
            mimo_proofread_srt=workdir / "experiments" / "mimo-proofread" / "subtitles.mimo-proofread.srt",
            normalized_manifest=workdir / "normalized_segments.json",
            normalized_srt=workdir / "subtitles.normalized.srt",
            transcript_text=workdir / "transcript.txt",
            subtitles_srt=workdir / "subtitles.srt",
            subtitles_vtt=workdir / "subtitles.vtt",
            transcribe_profile_path=workdir / "transcribe.profile.json",
            progress_path=workdir / "progress.json",
            logs_dir=workdir / "logs",
            project_metadata=workdir / "project.json",
        )


@dataclass(slots=True)
class SpeechRegion:
    start_time: float
    end_time: float


@dataclass(slots=True)
class SilenceRegion:
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def midpoint(self) -> float:
        return self.start_time + (self.duration / 2.0)


@dataclass(slots=True)
class AudioSegment:
    segment_id: str
    audio_path: str
    source_audio_path: str
    global_start_time: float
    global_end_time: float
    duration: float
    logical_start_time: float | None = None
    logical_end_time: float | None = None
    status: str = "pending"
    error: str | None = None


@dataclass(slots=True)
class TranscriptSegment:
    segment_id: str
    audio_path: str
    global_start_time: float
    global_end_time: float
    text: str
    language: str | None = None
    raw_model_output: dict[str, Any] | list[Any] | str | None = None
    status: str = "completed"
    error: str | None = None


@dataclass(slots=True)
class AlignedToken:
    text: str
    start_time: float
    end_time: float


@dataclass(slots=True)
class AlignedSegment:
    segment_id: str
    audio_path: str
    global_start_time: float
    global_end_time: float
    text: str
    language: str | None = None
    tokens: list[AlignedToken] = field(default_factory=list)
    raw_model_output: dict[str, Any] | list[Any] | str | None = None
    status: str = "completed"
    error: str | None = None
