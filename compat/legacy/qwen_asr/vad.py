from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from qwen_asr.models import SilenceRegion, SpeechRegion

LOGGER = logging.getLogger(__name__)


class VADAdapter(Protocol):
    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        ...


class SileroVADAdapter:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 150,
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        try:
            import torch
            from silero_vad import get_speech_timestamps  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "silero-vad is required for VAD. Install dependencies from requirements.txt."
            ) from exc

        LOGGER.info("Running CPU VAD on %s", audio_path)
        audio_tensor = _load_audio_tensor(audio_path)
        timestamps = get_speech_timestamps(
            audio_tensor,
            model=self._load_model(),
            threshold=self.threshold,
            sampling_rate=16000,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )
        del audio_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        regions = [SpeechRegion(float(item["start"]), float(item["end"])) for item in timestamps]
        return normalize_speech_regions(regions)

    _model = None

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return cls._model
        try:
            from silero_vad import load_silero_vad  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "silero-vad is required for VAD. Install dependencies from requirements.txt."
            ) from exc
        cls._model = load_silero_vad()
        return cls._model


class PyannoteONNXV3Adapter:
    def __init__(
        self,
        onset: float = 0.5,
        offset: float = 0.35,
        min_speech_duration_ms: int = 180,
        min_silence_duration_ms: int = 250,
        speech_pad_ms: int = 120,
        model_name: str = "segmentation-3.0",
        show_progress: bool = True,
    ) -> None:
        self.onset = onset
        self.offset = offset
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.model_name = _normalize_pyannote_model_name(model_name)
        self.show_progress = show_progress

    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        try:
            from pyannote_onnx import PyannoteONNX  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyannote_onnx_v3 VAD requires pyannote-onnx and onnxruntime. Install requirements.txt first."
            ) from exc

        LOGGER.info("Running pyannote_onnx_v3 VAD on %s", audio_path)
        model = PyannoteONNX(model_name=self.model_name, show_progress=self.show_progress)
        tracks = list(model.itertracks(str(audio_path), onset=self.onset, offset=self.offset))
        regions = [
            SpeechRegion(float(item["start"]), float(item["stop"]))
            for item in tracks
            if isinstance(item, dict) and "start" in item and "stop" in item
        ]
        regions = _filter_short_regions(regions, self.min_speech_duration_ms / 1000.0)
        regions = _pad_regions(regions, self.speech_pad_ms / 1000.0, audio_path)
        regions = normalize_speech_regions(regions, merge_gap_seconds=self.min_silence_duration_ms / 1000.0)
        LOGGER.info("pyannote_onnx_v3 VAD found %d speech regions", len(regions))
        return regions


def create_vad_adapter(
    backend: str = "pyannote_onnx_v3",
    *,
    threshold: float = 0.5,
    onset: float = 0.5,
    offset: float = 0.35,
    min_speech_duration_ms: int = 180,
    min_silence_duration_ms: int = 250,
    speech_pad_ms: int = 120,
    pyannote_model: str = "segmentation-3.0",
) -> VADAdapter:
    normalized = str(backend or "pyannote_onnx_v3").strip().lower().replace("-", "_")
    if normalized in {"pyannote", "pyannote_onnx", "pyannote_onnx_v3", "segmentation_3_0"}:
        return PyannoteONNXV3Adapter(
            onset=onset,
            offset=offset,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            model_name=pyannote_model,
        )
    if normalized == "silero":
        return SileroVADAdapter(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
    raise ValueError(f"Unsupported VAD backend: {backend}")


def _load_audio_tensor(audio_path: Path):
    import torch

    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16kHz WAV for VAD, got {sample_rate} Hz: {audio_path}")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def normalize_speech_regions(regions: list[SpeechRegion], merge_gap_seconds: float = 0.0) -> list[SpeechRegion]:
    if not regions:
        return []
    ordered = sorted(regions, key=lambda item: item.start_time)
    merged: list[SpeechRegion] = [ordered[0]]
    for region in ordered[1:]:
        current = merged[-1]
        if region.start_time <= current.end_time + merge_gap_seconds:
            current.end_time = max(current.end_time, region.end_time)
            continue
        merged.append(region)
    return merged


def _normalize_pyannote_model_name(model_name: str) -> str:
    value = str(model_name or "segmentation-3.0").strip()
    if value in {"pyannote_onnx_v3", "pyannote-onnx-v3", "segmentation_3_0"}:
        return "segmentation-3.0"
    return value


def _filter_short_regions(regions: list[SpeechRegion], min_duration: float) -> list[SpeechRegion]:
    return [item for item in regions if item.end_time - item.start_time >= min_duration]


def _pad_regions(regions: list[SpeechRegion], pad_seconds: float, audio_path: Path) -> list[SpeechRegion]:
    if pad_seconds <= 0 or not regions:
        return regions
    _, duration = _load_audio_duration(audio_path)
    return [
        SpeechRegion(start_time=max(0.0, item.start_time - pad_seconds), end_time=min(duration, item.end_time + pad_seconds))
        for item in regions
    ]


def _load_audio_duration(audio_path: Path) -> tuple[int, float]:
    info = sf.info(str(audio_path))
    return int(info.samplerate), float(info.frames / info.samplerate)


def derive_silence_regions(
    speech_regions: list[SpeechRegion],
    audio_duration: float,
) -> list[SilenceRegion]:
    if not speech_regions:
        return [SilenceRegion(start_time=0.0, end_time=audio_duration)]

    silences: list[SilenceRegion] = []
    cursor = 0.0
    for region in speech_regions:
        if region.start_time > cursor:
            silences.append(SilenceRegion(start_time=cursor, end_time=region.start_time))
        cursor = max(cursor, region.end_time)
    if cursor < audio_duration:
        silences.append(SilenceRegion(start_time=cursor, end_time=audio_duration))
    return silences
