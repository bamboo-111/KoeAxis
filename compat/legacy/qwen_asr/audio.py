from __future__ import annotations

import logging
import subprocess
import wave
from pathlib import Path

import soundfile as sf

from qwen_asr.models import AudioSegment
from qwen_asr.storage import ensure_directory

LOGGER = logging.getLogger(__name__)


def extract_audio(
    media_path: Path,
    output_path: Path,
    overwrite: bool = False,
    denoise: bool = False,
    denoise_level: float = 12.0,
    denoise_backend: str = "mdx_net",
    denoise_profile: str = "strong",
    mdx_model: str = "UVR-MDX-NET-Inst_HQ_3.onnx",
    mdx_model_dir: str | None = None,
) -> Path:
    if output_path.exists() and not overwrite:
        LOGGER.info("Reusing extracted audio: %s", output_path)
        return output_path

    ensure_directory(output_path.parent)
    backend = str(denoise_backend or "mdx_net").strip().lower()
    if denoise and backend == "mdx_net":
        raw_path = output_path.with_name(f"{output_path.stem}.raw.wav")
        _extract_audio_with_ffmpeg(media_path, raw_path, overwrite=True)
        vocals_path = separate_vocals_with_mdx_net(
            raw_path,
            output_path.parent / "mdx-net",
            model_name=mdx_model,
            model_dir=Path(mdx_model_dir) if mdx_model_dir else None,
        )
        _normalize_wav_with_ffmpeg(vocals_path, output_path, denoise_level=denoise_level)
        return output_path

    filters: str | None = None
    if denoise:
        if backend not in {"ffmpeg", "legacy", "filter"}:
            raise ValueError(f"Unsupported denoise backend: {denoise_backend}")
        filters = build_denoise_filter(denoise_level, denoise_profile)
    _extract_audio_with_ffmpeg(media_path, output_path, overwrite=True, filters=filters)
    return output_path


def _extract_audio_with_ffmpeg(
    media_path: Path,
    output_path: Path,
    overwrite: bool = False,
    filters: str | None = None,
) -> Path:
    if output_path.exists() and not overwrite:
        LOGGER.info("Reusing extracted audio: %s", output_path)
        return output_path

    ensure_directory(output_path.parent)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if filters:
        command.extend(["-af", filters])
    command.extend(
        [
        "-acodec",
        "pcm_s16le",
        str(output_path),
        ]
    )
    action = "Extracting filtered audio" if filters else "Extracting audio"
    LOGGER.info("%s with ffmpeg from %s", action, media_path)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg audio extraction failed: {exc.stderr}") from exc
    return output_path


def _normalize_wav_with_ffmpeg(source_path: Path, output_path: Path, denoise_level: float = 12.0) -> Path:
    filter_chain = ",".join(
        [
            "highpass=f=70",
            "lowpass=f=7800",
            f"afftdn=nr={max(6.0, float(denoise_level))}:nf=-35",
            "dynaudnorm=f=150:g=15:p=0.95",
        ]
    )
    return _extract_audio_with_ffmpeg(source_path, output_path, overwrite=True, filters=filter_chain)


def separate_vocals_with_mdx_net(
    audio_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    model_dir: Path | None = None,
) -> Path:
    """Run an MDX-Net separator and return the vocal stem path."""
    try:
        from audio_separator.separator import Separator  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MDX-Net denoise requires audio-separator. Install requirements.txt and make sure the MDX model is available."
        ) from exc

    ensure_directory(output_dir)
    if model_dir is not None:
        ensure_directory(model_dir)
    separator_kwargs = {
        "log_level": logging.WARNING,
        "output_dir": str(output_dir),
        "output_format": "WAV",
    }
    if model_dir is not None:
        separator_kwargs["model_file_dir"] = str(model_dir)
    separator = Separator(**separator_kwargs)
    LOGGER.info("Running MDX-Net vocal separation: model=%s input=%s", model_name, audio_path)
    separator.load_model(model_filename=model_name)
    outputs = separator.separate(str(audio_path))
    candidates = [Path(item) for item in outputs]
    resolved = [item if item.is_absolute() else output_dir / item for item in candidates]
    vocal_paths = [
        item
        for item in resolved
        if item.exists() and any(token in item.name.lower() for token in ("vocal", "voice", "speech"))
    ]
    if not vocal_paths:
        existing = [item for item in resolved if item.exists()]
        if len(existing) == 1:
            return existing[0]
        raise RuntimeError(f"MDX-Net did not produce a recognizable vocal stem. Outputs: {outputs}")
    return vocal_paths[0]


def build_denoise_filter(denoise_level: float = 12.0, denoise_profile: str = "strong") -> str:
    """Build the ffmpeg audio filter chain used before VAD/ASR.

    The default legacy behavior was only afftdn=nr=12, which is too weak for
    BGM, room noise, or compressed video audio. These profiles intentionally
    favor intelligibility over speed.
    """
    profile = str(denoise_profile or "strong").strip().lower()
    level = max(0.0, float(denoise_level))

    if profile in {"off", "none", "false"}:
        return "anull"
    if profile in {"light", "legacy"}:
        return f"afftdn=nr={level}"
    if profile == "medium":
        return ",".join(
            [
                "highpass=f=80",
                "lowpass=f=7800",
                f"afftdn=nr={max(level, 18.0)}:nf=-35",
                "dynaudnorm=f=150:g=12:p=0.9",
            ]
        )
    if profile in {"strong", "speech"}:
        return ",".join(
            [
                "highpass=f=90",
                "lowpass=f=7200",
                f"afftdn=nr={max(level, 24.0)}:nf=-38",
                "anlmdn=s=7:p=0.002:r=0.01",
                "dynaudnorm=f=150:g=15:p=0.95",
            ]
        )
    raise ValueError(f"Unsupported denoise profile: {denoise_profile}")


def load_audio_metadata(audio_path: Path) -> tuple[int, float]:
    with wave.open(str(audio_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    duration = frames / float(sample_rate)
    return sample_rate, duration


def export_segment_audio(source_audio_path: Path, segment: AudioSegment) -> Path:
    ensure_directory(Path(segment.audio_path).parent)
    start_frame = int(segment.global_start_time * 16000)
    end_frame = int(segment.global_end_time * 16000)
    frame_count = max(0, end_frame - start_frame)

    with sf.SoundFile(source_audio_path) as source_file:
        source_file.seek(start_frame)
        audio = source_file.read(frame_count, dtype="int16")
        sf.write(segment.audio_path, audio, source_file.samplerate, subtype="PCM_16")

    return Path(segment.audio_path)


def ensure_segment_audio(source_audio_path: Path, segment: AudioSegment) -> Path:
    target_path = Path(segment.audio_path)
    if _is_reusable_segment_audio(target_path):
        return target_path
    return export_segment_audio(source_audio_path, segment)


def _is_reusable_segment_audio(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False
