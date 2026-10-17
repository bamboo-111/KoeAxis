"""Who is speaking when: enrolled-voiceprint scores per 10 ms frame (ReDimNet-M sliding windows).

The enrollment file lists the program's members with one L2-normalised embedding centroid each, computed from
hand-labelled subtitles of earlier episodes (see experiments/speaker-change). For an episode, every window of
``win`` seconds (hop ``hop``) is embedded and compared to the centroids by cosine; each 10 ms frame gets the mean
cosine of the windows covering it. Members whose voice is (almost) never the confident best match are treated as
absent and dropped before scoring, so a two-person episode is only ever labelled with those two people.

No voice-activity detection is needed: callers only look at the frames under ASR tokens.
"""
from __future__ import annotations

import hashlib
import json
import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
STEP = 0.01  # seconds per score frame
CACHE_VERSION = 1
DEFAULT_WEIGHTS = PROJECT_ROOT / ".model-cache" / "speaker" / "redimnet" / "M-vb2+vox2+cnc-ft_mix.pt"
WEIGHTS_URL = "https://github.com/IDRnD/ReDimNet/releases/download/latest/M-vb2+vox2+cnc-ft_mix.pt"
# Present-member detection (validated on episodes 15/16/18/30 with leave-one-episode-out centroids, no VAD):
# present members had >= 3.6 % confident windows and a 95th-percentile cosine >= 0.55; absent ones <= 0.5 % / 0.47.
DETECT_MIN_SHARE = 0.02
DETECT_MIN_P95 = 0.52
DETECT_MARGIN = 0.1


@dataclass
class Enrollment:
    members: list[str]
    centroids: np.ndarray  # (members, dim), L2-normalised
    model: str = "redimnet_m"
    win: float = 0.75
    hop: float = 0.2
    digest: str = ""
    ass: dict[str, Any] = field(default_factory=dict)  # optional {"script_info": {...}, "styles": {member: "Style: ..."}}

    def subset(self, names: list[str]) -> Enrollment:
        index = [self.members.index(name) for name in names]
        return Enrollment(list(names), self.centroids[index], self.model, self.win, self.hop, self.digest, self.ass)


@dataclass
class SpeakerTrack:
    members: list[str]
    scores: np.ndarray  # (frames, members) mean cosine, frame i covers [i * step, (i + 1) * step)
    step: float = STEP
    info: dict[str, Any] = field(default_factory=dict)

    def span_scores(self, start_s: float, end_s: float, pad: float = 0.05) -> np.ndarray:
        """Mean member scores over [start - pad, end + pad] (zeros outside the audio)."""
        n = len(self.scores)
        a = max(int((start_s - pad) / self.step), 0)
        b = min(max(int((end_s + pad) / self.step) + 1, a + 1), n)
        if b <= a:
            return np.zeros(len(self.members), np.float32)
        return self.scores[a:b].astype(np.float32).mean(0)

    def save(self, path: Path, key: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            scores=self.scores.astype(np.float16),
            members=np.array(self.members),
            step=self.step,
            meta=json.dumps({"key": key, "info": self.info}, ensure_ascii=False),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, key: dict[str, Any] | None = None) -> SpeakerTrack | None:
        """The cached track, or None when missing, unreadable or computed with other inputs than ``key``."""
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if key is not None and meta.get("key") != key:
                    return None
                return cls([str(m) for m in z["members"]], z["scores"].astype(np.float32), float(z["step"]),
                           meta.get("info") or {})
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning("Ignoring unreadable speaker score cache %s: %s", path, exc)
            return None


def speaker_scores_path(manifests_dir: Path) -> Path:
    return manifests_dir / "speaker_scores.npz"


def load_enrollment(path: Path | str) -> Enrollment:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    members = [str(name) for name in data["members"]]
    centroids = np.asarray([data["members"][name] for name in members], dtype=np.float32)
    if centroids.ndim != 2 or not len(members):
        raise ValueError(f"Speaker enrollment {path} has no member centroids")
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return Enrollment(
        members=members,
        centroids=centroids,
        model=str(data.get("model", "redimnet_m")),
        win=float(data.get("win", 0.75)),
        hop=float(data.get("hop", 0.2)),
        digest=_voiceprint_digest(data),
        ass=data.get("ass") if isinstance(data.get("ass"), dict) else {},
    )


def _voiceprint_digest(data: dict[str, Any]) -> str:
    """Hash of what the scores depend on (centroids, model, windows), not of display settings such as ASS styles."""
    keyed = {k: data.get(k) for k in ("members", "model", "weights", "win", "hop")}
    return hashlib.sha256(json.dumps(keyed, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def read_wav_16k(path: Path | str) -> np.ndarray:
    """Mono float32 samples at 16 kHz (the workspace's audio/source.wav already is 16 kHz mono PCM16)."""
    with wave.open(str(path), "rb") as source:
        rate, channels, width = source.getframerate(), source.getnchannels(), source.getsampwidth()
        raw = source.readframes(source.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM wav is supported (sample width {width})")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(1)
    if rate != SAMPLE_RATE:
        import torch
        import torchaudio.functional as AF

        samples = AF.resample(torch.from_numpy(samples), rate, SAMPLE_RATE).numpy()
    return samples


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_redimnet(weights: Path | str | None = None, device: str = "cpu"):
    import torch

    from qwen_asr.speaker.redimnet.model import ReDimNetWrap

    path = Path(weights) if weights else DEFAULT_WEIGHTS
    if not path.exists():
        raise FileNotFoundError(f"ReDimNet weights not found: {path} (download {WEIGHTS_URL})")
    full = torch.load(path, map_location="cpu", weights_only=False)
    model = ReDimNetWrap(**full["model_config"])
    result = model.load_state_dict(full["state_dict"])
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"ReDimNet weights do not match the model: {result}")
    return model.eval().to(device)


def window_embeddings(
    samples: np.ndarray,
    model,
    *,
    win: float,
    hop: float,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """L2-normalised embeddings of windows starting at 0, hop, 2*hop, ... (fp16 autocast on CUDA)."""
    import torch

    w, h = int(win * SAMPLE_RATE), int(hop * SAMPLE_RATE)
    x = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    if len(x) < w:
        x = torch.nn.functional.pad(x, (0, w - len(x)))
    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        model = model.to(memory_format=torch.channels_last)
    frames = x.to(device).unfold(0, w, h)  # (windows, w) view, no copy
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=on_cuda):
        for k in range(0, len(frames), batch_size):
            out.append(model(frames[k : k + batch_size].contiguous()).float().cpu())
    emb = torch.cat(out).numpy()
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)


def detect_members(
    window_scores: np.ndarray,
    members: list[str],
    *,
    min_share: float = DETECT_MIN_SHARE,
    min_p95: float = DETECT_MIN_P95,
    margin: float = DETECT_MARGIN,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Members heard in the episode: share of confidently-best windows and 95th-percentile cosine both high."""
    stats: dict[str, dict[str, float]] = {}
    if len(members) < 2 or not len(window_scores):
        return list(members), stats
    top = np.sort(window_scores, 1)
    best, gap = window_scores.argmax(1), top[:, -1] - top[:, -2]
    for j, name in enumerate(members):
        share = float(np.mean((best == j) & (gap > margin)))
        p95 = float(np.percentile(window_scores[:, j], 95))
        stats[name] = {"share": round(share, 4), "p95": round(p95, 4)}
    present = [name for name in members if stats[name]["share"] >= min_share and stats[name]["p95"] >= min_p95]
    if not present:  # nothing confident (very short or noisy audio): keep everyone rather than guess
        present = list(members)
    return present, stats


def frame_scores(window_scores: np.ndarray, n_frames: int, *, win: float, hop: float) -> np.ndarray:
    """Per 10 ms frame, the mean score of the windows covering it (difference arrays, O(frames))."""
    starts = (np.arange(len(window_scores)) * hop / STEP).round().astype(int)
    length = int(round(win / STEP))
    m = window_scores.shape[1]
    acc = np.zeros((n_frames + 1, m), np.float64)
    cover = np.zeros(n_frames + 1, np.float64)
    a = np.minimum(starts, n_frames)
    b = np.minimum(starts + length, n_frames)
    np.add.at(acc, a, window_scores)
    np.add.at(acc, b, -window_scores)
    np.add.at(cover, a, 1)
    np.add.at(cover, b, -1)
    total = np.cumsum(acc, 0)[:n_frames]
    count = np.cumsum(cover)[:n_frames]
    return (total / np.maximum(count, 1)[:, None]).astype(np.float32)


def compute_speaker_track(
    audio_path: Path | str,
    enrollment: Enrollment,
    *,
    speakers: list[str] | None = None,
    weights: Path | str | None = None,
    device: str = "auto",
) -> SpeakerTrack:
    unknown = [name for name in speakers or [] if name not in enrollment.members]
    if unknown:
        raise ValueError(f"Speakers not in the enrollment file: {', '.join(unknown)}")
    device = resolve_device(device)
    samples = read_wav_16k(audio_path)
    model = load_redimnet(weights, device)
    try:
        emb = window_embeddings(samples, model, win=enrollment.win, hop=enrollment.hop, device=device)
    finally:
        del model
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    window_scores = emb @ enrollment.centroids.T
    present, stats = detect_members(window_scores, enrollment.members)
    chosen = list(speakers) if speakers else present
    index = [enrollment.members.index(name) for name in chosen]
    n_frames = int(len(samples) / SAMPLE_RATE / STEP) + 1
    scores = frame_scores(window_scores[:, index], n_frames, win=enrollment.win, hop=enrollment.hop)
    info = {
        "members": chosen,
        "detected": present,
        "manual": bool(speakers),
        "detection": stats,
        "windows": int(len(emb)),
        "device": device,
    }
    LOGGER.info(
        "Speaker track: %s (%s); window stats %s",
        ", ".join(chosen),
        "manual" if speakers else "auto-detected",
        ", ".join(f"{k}: {v['share'] * 100:.1f}% p95 {v['p95']:.2f}" for k, v in stats.items()),
    )
    return SpeakerTrack(chosen, scores, STEP, info)


def cache_key(audio_path: Path | str, enrollment: Enrollment, speakers: list[str] | None) -> dict[str, Any]:
    stat = Path(audio_path).stat()
    return {
        "version": CACHE_VERSION,
        "audio_size": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "enrollment": enrollment.digest,
        "speakers": list(speakers or []),
    }


def ensure_speaker_track(
    audio_path: Path | str,
    manifests_dir: Path,
    enrollment_path: Path | str,
    *,
    speakers: list[str] | None = None,
    weights: Path | str | None = None,
    device: str = "auto",
) -> SpeakerTrack:
    """Cached speaker track of the workspace audio (recomputed when the audio, enrollment or speakers change)."""
    enrollment = load_enrollment(enrollment_path)
    key = cache_key(audio_path, enrollment, speakers)
    path = speaker_scores_path(manifests_dir)
    track = SpeakerTrack.load(path, key)
    if track is not None:
        return track
    track = compute_speaker_track(audio_path, enrollment, speakers=speakers, weights=weights, device=device)
    track.save(path, key)
    return track


def parse_speakers(value: str | None) -> list[str] | None:
    names = [name.strip() for name in str(value or "").replace("，", ",").split(",") if name.strip()]
    return names or None
