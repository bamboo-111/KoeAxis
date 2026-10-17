"""Run pyannote speaker-diarization-community-1 on a whole episode (or a time window).

Usage (whisperx venv, pyannote.audio 4.x):
  .venv-whisperx312/Scripts/python.exe experiments/speaker-change/pyannote_diarize.py \
      <audio.wav> <out_dir> [--start 0] [--duration 0] [--device cuda]

Speaker count is NOT fixed (the pipeline estimates it). Output <out_dir>/diarization.json:
  {model, audio, offset, duration, elapsed_s, peak_vram_gb,
   regular:   [{start, end, speaker}],   # may overlap (overlapped speech kept)
   exclusive: [{start, end, speaker}],   # one speaker per instant
   speakers:  {label: {total_s, embedding: [...]}}}
Times are absolute (seconds in the source audio). HF token is read from HF_TOKEN.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000


def load_window(path: str, start: float, duration: float) -> tuple[torch.Tensor, int]:
    """Decode via ffmpeg to 16 kHz mono float32 (avoids soundfile/torchcodec in this venv)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start}"]
    if duration > 0:
        cmd += ["-t", f"{duration}"]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    mono = np.frombuffer(raw, dtype=np.float32).copy()
    return torch.from_numpy(mono).unsqueeze(0), SAMPLE_RATE


def annotation_rows(annotation, offset: float) -> list[dict]:
    rows = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append({"start": round(turn.start + offset, 3), "end": round(turn.end + offset, 3), "speaker": speaker})
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = to end of file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-speakers", type=int, default=0, help="fix the speaker count (0 = estimate)")
    ap.add_argument("--min-speakers", type=int, default=0)
    ap.add_argument("--max-speakers", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from pyannote.audio import Pipeline  # lazy: ~4 s import (lightning), not needed by load_window users

    pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    device = torch.device(args.device)
    pipeline.to(device)

    wave, sr = load_window(args.audio, args.start, args.duration)
    length = wave.shape[1] / sr
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    hints = {k: v for k, v in (("num_speakers", args.num_speakers), ("min_speakers", args.min_speakers),
                               ("max_speakers", args.max_speakers)) if v}
    result = pipeline({"waveform": wave, "sample_rate": sr}, **hints)
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0

    regular = annotation_rows(result.speaker_diarization, args.start)
    exclusive = annotation_rows(result.exclusive_speaker_diarization, args.start)
    labels = result.speaker_diarization.labels()
    embeddings = getattr(result, "speaker_embeddings", None)
    speakers = {}
    for idx, label in enumerate(labels):
        total = sum(r["end"] - r["start"] for r in exclusive if r["speaker"] == label)
        entry: dict = {"total_s": round(total, 2)}
        if embeddings is not None and idx < len(embeddings):
            entry["embedding"] = [round(float(x), 6) for x in np.asarray(embeddings[idx]).ravel()]
        speakers[label] = entry

    payload = {
        "model": MODEL,
        "audio": str(Path(args.audio).resolve()),
        "offset": args.start,
        "duration": round(length, 3),
        "elapsed_s": round(elapsed, 2),
        "peak_vram_gb": round(peak, 2),
        "regular": regular,
        "exclusive": exclusive,
        "speakers": speakers,
    }
    (out / "diarization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = ", ".join(f"{k}={v['total_s']}s" for k, v in speakers.items())
    print(f"audio {length:.1f}s elapsed {elapsed:.1f}s (RTF {elapsed / max(length, 1e-6):.3f}) "
          f"peak_vram {peak:.2f}GB speakers {len(speakers)}: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
