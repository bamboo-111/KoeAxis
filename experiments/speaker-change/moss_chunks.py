"""Run MOSS-Transcribe-Diarize on fixed-length chunks (speaker labels are chunk-local).

Usage (MOSS venv):
  experiments/moss-transcribe-diarize/.venv/Scripts/python.exe experiments/speaker-change/moss_chunks.py \
      <audio> <out_dir> [--start 0] [--duration 360] [--chunk 120]
Writes <out_dir>/chunks.json: [{chunk, offset, elapsed_s, prompt_len, generated_tokens, segments:[...]}]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from moss_transcribe_diarize.app.model_runner import ModelRunner
from moss_transcribe_diarize.transcript_parser import parse_transcript

MODEL = Path(__file__).resolve().parents[1] / "moss-transcribe-diarize/model-cache/local/MOSS-Transcribe-Diarize"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=360.0)
    ap.add_argument("--chunk", type=float, default=120.0)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = ModelRunner(MODEL, device="cuda", dtype="bf16")
    results = []
    t0 = args.start
    end = args.start + args.duration
    i = 0
    while t0 < end - 1:
        length = min(args.chunk, end - t0)
        wav = out / f"chunk{i:03d}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0}", "-t", f"{length}", "-i", args.audio,
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        started = time.time()
        res = runner.transcribe(wav, max_new_tokens=args.max_new_tokens)
        segs = []
        for s in parse_transcript(res.text):
            segs.append({"start": float(s.start) + t0, "end": float(s.end) + t0, "speaker": s.speaker, "text": s.text})
        results.append({"chunk": i, "offset": t0, "length": length, "elapsed_s": round(time.time() - started, 1),
                        "prompt_len": res.prompt_len, "generated_tokens": res.generated_tokens, "segments": segs})
        print(f"chunk {i} @{t0:.0f}s len={length:.0f}s elapsed={results[-1]['elapsed_s']}s segs={len(segs)}", flush=True)
        t0 += length
        i += 1
    (out / "chunks.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
