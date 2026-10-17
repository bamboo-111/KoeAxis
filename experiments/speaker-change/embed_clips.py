"""Embed 声ログ clips in short chunks: is a clip single-speaker, and whose voice is it closest to?

Usage (whisperx venv; set HF_HUB_OFFLINE=1 to skip the slow huggingface.co retries):
  .venv-whisperx312/Scripts/python.exe experiments/speaker-change/embed_clips.py <clips.json> <reference.json> <out.json>
      [--chunk 3.0] [--hop 1.5]

<clips.json> comes from koelog_clips.py; <reference.json> is labelled pieces with embeddings (embed_probe.py
output). Per clip it prints: chunk count, mean pairwise cosine between chunks (single speaker -> high), the
share of chunks nearest to each reference speaker and the mean best similarity. These are hints for the human
who assigns the clip speaker; nothing is decided automatically. Chunk embeddings are saved for enrollment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from pyannote_diarize import MODEL, SAMPLE_RATE, load_window  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("reference")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=float, default=3.0)
    ap.add_argument("--hop", type=float, default=1.5)
    args = ap.parse_args()

    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    names = sorted({r["speaker"] for r in ref})
    R = np.array([r["embedding"] for r in ref])
    cents = np.stack([R[[r["speaker"] == n for r in ref]].mean(0) for n in names])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)

    pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    pipeline.to(torch.device("cuda"))
    emb = pipeline._embedding
    n_chunk, n_hop = int(args.chunk * SAMPLE_RATE), int(args.hop * SAMPLE_RATE)
    print(f"{'ep':>3} {'dur':>6} {'chunks':>6} {'self-cos':>8}  " + "  ".join(f"{n:>6}" for n in names) + "  best-sim")
    for c in clips:
        wave, _ = load_window(c["audio"], c["start"], c["end"] - c["start"])
        starts = list(range(0, max(wave.shape[1] - n_chunk, 0) + 1, n_hop))
        batch = torch.stack([wave[:, s:s + n_chunk] for s in starts])  # (B, 1, samples)
        E = np.asarray(emb(batch))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        sims = E @ cents.T
        near = sims.argmax(1)
        pair = E @ E.T
        self_cos = float((pair.sum() - len(E)) / max(len(E) * (len(E) - 1), 1))
        shares = "  ".join(f"{(near == k).mean():6.0%}" for k in range(len(names)))
        print(f"{c['episode']:>3} {c['duration']:6.1f} {len(E):6d} {self_cos:8.3f}  {shares}  {sims.max(1).mean():.3f}")
        c["chunks"] = [{"start": round(c["start"] + s / SAMPLE_RATE, 2), "end": round(c["start"] + (s + n_chunk) / SAMPLE_RATE, 2),
                        "nearest": names[int(k)], "sim": [round(float(x), 4) for x in row],
                        "embedding": [round(float(v), 5) for v in e]}
                       for s, k, row, e in zip(starts, near, sims, E, strict=True)]
        c["self_cos"] = round(self_cos, 4)
    Path(args.out).write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
