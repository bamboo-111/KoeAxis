"""Can pyannote's speaker embedding separate the hosts at all? Probe on gold single-speaker pieces.

Usage (whisperx venv):
  .venv-whisperx312/Scripts/python.exe experiments/speaker-change/embed_probe.py <gold.ass> <source.wav> <out.json>
      [--min-dur 1.0]

Takes every non-empty gold line, removes the parts overlapped by any other gold line (other speakers or
fillers), keeps pieces >= --min-dur seconds, embeds them with the community-1 embedding model, and reports
leave-one-out nearest-centroid accuracy (cosine) per duration bucket. Embeddings are saved to <out.json> so
enrollment experiments can reuse them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from pyannote_diarize import MODEL, SAMPLE_RATE, load_window  # noqa: E402

PLACEHOLDER = re.compile(r"^S\d\d$")


def ass_time(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def gold_pieces(path: Path, min_dur: float) -> list[dict]:
    offset, lines = 0.0, []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if raw.startswith("KoeAxis Slice Offset:"):
            offset = float(raw.split(":", 1)[1])
        if raw.startswith("Dialogue:"):
            f = raw.split(",", 9)
            if not PLACEHOLDER.match(f[4]):
                lines.append((ass_time(f[1]) + offset, ass_time(f[2]) + offset, f[4], f[9].strip()))
    pieces = []
    for i, (s, e, spk, text) in enumerate(lines):
        if not text or spk not in ("朱李", "理名"):
            continue
        cuts = sorted((max(s, s2), min(e, e2)) for j, (s2, e2, _, _) in enumerate(lines)
                      if j != i and s2 < e and e2 > s)
        cur = s
        for a, b in cuts + [(e, e)]:
            if a - cur >= min_dur:
                pieces.append({"start": cur, "end": a, "speaker": spk})
            cur = max(cur, b)
    return pieces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--min-dur", type=float, default=1.0)
    args = ap.parse_args()

    pieces = gold_pieces(Path(args.gold), args.min_dur)
    lo, hi = min(p["start"] for p in pieces), max(p["end"] for p in pieces)
    wave, _ = load_window(args.audio, lo, hi - lo)
    pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    pipeline.to(torch.device("cuda"))
    emb_model = pipeline._embedding
    embs = []
    for p in pieces:
        a, b = int((p["start"] - lo) * SAMPLE_RATE), int((p["end"] - lo) * SAMPLE_RATE)
        x = wave[:, a:b].unsqueeze(0)  # (1, 1, samples)
        embs.append(np.asarray(emb_model(x)).ravel())
    E = np.stack(embs)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    spk = np.array([p["speaker"] for p in pieces])
    names = sorted(set(spk))
    correct = []
    for i in range(len(pieces)):
        mask = np.arange(len(pieces)) != i
        cents = [E[mask & (spk == n)].mean(0) for n in names]
        sims = [float(E[i] @ c / np.linalg.norm(c)) for c in cents]
        correct.append(names[int(np.argmax(sims))] == spk[i])
        pieces[i]["margin"] = round(sims[names.index(spk[i])] - max(s for k, s in enumerate(sims)
                                                                   if names[k] != spk[i]), 4)
    correct = np.array(correct)
    dur = np.array([p["end"] - p["start"] for p in pieces])
    print(f"{len(pieces)} pieces >= {args.min_dur}s: " + ", ".join(f"{n} {int((spk == n).sum())}" for n in names))
    print(f"LOO nearest-centroid acc: {correct.mean():.1%}")
    for a, b in ((args.min_dur, 1.5), (1.5, 2.5), (2.5, 99)):
        m = (dur >= a) & (dur < b)
        if m.any():
            print(f"  {a:.1f}-{b:.1f}s: {correct[m].mean():.1%} ({int(m.sum())})")
    c = [E[spk == n].mean(0) for n in names]
    print(f"centroid cosine between hosts: {float(c[0] @ c[1] / np.linalg.norm(c[0]) / np.linalg.norm(c[1])):.3f}")
    within = [float(np.mean(E[spk == n] @ (c[k] / np.linalg.norm(c[k])))) for k, n in enumerate(names)]
    print("mean cosine to own centroid: " + ", ".join(f"{n} {w:.3f}" for n, w in zip(names, within, strict=True)))
    for p, e in zip(pieces, E, strict=True):
        p["embedding"] = [round(float(v), 5) for v in e]
    Path(args.out).write_text(json.dumps(pieces, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
