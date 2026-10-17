"""Enroll speaker voiceprints from labelled material and test them on the gold pieces of another episode.

Usage (whisperx venv, HF_HUB_OFFLINE=1):
  .venv-whisperx312/Scripts/python.exe experiments/speaker-change/enroll_eval.py \
      experiments/speaker-change/enroll_sources.json <gold_pieces.json> <out.json>

Sources (see enroll_sources.json):
  kind=ass   : speaker = style name without "1." prefix; parts overlapped by another speaker's line are cut away,
               pieces >= 1 s kept, pieces > 6 s split into 3 s chunks
  kind=range : one known speaker; cut into 3 s chunks
Test: gold pieces (embed_probe.py output, same embedding model, held-out episode). For each combination of
source sets, per-speaker centroids (cosine) and 5-NN vote are evaluated, both restricted to the speakers present
in the gold and over all enrolled speakers. All enrollment embeddings are written to <out.json> for reuse.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from pyannote_diarize import MODEL, SAMPLE_RATE, load_window  # noqa: E402

MIN_DUR, MAX_DUR, CHUNK = 1.0, 6.0, 3.0
COMBOS = [("studio_ass",), ("studio_ass", "studio_ass_unproofed"), ("koelog",),
          ("studio_ass", "studio_ass_unproofed", "koelog")]


def ass_time(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def ass_pieces(path: str) -> list[tuple[float, float, str]]:
    lines = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if raw.startswith("Dialogue:"):
            f = raw.split(",", 9)
            spk = re.sub(r"^\d+\.", "", f[3]).strip()
            if spk and spk != "Default" and f[9].strip():
                lines.append((ass_time(f[1]), ass_time(f[2]), spk))
    out = []
    for i, (s, e, spk) in enumerate(lines):
        cuts = sorted((max(s, s2), min(e, e2)) for j, (s2, e2, spk2) in enumerate(lines)
                      if j != i and spk2 != spk and s2 < e and e2 > s)
        cur = s
        for a, b in cuts + [(e, e)]:
            out += split(cur, a, spk)
            cur = max(cur, b)
    return out


def split(a: float, b: float, spk: str) -> list[tuple[float, float, str]]:
    if b - a < MIN_DUR:
        return []
    if b - a <= MAX_DUR:
        return [(a, b, spk)]
    n = int((b - a) // CHUNK)
    return [(a + k * CHUNK, a + (k + 1) * CHUNK, spk) for k in range(n)]


def evaluate(E: np.ndarray, spk: np.ndarray, P: np.ndarray, truth: np.ndarray) -> dict:
    names = sorted(set(spk))
    C = np.stack([E[spk == n].mean(0) for n in names])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = P @ C.T
    gold_names = [n for n in names if n in set(truth)]
    res = {"enrolled": dict(Counter(spk.tolist()))}
    if len(gold_names) < len(set(truth)):
        res["note"] = "not every gold speaker enrolled"
        return res
    idx = [names.index(n) for n in gold_names]
    res["centroid_gold_only"] = float(np.mean(np.array(gold_names)[S[:, idx].argmax(1)] == truth))
    res["centroid_all"] = float(np.mean(np.array(names)[S.argmax(1)] == truth))
    K = P @ E.T
    knn_all, knn_gold = [], []
    mask = np.isin(spk, gold_names)
    for row in K:
        top = np.argsort(row)[::-1][:5]
        knn_all.append(Counter(spk[top]).most_common(1)[0][0])
        topg = np.where(mask)[0][np.argsort(row[mask])[::-1][:5]]
        knn_gold.append(Counter(spk[topg]).most_common(1)[0][0])
    res["knn5_gold_only"] = float(np.mean(np.array(knn_gold) == truth))
    res["knn5_all"] = float(np.mean(np.array(knn_all) == truth))
    res["per_speaker_centroid_gold_only"] = {
        n: float(np.mean(np.array(gold_names)[S[truth == n][:, idx].argmax(1)] == n)) for n in gold_names}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources")
    ap.add_argument("gold_pieces")
    ap.add_argument("out")
    args = ap.parse_args()

    cfg = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    pipeline.to(torch.device("cuda"))
    emb = pipeline._embedding

    rows = []
    for src in cfg["sources"]:
        if src["kind"] == "ass":
            pieces = ass_pieces(src["ass"])
            x, _ = load_window(src["audio"], 0, 0)
            base = 0.0
        else:
            pieces = [(src["start"] + k * CHUNK, src["start"] + (k + 1) * CHUNK, src["speaker"])
                      for k in range(int((src["end"] - src["start"]) // CHUNK))]
            x, _ = load_window(src["audio"], src["start"], src["end"] - src["start"])
            base = src["start"]
        kept = 0
        for a, b, spk in pieces:
            lo, hi = int((a - base) * SAMPLE_RATE), int((b - base) * SAMPLE_RATE)
            if hi - lo < int(MIN_DUR * SAMPLE_RATE * 0.9) or hi > x.shape[1]:
                continue
            e = np.asarray(emb(x[:, lo:hi].unsqueeze(0))).ravel()
            rows.append({"set": src["set"], "id": src["id"], "speaker": spk, "start": round(a, 2), "end": round(b, 2),
                         "embedding": (e / np.linalg.norm(e)).round(5).tolist()})
            kept += 1
        per = Counter(r["speaker"] for r in rows if r["id"] == src["id"])
        print(f"{src['set']:22} {src['id']:>4}: {kept} pieces {dict(per)}", flush=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    gold = json.loads(Path(args.gold_pieces).read_text(encoding="utf-8"))
    P = np.array([g["embedding"] for g in gold])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    truth = np.array([g["speaker"] for g in gold])
    print(f"\ngold: {len(gold)} pieces {dict(Counter(truth.tolist()))}")
    E = np.array([r["embedding"] for r in rows])
    for combo in COMBOS:
        m = np.array([r["set"] in combo for r in rows])
        if not m.any():
            continue
        res = evaluate(E[m], np.array([r["speaker"] for r in rows])[m], P, truth)
        print(f"\n[{' + '.join(combo)}] enrolled {res['enrolled']}")
        if "note" in res:
            print(f"  {res['note']}")
            continue
        print(f"  centroid: gold speakers only {res['centroid_gold_only']:.1%}, all enrolled {res['centroid_all']:.1%}"
              f" | 5-NN: {res['knn5_gold_only']:.1%} / {res['knn5_all']:.1%}"
              f" | per speaker {({k: f'{v:.0%}' for k, v in res['per_speaker_centroid_gold_only'].items()})}")


if __name__ == "__main__":
    main()
