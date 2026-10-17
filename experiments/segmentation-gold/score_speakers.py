"""Score diarization outputs (pyannote diarization.json) against the speaker labels of a gold slice.

Usage (.venv312):
  python experiments/segmentation-gold/score_speakers.py <gold.ass> NAME=diarization.json [NAME=...]
      [--mode exclusive|regular] [--tol 0.3]

Each diarization cluster is mapped to the gold speaker it overlaps most (several clusters may map to the
same person). Metrics, over the labelled gold region (named speakers only, '?' and placeholders skipped):
  * frame acc : 10 ms frames where exactly one gold speaker talks (fillers included) and the mapped
                cluster agrees; "no cluster" counts as wrong. covered = share of those frames with any
                cluster (speech detected), acc|cov = accuracy on covered frames only
  * line acc  : non-empty gold lines whose majority cluster maps to the right person
  * change rec: gold speaker changes (consecutive non-empty lines, different speaker, by start time) with a
                diarization change of the right direction within +-tol seconds
"""
from __future__ import annotations

import argparse
import json
import re

from pathlib import Path

import numpy as np

PLACEHOLDER = re.compile(r"^S\d\d$")
STEP = 0.01


def ass_time(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def load_gold(path: Path) -> list[dict]:
    offset, lines = 0.0, []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if raw.startswith("KoeAxis Slice Offset:"):
            offset = float(raw.split(":", 1)[1])
        if raw.startswith("Dialogue:"):
            f = raw.split(",", 9)
            if PLACEHOLDER.match(f[4]) or f[4] == "?":
                continue
            lines.append({"s": ass_time(f[1]), "e": ass_time(f[2]), "spk": f[4], "text": f[9].strip()})
    for g in lines:
        g["s"] += offset
        g["e"] += offset
    return sorted(lines, key=lambda g: g["s"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("runs", nargs="+", help="NAME=diarization.json")
    ap.add_argument("--mode", default="exclusive", choices=["exclusive", "regular"])
    ap.add_argument("--tol", type=float, default=0.3)
    args = ap.parse_args()

    gold = load_gold(Path(args.gold))
    lo, hi = min(g["s"] for g in gold), max(g["e"] for g in gold)
    people = sorted({g["spk"] for g in gold})
    n = int(round((hi - lo) / STEP)) + 1
    active = {p: np.zeros(n, bool) for p in people}
    for g in gold:
        active[g["spk"]][int((g["s"] - lo) / STEP):int((g["e"] - lo) / STEP)] = True
    count = sum(active.values())
    single = count == 1
    truth = np.full(n, -1)
    for k, p in enumerate(people):
        truth[single & active[p]] = k
    talk = [g for g in gold if g["text"]]
    changes = [(a, b) for a, b in zip(talk, talk[1:], strict=False) if a["spk"] != b["spk"]]
    print(f"gold: {hi - lo:.1f}s, speakers {people}, single-speaker {single.sum() * STEP:.1f}s, "
          f"overlap {(count > 1).sum() * STEP:.1f}s, {len(talk)} lines, {len(changes)} changes")
    print(f"{'run':22} {'clusters→person':40} {'frame acc':>9} {'covered':>8} {'acc|cov':>8} {'line acc':>9} {'change rec':>10}")

    for spec in args.runs:
        name, path = spec.split("=", 1)
        turns = [t for t in json.loads(Path(path).read_text(encoding="utf-8"))[args.mode]
                 if t["end"] > lo and t["start"] < hi]
        labels = sorted({t["speaker"] for t in turns})
        hyp = {c: np.zeros(n, bool) for c in labels}
        for t in turns:
            hyp[t["speaker"]][max(0, int((t["start"] - lo) / STEP)):max(0, int((t["end"] - lo) / STEP))] = True
        mapping = {c: people[int(np.argmax([(hyp[c] & active[p]).sum() for p in people]))] for c in labels}
        pred = np.full(n, -1)
        for c in labels:  # exclusive mode: at most one cluster per frame
            pred[hyp[c]] = people.index(mapping[c])
        frame_acc = (pred[single] == truth[single]).mean()
        cov = pred[single] >= 0
        covered = cov.mean()
        acc_cov = (pred[single][cov] == truth[single][cov]).mean()

        def majority(s: float, e: float, _pred: np.ndarray = pred) -> str | None:
            seg = _pred[int((s - lo) / STEP):int((e - lo) / STEP)]
            seg = seg[seg >= 0]
            return people[int(np.bincount(seg).argmax())] if seg.size else None

        line_acc = sum(majority(g["s"], g["e"]) == g["spk"] for g in talk) / len(talk)
        seq = sorted(((t["start"], mapping[t["speaker"]]) for t in turns))
        hyp_changes = [(t, b) for (_, a), (t, b) in zip(seq, seq[1:], strict=False) if a != b]
        hit = sum(any(abs(t - b["s"]) <= args.tol and who == b["spk"] for t, who in hyp_changes)
                  for _a, b in changes)
        cmap = ", ".join(f"{c[-2:]}→{mapping[c]}" for c in labels)
        print(f"{name:22} {cmap:40} {frame_acc:9.1%} {covered:8.1%} {acc_cov:8.1%} {line_acc:9.1%} {hit:>5}/{len(changes):<4}")


if __name__ == "__main__":
    main()
