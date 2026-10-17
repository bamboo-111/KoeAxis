"""Flag subtitle cues that span a MOSS speaker change (chunk-local speaker labels).

Usage: python flag_cues.py <chunks.json> <subtitle.srt> [--min-overlap 0.3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen_asr.clip_planner import parse_srt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("srt")
    ap.add_argument("--min-overlap", type=float, default=0.3)
    args = ap.parse_args()
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    segs = [dict(s, chunk=c["chunk"]) for c in chunks for s in c["segments"]]
    lo = min(c["offset"] for c in chunks)
    hi = max(c["offset"] + c["length"] for c in chunks)
    cues = [c for c in parse_srt(Path(args.srt).read_text(encoding="utf-8-sig")) if c.start >= lo and c.end <= hi]
    flagged = 0
    for cue in cues:
        hits = [s for s in segs if min(cue.end, s["end"]) - max(cue.start, s["start"]) >= args.min_overlap]
        speakers = {(s["chunk"], s["speaker"]) for s in hits}
        if len(speakers) < 2:
            continue
        flagged += 1
        print(f"#{cue.index} {cue.start:.2f}-{cue.end:.2f} {cue.text.splitlines()[0]}")
        for s in hits:
            print(f"     {s['speaker']} {s['start']:.2f}-{s['end']:.2f} {s['text']}")
    print(f"cues={len(cues)} flagged={flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
