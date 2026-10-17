"""Build an unbiased labeling sheet: union of cues flagged by pyannote (exclusive/regular) and MOSS.

Usage (.venv312):
  python experiments/speaker-change/make_label_sheet.py <diarization.json> <subtitle.srt> <out_dir>
      [--window 0 360] [--moss chunks.json] [--min-overlap 0.3]
Writes <out_dir>/label_sheet.tsv (fill column `truth`: 1 = cue contains a real speaker change, 0 = no,
? = unsure) and <out_dir>/label_sheet.audacity.txt (import in Audacity: File > Import > Labels).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen_asr.clip_planner import parse_srt  # noqa: E402


def multi_speaker(cue, rows, key, min_overlap):
    acc: dict = {}
    for r in rows:
        ov = min(cue.end, r["end"]) - max(cue.start, r["start"])
        if ov > 0:
            acc[key(r)] = acc.get(key(r), 0.0) + ov
    return sum(1 for v in acc.values() if v >= min_overlap) >= 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("diarization")
    ap.add_argument("srt")
    ap.add_argument("out_dir")
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--moss", default=None)
    ap.add_argument("--min-overlap", type=float, default=0.3)
    args = ap.parse_args()

    diar = json.loads(Path(args.diarization).read_text(encoding="utf-8"))
    lo, hi = args.window if args.window else (float("-inf"), float("inf"))
    cues = [c for c in parse_srt(Path(args.srt).read_text(encoding="utf-8-sig")) if c.start >= lo and c.end <= hi]
    moss = []
    if args.moss:
        chunks = json.loads(Path(args.moss).read_text(encoding="utf-8"))
        moss = [dict(s, chunk=c["chunk"]) for c in chunks for s in c["segments"]]

    rows = []
    for cue in cues:
        flags = {
            "py_excl": multi_speaker(cue, diar["exclusive"], lambda r: r["speaker"], args.min_overlap),
            "py_reg": multi_speaker(cue, diar["regular"], lambda r: r["speaker"], args.min_overlap),
            "moss": bool(moss) and multi_speaker(cue, moss, lambda r: (r["chunk"], r["speaker"]), args.min_overlap),
        }
        if any(flags.values()):
            rows.append((cue, flags))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tsv = ["cue\tstart\tend\tpy_excl\tpy_reg\tmoss\ttruth\ttext"]
    aud = []
    for cue, f in rows:
        text = cue.text.splitlines()[0].replace("\t", " ")
        tsv.append(f"{cue.index}\t{cue.start:.2f}\t{cue.end:.2f}\t{int(f['py_excl'])}\t{int(f['py_reg'])}"
                   f"\t{int(f['moss'])}\t\t{text}")
        aud.append(f"{cue.start:.3f}\t{cue.end:.3f}\t#{cue.index} {text}")
    (out / "label_sheet.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8-sig")
    (out / "label_sheet.audacity.txt").write_text("\n".join(aud) + "\n", encoding="utf-8")
    print(f"cues in window {len(cues)}, candidates {len(rows)} "
          f"(py_excl {sum(f['py_excl'] for _, f in rows)}, py_reg {sum(f['py_reg'] for _, f in rows)}, "
          f"moss {sum(f['moss'] for _, f in rows)}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
