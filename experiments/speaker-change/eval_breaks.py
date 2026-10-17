"""Map pyannote speakers onto aligned tokens, derive speaker breaks, flag subtitle cues that span them.

Usage (any venv with the repo importable, e.g. .venv312):
  python experiments/speaker-change/eval_breaks.py <diarization.json> <aligned_segments.json> <subtitle.srt> <out_dir>
      [--min-side 0.5] [--edge 0.15] [--window START END] [--moss chunks.json] [--truth T1,T2,...]

Outputs in <out_dir>:
  speaker_breaks.json   all breaks: {time, prev_speaker, next_speaker, left_s, right_s, strong}
  token_speakers.json   per-token speaker labels
  report.md             flagged cues (pyannote vs MOSS), truth-point hits, summary numbers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen_asr.clip_planner import parse_srt  # noqa: E402


def load_tokens(path: Path) -> list[dict]:
    segments = json.loads(path.read_text(encoding="utf-8-sig"))
    tokens = []
    for seg in segments:
        for tok in seg.get("tokens") or []:
            if tok["end_time"] > tok["start_time"]:
                tokens.append({"text": tok["text"], "start": tok["start_time"], "end": tok["end_time"]})
    tokens.sort(key=lambda t: (t["start"], t["end"]))
    # Drop exact duplicates produced by overlapping segment padding.
    deduped = []
    for tok in tokens:
        if deduped and abs(deduped[-1]["start"] - tok["start"]) < 1e-3 and deduped[-1]["text"] == tok["text"]:
            continue
        deduped.append(tok)
    return deduped


def assign_speakers(tokens: list[dict], turns: list[dict]) -> None:
    j = 0
    for tok in tokens:
        while j < len(turns) and turns[j]["end"] <= tok["start"] - 2.0:
            j += 1
        overlap: dict[str, float] = {}
        k = j
        while k < len(turns) and turns[k]["start"] < tok["end"] + 2.0:
            ov = min(tok["end"], turns[k]["end"]) - max(tok["start"], turns[k]["start"])
            if ov > 0:
                overlap[turns[k]["speaker"]] = overlap.get(turns[k]["speaker"], 0.0) + ov
            k += 1
        if overlap:
            tok["speaker"] = max(overlap, key=overlap.get)
            continue
        # No overlap (token in a diarization gap): nearest turn within 0.5 s.
        mid = (tok["start"] + tok["end"]) / 2
        best, dist = None, 0.5
        for t in turns[max(0, j - 1):k + 1]:
            d = max(t["start"] - mid, mid - t["end"], 0.0)
            if d <= dist:
                best, dist = t["speaker"], d
        tok["speaker"] = best


def build_runs(tokens: list[dict]) -> list[dict]:
    runs: list[dict] = []
    for tok in tokens:
        if tok.get("speaker") is None:
            continue
        if runs and runs[-1]["speaker"] == tok["speaker"]:
            run = runs[-1]
            run["end"] = tok["end"]
            run["speech"] += tok["end"] - tok["start"]
            run["text"] += tok["text"]
        else:
            runs.append({"speaker": tok["speaker"], "start": tok["start"], "end": tok["end"],
                         "speech": tok["end"] - tok["start"], "text": tok["text"]})
    return runs


def build_breaks(runs: list[dict], min_side: float) -> list[dict]:
    breaks = []
    for a, b in zip(runs, runs[1:], strict=False):
        left, right = a["speech"], b["speech"]
        breaks.append({
            "time": round((a["end"] + b["start"]) / 2, 3),
            "gap": round(b["start"] - a["end"], 3),
            "prev_speaker": a["speaker"], "next_speaker": b["speaker"],
            "left_s": round(left, 2), "right_s": round(right, 2),
            "strong": left >= min_side and right >= min_side,
            "left_text": a["text"][-12:], "right_text": b["text"][:12],
        })
    return breaks


def moss_flags(chunks_path: Path, cues, min_overlap: float = 0.3) -> set[int]:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    segs = [dict(s, chunk=c["chunk"]) for c in chunks for s in c["segments"]]
    flagged = set()
    for cue in cues:
        hits = [s for s in segs if min(cue.end, s["end"]) - max(cue.start, s["start"]) >= min_overlap]
        if len({(s["chunk"], s["speaker"]) for s in hits}) >= 2:
            flagged.add(cue.index)
    return flagged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("diarization")
    ap.add_argument("aligned")
    ap.add_argument("srt")
    ap.add_argument("out_dir")
    ap.add_argument("--min-side", type=float, default=0.5)
    ap.add_argument("--edge", type=float, default=0.15, help="ignore breaks this close to a cue edge")
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--moss", default=None)
    ap.add_argument("--truth", default="", help="comma-separated manually confirmed change times (s)")
    ap.add_argument("--truth-tol", type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    diar = json.loads(Path(args.diarization).read_text(encoding="utf-8"))
    turns = sorted(diar["exclusive"], key=lambda t: t["start"])
    tokens = load_tokens(Path(args.aligned))
    lo, hi = args.window if args.window else (float("-inf"), float("inf"))
    tokens = [t for t in tokens if t["start"] >= lo and t["end"] <= hi]
    assign_speakers(tokens, turns)
    runs = build_runs(tokens)
    breaks = build_breaks(runs, args.min_side)
    (out / "token_speakers.json").write_text(json.dumps(tokens, ensure_ascii=False, indent=0), encoding="utf-8")
    (out / "speaker_breaks.json").write_text(json.dumps(breaks, ensure_ascii=False, indent=1), encoding="utf-8")

    cues = [c for c in parse_srt(Path(args.srt).read_text(encoding="utf-8-sig")) if c.start >= lo and c.end <= hi]
    strong = [b for b in breaks if b["strong"]]
    weak = [b for b in breaks if not b["strong"]]

    def inside(cue, items):
        return [b for b in items if cue.start + args.edge < b["time"] < cue.end - args.edge]

    py_strong = {c.index for c in cues if inside(c, strong)}
    py_weak_only = {c.index for c in cues if c.index not in py_strong and inside(c, weak)}
    moss = moss_flags(Path(args.moss), cues) if args.moss else None

    lines = ["# pyannote 换人断点评估", "",
             f"- diarization: `{args.diarization}`（{len(diar['speakers'])} 个说话人，耗时 {diar['elapsed_s']}s）",
             f"- 字幕: `{args.srt}`，范围内 {len(cues)} 条；token {len(tokens)} 个，未分配 "
             f"{sum(1 for t in tokens if t.get('speaker') is None)} 个",
             f"- 断点：强 {len(strong)}（两侧各 ≥{args.min_side}s 语音）/ 弱 {len(weak)}",
             f"- 含强断点的字幕 {len(py_strong)} 条；只含弱断点 {len(py_weak_only)} 条"]
    if moss is not None:
        lines.append(f"- MOSS 标出 {len(moss)} 条；与 pyannote 强断点重合 {len(moss & py_strong)}，"
                     f"仅 MOSS {len(moss - py_strong)}，仅 pyannote {len(py_strong - moss)}")
    if args.truth:
        lines += ["", "## 人工确认切点"]
        for t in (float(x) for x in args.truth.split(",")):
            near = [b for b in breaks if abs(b["time"] - t) <= args.truth_tol]
            desc = "; ".join(f"{b['time']:.2f}s {'强' if b['strong'] else '弱'} "
                             f"{b['prev_speaker']}→{b['next_speaker']} 「{b['left_text']}｜{b['right_text']}」"
                             for b in near) or "未检出"
            lines.append(f"- {t:.2f}s：{desc}")
    lines += ["", "## 逐条字幕", "", "| # | 时间 | 字幕 | pyannote | MOSS |", "|---|---|---|---|---|"]
    for cue in cues:
        s, w = inside(cue, strong), inside(cue, weak)
        m = moss is not None and cue.index in moss
        if not (s or w or m):
            continue
        tag = "；".join(f"{'强' if b['strong'] else '弱'}@{b['time']:.2f} 「{b['left_text']}｜{b['right_text']}」"
                       for b in s + w) or "—"
        text = cue.text.splitlines()[0].replace("|", "/")
        lines.append(f"| {cue.index} | {cue.start:.2f}-{cue.end:.2f} | {text} | {tag} | {'✓' if m else ''} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
