"""Label speaker-change cues in Aegisub, then score the detectors.

export (.venv312):
  python experiments/speaker-change/aegisub_labels.py export <diarization.json> <subtitle.srt> <audio.wav> <out.ass>
      [--window 0 360] [--moss chunks.json] [--min-overlap 0.3]
  Every cue in the window becomes a Dialogue line (audio linked, opens with the ASS in Aegisub).
  Effect column = detector hints: E = pyannote exclusive, R = pyannote regular, M = MOSS ("-" = not flagged).
  Candidates (any detector flagged) use style "候选"; the rest use "其他".

  How to label: put the verdict in the Actor (说话人) field:
    1 = this cue contains a real speaker change, 0 = no change, ? = unsure.
  Candidates must be labelled; for other cues only fill 1 if you notice a missed change (empty = 0).

score:
  python experiments/speaker-change/aegisub_labels.py score <labelled.ass> [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_label_sheet import multi_speaker  # noqa: E402

from qwen_asr.clip_planner import parse_srt  # noqa: E402

HEADER = """[Script Info]
; Speaker-change labelling sheet. Actor field: 1 = real change, 0 = none, ? = unsure.
Title: speaker-change labels
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None
PlayResX: 1920
PlayResY: 1080

[Aegisub Project Garbage]
Audio File: {audio}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, \
Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, \
MarginV, Encoding
Style: 候选,Microsoft YaHei,60,&H0000FFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,40,1
Style: 其他,Microsoft YaHei,60,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
METHODS = {"E": "pyannote exclusive", "R": "pyannote regular", "M": "MOSS"}


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def parse_ass_time(s: str) -> float:
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def export(args) -> int:
    diar = json.loads(Path(args.diarization).read_text(encoding="utf-8"))
    lo, hi = args.window if args.window else (float("-inf"), float("inf"))
    cues = [c for c in parse_srt(Path(args.srt).read_text(encoding="utf-8-sig")) if c.start >= lo and c.end <= hi]
    moss = []
    if args.moss:
        chunks = json.loads(Path(args.moss).read_text(encoding="utf-8"))
        moss = [dict(s, chunk=c["chunk"]) for c in chunks for s in c["segments"]]
    lines = [HEADER.format(audio=str(Path(args.audio).resolve()))]
    n_cand = 0
    for cue in cues:
        flags = {
            "E": multi_speaker(cue, diar["exclusive"], lambda r: r["speaker"], args.min_overlap),
            "R": multi_speaker(cue, diar["regular"], lambda r: r["speaker"], args.min_overlap),
            "M": bool(moss) and multi_speaker(cue, moss, lambda r: (r["chunk"], r["speaker"]), args.min_overlap),
        }
        cand = any(flags.values())
        n_cand += cand
        effect = "".join(k if v else "-" for k, v in flags.items()) + f" #{cue.index}"
        text = " ".join(cue.text.splitlines()[0].split()).replace(",", "，") if cue.text else ""
        lines.append(f"Dialogue: 0,{ass_time(cue.start)},{ass_time(cue.end)},{'候选' if cand else '其他'},,"
                     f"0,0,0,{effect},{text}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"{len(cues)} cues ({n_cand} candidates) -> {args.out}")
    return 0


def read_labels(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line[len("Dialogue:"):].strip().split(",", 9)
        if len(parts) < 10:
            continue
        effect = parts[8].strip()
        flags, _, idx = effect.partition(" #")
        rows.append({"cue": idx, "start": parse_ass_time(parts[1]), "end": parse_ass_time(parts[2]),
                     "style": parts[3], "label": parts[4].strip(), "flags": flags, "text": parts[9]})
    return rows


def score(args) -> int:
    rows = read_labels(Path(args.ass))
    unlabelled = [r for r in rows if r["style"] == "候选" and r["label"] not in {"0", "1", "?"}]
    truth = {}
    for r in rows:
        if r["label"] in {"0", "1"}:
            truth[r["cue"]] = r["label"] == "1"
        elif r["label"] == "" and r["style"] != "候选":
            truth[r["cue"]] = False
    by_cue = {r["cue"]: r for r in rows}
    positives = sum(truth.values())
    out = ["# 换人检测评分", "",
           f"- 字幕 {len(rows)} 条，参与评分 {len(truth)} 条，真换人 {positives} 条；"
           f"不确定 {sum(1 for r in rows if r['label'] == '?')} 条，未标候选 {len(unlabelled)} 条", "",
           "| 方法 | 标出 | 命中 | 误报 | 漏检 | 精确率 | 召回率 |", "|---|---|---|---|---|---|---|"]
    for pos, (key, name) in enumerate(METHODS.items()):
        pred = {c for c in truth if by_cue[c]["flags"][pos:pos + 1] == key}
        tp = sum(1 for c in pred if truth[c])
        fp, fn = len(pred) - tp, positives - tp
        prec = f"{tp / len(pred):.2f}" if pred else "-"
        rec = f"{tp / positives:.2f}" if positives else "-"
        out.append(f"| {name} | {len(pred)} | {tp} | {fp} | {fn} | {prec} | {rec} |")
    missed_all = [c for c, v in truth.items() if v and set(by_cue[c]["flags"]) <= {"-"}]
    if missed_all:
        out += ["", f"三种方法都漏掉：{', '.join('#' + c for c in missed_all)}"]
    if unlabelled:
        out += ["", "未标注的候选：" + ", ".join("#" + r["cue"] for r in unlabelled)]
    report = "\n".join(out) + "\n"
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("diarization")
    ex.add_argument("srt")
    ex.add_argument("audio")
    ex.add_argument("out")
    ex.add_argument("--window", type=float, nargs=2, default=None)
    ex.add_argument("--moss", default=None)
    ex.add_argument("--min-overlap", type=float, default=0.3)
    sc = sub.add_parser("score")
    sc.add_argument("ass")
    sc.add_argument("--out", default=None)
    args = ap.parse_args()
    return export(args) if args.cmd == "export" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
