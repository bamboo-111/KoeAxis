"""Segmentation-independent speaker-turn labelling in Aegisub (does NOT use split/normalize cues).

export (.venv312):
  python experiments/speaker-change/aegisub_turns.py export <diarization.json> <aligned_segments.json> <audio.wav>
      <out.ass> [--window 0 360] [--moss chunks.json] [--max-gap 0.6] [--max-line 8]
  One line = one stretch of a single speaker, built from aligned tokens (pre-split) and pyannote exclusive
  speakers; a new line also starts at pauses >= max-gap or after max-line seconds (same speaker on both sides
  of such a cut is NOT a change). Actor = pyannote speaker (S00, S01...).
  Effect = original pyannote label + hints: R = pyannote regular sees another speaker inside the line,
  M = MOSS sees >= 2 speakers inside the line. Hinted lines use the yellow style "候选" (check these first).

  How to correct in Aegisub:
    - wrong speaker: edit the Actor field (keep one consistent name per real person, e.g. the pyannote label
      or the host's name; anyone else = X; unsure = ?)
    - missed change inside a line: split it (edit box right-click > Split at cursor), fix times on the waveform,
      set the Actor of each half
    - false change: just give both lines the same Actor (no need to join)
    - lines that are not speech (music/laughter only): change Style to "其他" and set Actor to "-" (ignored)

score:
  python experiments/speaker-change/aegisub_turns.py score <labelled.ass> <diarization.json> <aligned_segments.json>
      [--moss chunks.json] [--window 0 360] [--tol 0.5] [--min-side 0.5] [--out report.md]
  Truth change points = boundaries between consecutive lines whose Actor differs (lines with Actor ? or - are
  skipped together with their boundaries).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aegisub_labels import HEADER, ass_time, parse_ass_time  # noqa: E402
from eval_breaks import assign_speakers, build_breaks, build_runs, load_tokens  # noqa: E402


def short(label: str | None) -> str:
    return "S?" if label is None else "S" + label.rsplit("_", 1)[-1]


def windowed_tokens(args, diar) -> list[dict]:
    tokens = load_tokens(Path(args.aligned))
    lo, hi = args.window if args.window else (float("-inf"), float("inf"))
    tokens = [t for t in tokens if t["start"] >= lo and t["end"] <= hi]
    assign_speakers(tokens, sorted(diar["exclusive"], key=lambda t: t["start"]))
    return tokens


def load_moss(path: str | None) -> list[dict]:
    if not path:
        return []
    chunks = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict(s, chunk=c["chunk"]) for c in chunks for s in c["segments"]]


def speakers_inside(start, end, rows, key, min_overlap=0.25) -> set:
    acc: dict = {}
    for r in rows:
        ov = min(end, r["end"]) - max(start, r["start"])
        if ov > 0:
            acc[key(r)] = acc.get(key(r), 0.0) + ov
    return {k for k, v in acc.items() if v >= min_overlap}


def export(args) -> int:
    diar = json.loads(Path(args.diarization).read_text(encoding="utf-8"))
    tokens = windowed_tokens(args, diar)
    lines: list[dict] = []
    for tok in tokens:
        spk = tok.get("speaker") or (lines[-1]["speaker"] if lines else None)
        cur = lines[-1] if lines else None
        if (cur is None or spk != cur["speaker"] or tok["start"] - cur["end"] >= args.max_gap
                or tok["end"] - cur["start"] > args.max_line):
            lines.append({"speaker": spk, "start": tok["start"], "end": tok["end"], "text": tok["text"]})
        else:
            cur["end"] = tok["end"]
            cur["text"] += tok["text"]
    moss = load_moss(args.moss)
    out = [HEADER.format(audio=str(Path(args.audio).resolve()))]
    hinted = 0
    for line in lines:
        hints = ""
        others = speakers_inside(line["start"], line["end"], diar["regular"], lambda r: r["speaker"]) - {line["speaker"]}
        if others:
            hints += "R"
        if moss and len(speakers_inside(line["start"], line["end"], moss, lambda r: (r["chunk"], r["speaker"]))) >= 2:
            hints += "M"
        hinted += bool(hints)
        label = short(line["speaker"])
        text = line["text"].replace(",", "，")
        out.append(f"Dialogue: 0,{ass_time(line['start'])},{ass_time(line['end'])},{'候选' if hints else '其他'},"
                   f"{label},0,0,0,原{label} {hints or '-'},{text}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8-sig")
    print(f"{len(tokens)} tokens -> {len(lines)} turn lines ({hinted} hinted) -> {args.out}")
    return 0


def read_lines(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.startswith("Dialogue:"):
            continue
        p = raw[len("Dialogue:"):].strip().split(",", 9)
        if len(p) == 10:
            rows.append({"start": parse_ass_time(p[1]), "end": parse_ass_time(p[2]), "actor": p[4].strip()})
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows


def truth_breaks(rows: list[dict]) -> list[dict]:
    breaks = []
    for a, b in zip(rows, rows[1:], strict=False):
        if a["actor"] in {"", "?", "-"} or b["actor"] in {"", "?", "-"} or a["actor"] == b["actor"]:
            continue
        breaks.append({"time": (a["end"] + b["start"]) / 2, "left_s": a["end"] - a["start"],
                       "right_s": b["end"] - b["start"]})
    return breaks


def moss_breaks(moss: list[dict]) -> list[float]:
    times = []
    for a, b in zip(moss, moss[1:], strict=False):
        if a["chunk"] == b["chunk"] and a["speaker"] != b["speaker"]:
            times.append((a["end"] + b["start"]) / 2)
    return times


def regular_breaks(regular: list[dict]) -> list[float]:
    times, last = [], None
    for turn in sorted(regular, key=lambda t: t["start"]):
        if last is not None and turn["speaker"] != last["speaker"]:
            times.append(turn["start"] if turn["start"] > last["start"] else last["end"])
        if last is None or turn["end"] >= last["end"]:
            last = turn
    return times


def match(truth: list[float], pred: list[float], tol: float) -> tuple[int, list[float]]:
    used, tp, errors = set(), 0, []
    for t in sorted(truth):
        best = None
        for i, p in enumerate(pred):
            if i not in used and abs(p - t) <= tol and (best is None or abs(p - t) < abs(pred[best] - t)):
                best = i
        if best is not None:
            used.add(best)
            tp += 1
            errors.append(abs(pred[best] - t))
    return tp, errors


def score(args) -> int:
    rows = read_lines(Path(args.ass))
    lo, hi = args.window if args.window else (min(r["start"] for r in rows) - 1, max(r["end"] for r in rows) + 1)
    diar = json.loads(Path(args.diarization).read_text(encoding="utf-8"))
    tokens = windowed_tokens(args, diar)
    runs = build_runs(tokens)
    excl = build_breaks(runs, args.min_side)

    def inwin(xs):
        return [x for x in xs if lo <= x <= hi]

    methods = {
        "pyannote exclusive（强断点）": inwin([b["time"] for b in excl if b["strong"]]),
        "pyannote exclusive（全部）": inwin([b["time"] for b in excl]),
        "pyannote regular": inwin(regular_breaks(diar["regular"])),
    }
    if args.moss:
        methods["MOSS"] = inwin(moss_breaks(load_moss(args.moss)))
    tb = truth_breaks(rows)
    sets = {"全部换人": [b["time"] for b in tb],
            f"两侧各 ≥{args.min_side}s": [b["time"] for b in tb if min(b["left_s"], b["right_s"]) >= args.min_side]}
    out = ["# 换人点评分（说话人轮次标注）", "",
           f"- 标注行 {len(rows)}，说话人：{', '.join(sorted({r['actor'] or '(空)' for r in rows}))}",
           f"- 真实换人点 {len(tb)}（两侧各 ≥{args.min_side}s：{len(sets[f'两侧各 ≥{args.min_side}s'])}），"
           f"容差 ±{args.tol}s"]
    for set_name, truth in sets.items():
        out += ["", f"## {set_name}（{len(truth)} 个）", "",
                "| 方法 | 预测 | 命中 | 精确率 | 召回率 | 平均切点误差 |", "|---|---|---|---|---|---|"]
        for name, pred in methods.items():
            tp, err = match(truth, pred, args.tol)
            prec = f"{tp / len(pred):.2f}" if pred else "-"
            rec = f"{tp / len(truth):.2f}" if truth else "-"
            mae = f"{sum(err) / len(err):.2f}s" if err else "-"
            out.append(f"| {name} | {len(pred)} | {tp} | {prec} | {rec} | {mae} |")
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
    ex.add_argument("aligned")
    ex.add_argument("audio")
    ex.add_argument("out")
    ex.add_argument("--window", type=float, nargs=2, default=None)
    ex.add_argument("--moss", default=None)
    ex.add_argument("--max-gap", type=float, default=0.6)
    ex.add_argument("--max-line", type=float, default=8.0)
    sc = sub.add_parser("score")
    sc.add_argument("ass")
    sc.add_argument("diarization")
    sc.add_argument("aligned")
    sc.add_argument("--moss", default=None)
    sc.add_argument("--window", type=float, nargs=2, default=None)
    sc.add_argument("--tol", type=float, default=0.5)
    sc.add_argument("--min-side", type=float, default=0.5)
    sc.add_argument("--out", default=None)
    args = ap.parse_args()
    return export(args) if args.cmd == "export" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
