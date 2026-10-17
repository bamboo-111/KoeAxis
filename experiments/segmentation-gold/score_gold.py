"""Score subtitle segmentations against a hand-made gold slice (see README.md).

Usage (.venv312):
  python experiments/segmentation-gold/score_gold.py <gold.ass> <aligned_segments.json> NAME=PATH [NAME=PATH ...]
      [--baseline NAME] [--until SECONDS] [--show NAME]

Gold model: every non-empty gold line is one *speech unit* (one speaker, Actor field). Final subtitles are
merges of adjacent units, so for each pair of consecutive ASR tokens:
  * same unit                     -> a break here is WRONG (切在一句话中间)
  * different unit, other speaker -> a break is REQUIRED (换人 / 附和必须单独成行)
  * different unit, same speaker  -> optional (only counted)
Empty gold lines are fillers / back-channels that should be deleted; ASR tokens falling in them must not be
shown. Candidates (manifest json with start_time/end_time ms + original_subtitle, or .srt) are mapped onto
the shared ASR token stream by character alignment, so text differences in the gold do not matter.
Only the labelled part of the gold is scored: lines whose Actor is not a pyannote placeholder (S00, S01, ?)
... up to --until (default: end of the last labelled line).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

PUNCT = re.compile(r"[\s\u3000、。，,．.！!？?…‥・「」『』（）()〜~—\-]+")
PLACEHOLDER = re.compile(r"^S\d\d$")
# Known labelling slips, fixed here until the gold file is corrected: (old text, new text).
GOLD_FIXES = {"わかんないも": "わかんないもう", "う歌う以前にもうね": "歌う以前にもうね"}
TOL = (0.02, 0.05, 0.1)
UNKNOWN = "\x00"  # normalized text of an inaudible gold unit ("？？？")


def norm(s: str) -> str:
    return PUNCT.sub("", s).lower()


def ass_time(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def load_gold(path: Path) -> tuple[float, list[dict]]:
    offset, lines = 0.0, []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if raw.startswith("KoeAxis Slice Offset:"):
            offset = float(raw.split(":", 1)[1])
        if not raw.startswith("Dialogue:"):
            continue
        f = raw.split(",", 9)
        text = GOLD_FIXES.get(f[9], f[9])
        n = norm(text)
        if not n and re.search(r"[？?]", text):
            n = UNKNOWN  # inaudible but real speech: a unit, not a filler
        lines.append({"s": ass_time(f[1]), "e": ass_time(f[2]), "spk": f[4], "eff": f[8], "text": text, "n": n})
    return offset, lines


def load_cues(path: Path) -> list[dict]:
    if path.suffix.lower() == ".srt":
        cues = []
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
            rows = block.strip().splitlines()
            ts = next(i for i, r in enumerate(rows) if "-->" in r)
            a, b = (x.strip().replace(",", ".") for x in rows[ts].split("-->"))
            cues.append({"s": ass_time(a), "e": ass_time(b), "text": rows[ts + 1] if len(rows) > ts + 1 else ""})
    else:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        items = data.values() if isinstance(data, dict) else data
        cues = [{"s": c["start_time"] / 1000, "e": c["end_time"] / 1000, "text": c["original_subtitle"],
                 "overlay": bool(c.get("overlay"))} for c in items]
    cues.sort(key=lambda c: (c["s"], c["e"]))
    return cues


def load_tokens(path: Path, lo: float, hi: float) -> list[dict]:
    toks = []
    for seg in json.loads(path.read_text(encoding="utf-8-sig")):
        for t in seg.get("tokens") or []:
            mid = (t["start_time"] + t["end_time"]) / 2
            if lo <= mid <= hi and norm(t["text"]):
                toks.append({"s": t["start_time"], "e": t["end_time"], "text": t["text"], "n": norm(t["text"])})
    toks.sort(key=lambda t: (t["s"], t["e"]))
    return toks


def assign_tokens(toks: list[dict], gold: list[dict]) -> None:
    """Label each token with the gold line it belongs to.

    Score = time overlap + text bonus (token text contained in the line, decaying with time gap up to 1.5 s)
    + continuity bonus (same line as the previous token). ASR times can be off by several hundred ms around
    overlaps, so text containment matters more than raw overlap.
    """
    prev = None
    for t in toks:
        best, best_score = None, 0.0
        for i, g in enumerate(gold):
            ov = min(t["e"] + 0.05, g["e"]) - max(t["s"] - 0.05, g["s"])
            gap = max(g["s"] - t["e"], t["s"] - g["e"], 0.0)
            text = 1.0 - gap / 1.5 if g["n"] and gap < 1.5 and t["n"] in g["n"] else 0.0
            if ov <= 0 and text <= 0:
                continue
            score = max(ov, 0.0) + text + (0.15 if i == prev else 0.0)
            if score > best_score:
                best, best_score = i, score
        t["g"] = prev = best  # None -> outside every gold line (treated as noise to delete)


def _map_subset(toks: list[dict], tok_ids: list[int], cues: list[dict], cue_ids: list[int],
                out: list[int | None]) -> None:
    tchars, towner = [], []
    for i in tok_ids:
        tchars += list(toks[i]["n"])
        towner += [i] * len(toks[i]["n"])
    cchars, cowner = [], []
    for j in cue_ids:
        n = norm(cues[j]["text"])
        cchars += list(n)
        cowner += [j] * len(n)
    votes: dict[int, dict[int, int]] = {}
    sm = difflib.SequenceMatcher(None, tchars, cchars, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            v = votes.setdefault(towner[a + k], {})
            v[cowner[b + k]] = v.get(cowner[b + k], 0) + 1
    for i, v in votes.items():
        if sum(v.values()) * 2 >= len(toks[i]["n"]):
            out[i] = max(v, key=v.get)


def map_cues(toks: list[dict], cues: list[dict]) -> list[int | None]:
    """For each token, the index of the candidate cue showing it (None = not shown).

    Overlay cues (manifest "overlay": true, an interjection shown on top of a cue that continues around it) are
    matched in a second pass against the tokens the normal cues left unmatched, so their out-of-order text does not
    break the monotonic character alignment.
    """
    out: list[int | None] = [None] * len(toks)
    normal = [j for j, c in enumerate(cues) if not c.get("overlay")]
    overlay = [j for j, c in enumerate(cues) if c.get("overlay")]
    _map_subset(toks, list(range(len(toks))), cues, normal, out)
    if overlay:
        _map_subset(toks, [i for i, o in enumerate(out) if o is None], cues, overlay, out)
    return out


def score(toks: list[dict], gold: list[dict], cues: list[dict], owner: list[int | None]) -> dict:
    content = [i for i, t in enumerate(toks) if t["g"] is not None and gold[t["g"]]["n"]]
    filler = [i for i, t in enumerate(toks) if i not in set(content)]
    r = {"cues": len({o for o in owner if o is not None}), "wrong_break": [], "missed_break": [],
         "req": 0, "opt": 0, "opt_break": 0, "filler_kept": [], "filler_tokens": len(filler)}
    for a, b in zip(content, content[1:], strict=False):
        ga, gb = toks[a]["g"], toks[b]["g"]
        oa, ob = owner[a], owner[b]
        if oa is None or ob is None:
            continue
        brk = oa != ob
        where = f"{toks[a]['text']}|{toks[b]['text']} @{toks[b]['s']:.2f}"
        if ga == gb:
            if brk:
                r["wrong_break"].append(where)
        elif gold[ga]["spk"] != gold[gb]["spk"] or gold[ga]["spk"] == "?":
            r["req"] += 1
            if not brk:
                r["missed_break"].append(where)
        else:
            r["opt"] += 1
            r["opt_break"] += brk
    for i in filler:
        if owner[i] is not None:
            r["filler_kept"].append(f"{toks[i]['text']}@{toks[i]['s']:.2f}")
    # per-cue correctness: content tokens = whole consecutive same-speaker units, no filler shown
    shown: dict[int, list[int]] = {}
    for i, o in enumerate(owner):
        if o is not None:
            shown.setdefault(o, []).append(i)
    unit_toks: dict[int, set[int]] = {}
    for i in content:
        unit_toks.setdefault(toks[i]["g"], set()).add(i)
    correct, timing, filler_only = {}, [], 0
    for j, idx in shown.items():
        cont = [i for i in idx if i in unit_toks.get(toks[i]["g"], ())]
        if not cont:
            filler_only += 1
            continue
        units = sorted({toks[i]["g"] for i in cont}, key=lambda g: gold[g]["s"])
        ok = (len(cont) == len(idx)
              and len({gold[g]["spk"] for g in units}) == 1 and gold[units[0]]["spk"] != "?"
              and set(cont) == set().union(*(unit_toks[g] for g in units)))
        if ok:
            correct[j] = frozenset(cont)
            gs, ge = min(gold[g]["s"] for g in units), max(gold[g]["e"] for g in units)
            timing.append((cues[j]["s"] - gs, cues[j]["e"] - ge))
    r["filler_only_cues"] = filler_only
    r["correct"] = correct
    r["timing"] = timing
    return r


def fmt_timing(timing: list[tuple[float, float]]) -> str:
    if not timing:
        return "-"
    parts = []
    for tol in TOL:
        both = sum(abs(a) <= tol + 1e-9 and abs(b) <= tol + 1e-9 for a, b in timing)
        parts.append(f"≤{int(tol * 1000)}ms {both}")
    late = sum(b > 0.3 for _, b in timing)
    return " / ".join(parts) + f" ; 结尾超出>0.3s {late}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("aligned")
    ap.add_argument("candidates", nargs="+", help="NAME=PATH (manifest json or .srt, absolute times)")
    ap.add_argument("--baseline", default=None, help="candidate NAME used for the 改坏 (regression) count")
    ap.add_argument("--until", type=float, default=None, help="score only gold time < this (slice seconds)")
    ap.add_argument("--show", action="append", default=[], help="print per-cue detail for this candidate")
    args = ap.parse_args()

    offset, lines = load_gold(Path(args.gold))
    labelled = [g for g in lines if not PLACEHOLDER.match(g["spk"])]
    named = [g for g in labelled if g["spk"] != "?"]
    until = args.until if args.until is not None else max(g["e"] for g in named) + 0.01
    gold = [g for g in labelled if g["s"] < until]
    for g in gold:  # to absolute time
        g["s"] += offset
        g["e"] += offset
    lo, hi = offset + min(g["s"] - offset for g in gold) - 0.3, offset + until + 0.1
    toks = load_tokens(Path(args.aligned), lo, hi)
    assign_tokens(toks, gold)
    units = [g for g in gold if g["n"]]
    covered = {t["g"] for t in toks if t["g"] is not None}
    print(f"gold {Path(args.gold).name}: 0–{until:.2f}s (abs {lo:.2f}–{hi:.2f}), "
          f"{len(units)} 个说话单元 + {len(gold) - len(units)} 条空行(应删), ASR token {len(toks)}")
    missing = [g["text"] for g in units if gold.index(g) not in covered]
    if missing:
        print(f"  ASR 无对应 token 的单元（漏识别，不计分）: {missing}")

    results = {}
    for spec in args.candidates:
        name, path = spec.split("=", 1)
        cues = [c for c in load_cues(Path(path)) if lo <= (c["s"] + c["e"]) / 2 <= hi]
        owner = map_cues(toks, cues)
        results[name] = (cues, owner, score(toks, gold, cues, owner))

    base = results[args.baseline][2]["correct"] if args.baseline else None
    print()
    hdr = f"{'版本':14} {'条数':>4} {'正确条':>6} {'错切':>4} {'漏切(换人)':>10} {'未删语气词':>10} {'语气词独立条':>12} {'同人可选切':>10}"
    if base is not None:
        hdr += f" {'改坏':>4}"
    print(hdr)
    broken_detail = {}
    for name, (_cues, _owner, r) in results.items():
        row = (f"{name:14} {r['cues']:>4} {len(r['correct']):>6} {len(r['wrong_break']):>4} "
               f"{len(r['missed_break']):>5}/{r['req']:<4} {len(r['filler_kept']):>5}/{r['filler_tokens']:<4} "
               f"{r['filler_only_cues']:>12} {r['opt_break']:>5}/{r['opt']:<4}")
        if base is not None:
            mine = set(r["correct"].values())
            broken = [s for s in base.values() if not any(s <= m for m in mine)]
            broken_detail[name] = ["".join(toks[i]["text"] for i in sorted(s)) for s in broken]
            row += f" {len(broken):>4}"
        print(row)
    for name, items in broken_detail.items():
        if items:
            print(f"  [{name}] 改坏（基线正确、此版不再正确）: {items}")
    print("\n正确条的起止时间（两端都在容差内的条数）:")
    for name, (_c, _o, r) in results.items():
        print(f"  {name:14} {fmt_timing(r['timing'])}")
    print("\n明细:")
    for name, (_c, _o, r) in results.items():
        print(f"  [{name}] 错切: {r['wrong_break']}")
        print(f"  [{name}] 漏切: {r['missed_break']}")
        print(f"  [{name}] 未删: {r['filler_kept']}")
    for name in args.show:
        cues, owner, r = results[name]
        print(f"\n[{name}] 逐条:")
        for j, c in enumerate(cues):
            idx = [i for i, o in enumerate(owner) if o == j]
            labs = "".join(
                "F" if toks[i]["g"] is None or not gold[toks[i]["g"]]["n"] else gold[toks[i]["g"]]["spk"][0]
                for i in idx)
            mark = "✓" if j in r["correct"] else " "
            print(f"  {mark} {c['s'] - offset:6.2f}-{c['e'] - offset:6.2f} {labs:12} {c['text']}")
    print("\ntoken→gold 映射:")
    print("  " + " ".join(
        f"{t['text']}[{'-' if t['g'] is None else (gold[t['g']]['spk'][0] + ('' if gold[t['g']]['n'] else '空'))}]"
        for t in toks))


if __name__ == "__main__":
    main()
