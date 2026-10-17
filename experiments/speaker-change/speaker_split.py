"""Split subtitle cues where the speaker changes, by word-level voting on enrolled-voiceprint frame scores.

Usage (.venv312):
  python experiments/speaker-change/speaker_split.py <aligned_segments.json> <cues.json> <frame_scores.npz> <out.json>
      [--min-margin 0.1] [--min-dur 0.3] [--strong-margin 0.2] [--pad 0.05] [--report out.tsv]

Inputs: ASR tokens with times (aligned_segments.json), candidate cues (manifest json: start_time/end_time in ms +
original_subtitle, e.g. split_segments.json after the LLM merge) and frame_scores.npz from
`enroll_diarize.py --save-scores` (10 ms frames x members, mean cosine to each enrolled member).

Per cue: every token gets the member with the highest mean score over [start - pad, end + pad] and a margin (top1 -
top2). Consecutive tokens of the same member form runs. A run is kept only if it is confident: mean margin >=
--min-margin and (duration >= --min-dur or mean margin >= --strong-margin); otherwise it is merged into the
neighbouring run whose member scores higher on it (repeat until stable). This keeps overlapped / boundary words,
where both voices are present and the margin is small, from producing a split. Cuts are made at the remaining run
boundaries only: the text is not changed (cut before the first character of the next run, punctuation stays left),
inner boundaries take the actual token times, outer ones keep the cue times. Each output cue gets "speaker".
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segmentation-gold"))
from score_gold import load_cues, load_tokens, norm  # noqa: E402


def char_positions(text: str) -> list[int]:
    """Indices in `text` of the characters kept by norm() (one per normalized char)."""
    return [i for i, ch in enumerate(text) if norm(ch)]


def map_token_chars(toks: list[dict], cues: list[dict]) -> list[tuple[int, int] | None]:
    """For each token: (cue index, first matched char position in the cue's original text) or None."""
    tchars, towner = [], []
    for i, t in enumerate(toks):
        tchars += list(t["n"])
        towner += [i] * len(t["n"])
    cchars, cref = [], []
    for j, c in enumerate(cues):
        pos = char_positions(c["text"])
        n = norm(c["text"])
        if len(n) != len(pos):  # lower() changed the length; fall back to per-char normalisation
            n = "".join(norm(c["text"][p]) for p in pos)
        cchars += list(n)
        cref += [(j, p) for p in pos]
    votes: list[dict[int, list[int]]] = [{} for _ in toks]
    sm = difflib.SequenceMatcher(None, tchars, cchars, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            j, p = cref[b + k]
            votes[towner[a + k]].setdefault(j, []).append(p)
    out = []
    for i, v in enumerate(votes):
        if not v:
            out.append(None)
            continue
        j = max(v, key=lambda c: len(v[c]))
        out.append((j, min(v[j])) if len(v[j]) * 2 >= len(toks[i]["n"]) else None)
    return out


def runs_of(labels: list[int]) -> list[tuple[int, int]]:
    out, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((s, i))
            s = i
    return out


def vote(idx: list[int], toks: list[dict], tsc: np.ndarray, args) -> list[int]:
    """Member label per token of one cue after merging unconfident runs into their neighbours."""
    labels = [int(tsc[i].argmax()) for i in idx]
    while True:
        rs = runs_of(labels)
        if len(rs) == 1:
            return labels
        weak = []
        for a, b in rs:
            sc = tsc[idx[a:b]]
            m = labels[a]
            others = np.delete(sc, m, axis=1).max(1)
            margin = float((sc[:, m] - others).mean())
            dur = toks[idx[b - 1]]["e"] - toks[idx[a]]["s"]
            if margin < args.min_margin or (dur < args.min_dur and margin < args.strong_margin):
                weak.append((margin, a, b))
        if not weak:
            return labels
        _, a, b = min(weak)
        k = rs.index((a, b))
        nb = [rs[x] for x in (k - 1, k + 1) if 0 <= x < len(rs)]
        sc = tsc[idx[a:b]].mean(0)
        target = max((labels[r[0]] for r in nb), key=lambda m: sc[m])
        labels[a:b] = [target] * (b - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("aligned")
    ap.add_argument("cues")
    ap.add_argument("scores")
    ap.add_argument("out")
    ap.add_argument("--min-margin", type=float, default=0.1)
    ap.add_argument("--min-dur", type=float, default=0.3)
    ap.add_argument("--strong-margin", type=float, default=0.2)
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--overlay-max", type=float, default=0.0,
                    help="A|B|A with B shorter than this (s): B becomes an overlay line instead of two cuts")
    ap.add_argument("--report", help="tsv listing every cut")
    args = ap.parse_args()

    z = np.load(args.scores)
    F, members, off, step = z["scores"].astype(np.float32), [str(m) for m in z["members"]], float(z["offset"]), float(z["step"])
    cues = load_cues(Path(args.cues))
    toks = load_tokens(Path(args.aligned), -1e9, 1e9)
    tsc = np.zeros((len(toks), len(members)), np.float32)
    for i, t in enumerate(toks):
        a = max(int((t["s"] - args.pad - off) / step), 0)
        b = min(max(int((t["e"] + args.pad - off) / step) + 1, a + 1), len(F))
        tsc[i] = F[a:b].mean(0) if b > a else 0.0
    where = map_token_chars(toks, cues)
    by_cue: dict[int, list[int]] = {}
    for i, w in enumerate(where):
        if w is not None:
            by_cue.setdefault(w[0], []).append(i)

    out, report, n_cut, n_ovl = [], [], 0, 0
    for j, c in enumerate(cues):
        idx = by_cue.get(j, [])
        if not idx:
            out.append({"s": c["s"], "e": c["e"], "text": c["text"], "speaker": ""})
            continue
        labels = vote(idx, toks, tsc, args)
        # a run whose first char is not after the previous cut (char order disagrees with time order) is merged left
        changed = True
        while changed:
            changed, start_char = False, 0
            for a, b in runs_of(labels)[1:]:
                cut = where[idx[a]][1]
                if cut <= start_char:
                    labels[a:b] = [labels[a - 1]] * (b - a)
                    changed = True
                    break
                start_char = cut
        # pieces: one per run, [char_start, char_end) of the cue text, token-time span
        rs = runs_of(labels)
        pieces, start_char = [], 0
        for k, (a, b) in enumerate(rs):
            end_char = where[idx[rs[k + 1][0]]][1] if k + 1 < len(rs) else len(c["text"])
            pieces.append({"spk": labels[a], "text": c["text"][start_char:end_char].strip(),
                           "ts": toks[idx[a]]["s"], "te": toks[idx[b - 1]]["e"], "ovl": []})
            start_char = end_char
        # A | B | A with a short B -> B becomes an overlay line, A stays one cue
        k = 1
        while args.overlay_max > 0 and k + 1 < len(pieces):
            p, q, r = pieces[k - 1], pieces[k], pieces[k + 1]
            if p["spk"] == r["spk"] and q["te"] - q["ts"] <= args.overlay_max:
                merged = {"spk": p["spk"], "text": p["text"] + r["text"], "ts": p["ts"], "te": r["te"],
                          "ovl": p["ovl"] + [q] + r["ovl"]}
                pieces[k - 1:k + 2] = [merged]
                k = max(k - 1, 1)
            else:
                k += 1
        for k, p in enumerate(pieces):
            s_t = c["s"] if k == 0 else p["ts"]
            e_t = c["e"] if k == len(pieces) - 1 else p["te"]
            out.append({"s": s_t, "e": e_t, "text": p["text"], "speaker": members[p["spk"]]})
            for q in p["ovl"]:
                out.append({"s": q["ts"], "e": q["te"], "text": q["text"], "speaker": members[q["spk"]],
                            "overlay": True})
                report.append(f"{q['ts']:.2f}	叠加 {members[q['spk']]}	{q['text']}	|	{p['text']}")
                n_ovl += 1
            if k + 1 < len(pieces):
                nxt = pieces[k + 1]
                report.append(f"{nxt['ts']:.2f}	{members[p['spk']]}→{members[nxt['spk']]}	{p['text']}	|	{nxt['text']}")
                n_cut += 1
    out = [o for o in out if o["text"]]
    out.sort(key=lambda o: (o["s"], o.get("overlay", False)))

    data = {str(k + 1): {"start_time": round(p["s"] * 1000), "end_time": round(p["e"] * 1000),
                         "original_subtitle": p["text"], "translated_subtitle": "", "speaker": p["speaker"],
                         **({"overlay": True} if p.get("overlay") else {})}
            for k, p in enumerate(out)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.report:
        Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"{len(cues)} cues -> {len(out)} ({n_cut} speaker cuts, {n_ovl} overlays)")


if __name__ == "__main__":
    main()
