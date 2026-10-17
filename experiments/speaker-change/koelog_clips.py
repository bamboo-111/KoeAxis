"""Locate the 声ログ voice-memo clip of each episode (enrollment material for speaker identification).

Usage (.venv312):
  python experiments/speaker-change/koelog_clips.py <out.json> [--exclude 31]

Episode structure: station jingle "…トゲラジ" -> the member's voice memo (usually ONE person) -> title
"声ログ。/コエログ。" -> studio intro "…お届けしていく声ログ" -> studio talk ("今回は…" names who recorded it).
The clip window is [end of the last "トゲラジ" cue before the title, start of the title cue]. Speaker names are
NOT guessed here: the output lists the clip opening and the studio lines after the intro so a human can fill
in "speaker" (or "skip" for multi-person clips).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

TITLE = re.compile(r"(?<!記)(ログ|録)[。、]?$")  # short cue ending in the corner title (ASR spells it many ways)
INTRO = re.compile(r"(届けしていく|声のブログとして)")
JINGLE = re.compile(r"(トゲ|とげ)ラジ")


def load(ws: str) -> list[tuple[float, float, str]]:
    data = json.loads(Path(ws, "manifests", "split_segments.json").read_text(encoding="utf-8-sig"))
    items = data.values() if isinstance(data, dict) else data
    return sorted((c["start_time"] / 1000, c["end_time"] / 1000, c["original_subtitle"]) for c in items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--exclude", nargs="*", default=["31"], help="episodes reserved as test material")
    args = ap.parse_args()

    seen, rows = set(), []
    for ws in sorted(glob.glob("workspaces/radio-*")):
        ws = ws.replace(os.sep, "/")
        m = re.match(r"workspaces/radio-(?:cici-)?(\d+)", ws)
        if not m or "clip" in ws or not Path(ws, "manifests", "split_segments.json").exists():
            continue
        ep = m.group(1)
        if ep in seen or ep in args.exclude:
            continue
        seen.add(ep)
        cues = load(ws)
        intro = next((i for i, c in enumerate(cues) if INTRO.search(c[2]) and c[0] < 900), None)
        if intro is None:
            print(f"{ep}: no 声ログ intro, skipped")
            continue
        title = next((i for i in range(intro, max(intro - 8, 0), -1) if TITLE.search(cues[i][2].strip()) and len(cues[i][2].strip()) <= 12
                      and cues[intro][0] - cues[i][0] < 30 and not INTRO.search(cues[i][2])), None)
        if title is None:
            print(f"{ep}: no title cue before intro, skipped")
            continue
        jingle = next((i for i in range(title - 1, max(title - 60, 0), -1)
                       if JINGLE.search(cues[i][2]) and cues[title][0] - cues[i][1] < 200), None)
        start = cues[jingle][1] if jingle is not None else cues[title][0] - 60
        end = cues[title][0]
        audio = str(Path(ws, "audio", "source.wav"))
        clip = [c[2] for c in cues if start <= c[0] < end]
        joined = " / ".join(c[2] for c in cues[intro:intro + 10])
        marks = list(re.finditer(r"(届けしていく|声のブログとして(お)?)(声ログ|コエログ|声録)?。?", joined))
        rest = joined[marks[-1].end():] if marks else joined
        after = [x for x in (y.strip() for y in rest.split(" / ")) if x and not TITLE.fullmatch(x)][:6]
        rows.append({"episode": ep, "workspace": ws, "audio": audio, "start": round(start, 2), "end": round(end, 2),
                     "duration": round(end - start, 1), "jingle_found": jingle is not None,
                     "clip_head": " / ".join(clip[:4]), "studio_after": " / ".join(after), "speaker": ""})
        print(f"{ep}: {start:7.1f}-{end:7.1f}s ({end - start:5.1f}s) jingle={'y' if jingle is not None else 'n'}")
        print(f"    clip : {' / '.join(clip[:4])[:110]}")
        print(f"    after: {' / '.join(after)[:110]}")
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
