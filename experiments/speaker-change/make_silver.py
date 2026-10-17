"""Turn a proofread fansub ASS (style name = speaker, e.g. "1.理名") into the gold format of score_speakers.py.

Usage: python make_silver.py <fansub.ass> <out.ass>
The speaker goes to the Name field; lines whose style is not a host (e.g. Default) are dropped. Timing is the
fansub's (aligned to speech but looser than the hand-made gold), hence "silver".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HOSTS = {"理名", "夕莉", "朱李"}


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    out, kept = ["[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"], 0
    for raw in src.read_text(encoding="utf-8-sig").splitlines():
        if raw.startswith("Dialogue:"):
            f = raw.split(",", 9)
            spk = re.sub(r"^\d+\.", "", f[3]).strip()
            if spk in HOSTS and f[9].strip():
                f[4] = spk
                out.append(",".join(f))
                kept += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8-sig")
    print(f"{dst}: {kept} lines")


if __name__ == "__main__":
    main()
