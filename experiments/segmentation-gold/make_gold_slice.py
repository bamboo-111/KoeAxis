"""Cut an audio slice + starting-point ASS for hand-made gold subtitles (accurate segmentation, timing, speaker).

Usage (.venv312):
  python experiments/segmentation-gold/make_gold_slice.py <workspace> <diarization.json> <out_dir>
      [--manifest split_segments.json] [--length 300] [--start SECONDS] [--skip-edges 90]

Starting text/timing comes from <workspace>/manifests/<manifest> (default split_segments.json: rule split +
LLM semantic merge, times taken from aligned tokens, before normalize extends/snaps anything).
Actor is pre-filled from pyannote exclusive diarization (majority overlap); Effect marks cues where
pyannote hears >= 2 speakers ("多人").

Without --start, the window with the most (short cues + speaker changes) is chosen, avoiding the first/last
--skip-edges seconds (OP/ED). Window edges are snapped into the silence between cues.

ASS times are relative to the slice; the absolute offset is stored in [Script Info] as
"KoeAxis Slice Offset" (seconds), together with the source wav and manifest, so the gold can be mapped back.
The slice is cut from audio/source.wav (same timeline as the pipeline; no mp3 decoder delay).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PUNCT = re.compile(r"[\s、。，,．.！!？?…‥・「」『』（）()〜~ー—\-]+")

HEADER = """[Script Info]
; Gold subtitles: exact segmentation, timing (+-20 ms, no lead-out extension) and speaker (Actor field).
; Times are relative to the slice. Absolute time = slice time + KoeAxis Slice Offset.
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None
PlayResX: 1920
PlayResY: 1080
KoeAxis Slice Offset: {offset:.3f}
KoeAxis Slice Length: {length:.3f}
KoeAxis Source Audio: {source}
KoeAxis Source Manifest: {manifest}

[Aegisub Project Garbage]
Audio File: {audio}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, \
Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, \
MarginV, Encoding
Style: Default,Microsoft YaHei,60,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,40,1
Style: 附和,Microsoft YaHei,50,&H0000FFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,8,20,20,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def load_cues(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    items = data.values() if isinstance(data, dict) else data
    cues = [{"start": c["start_time"] / 1000, "end": c["end_time"] / 1000, "text": c["original_subtitle"]}
            for c in items]
    return sorted(cues, key=lambda c: (c["start"], c["end"]))


def speaker_share(start: float, end: float, turns: list[dict]) -> dict[str, float]:
    acc: dict[str, float] = {}
    for t in turns:
        ov = min(end, t["end"]) - max(start, t["start"])
        if ov > 0:
            acc[t["speaker"]] = acc.get(t["speaker"], 0.0) + ov
    return acc


def change_times(turns: list[dict]) -> list[float]:
    return [b["start"] for a, b in zip(turns, turns[1:], strict=False) if a["speaker"] != b["speaker"]]


def choose_window(cues, turns, length, skip, total) -> float:
    changes = change_times(turns)
    best, best_score = skip, -1.0
    t = skip
    while t + length <= total - skip:
        short = sum(1 for c in cues if t <= c["start"] and c["end"] <= t + length
                    and len(PUNCT.sub("", c["text"])) <= 4)
        chg = sum(1 for x in changes if t <= x <= t + length)
        score = short + chg
        if score > best_score:
            best, best_score = t, score
        t += 5.0
    return best


def snap(cues, t: float, direction: int) -> float:
    """Move t into the silence between cues (never inside a cue)."""
    for c in cues:
        if c["start"] < t < c["end"]:
            return c["start"] - 0.3 if direction < 0 else c["end"] + 0.3
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("diarization")
    ap.add_argument("out_dir")
    ap.add_argument("--manifest", default="split_segments.json")
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--skip-edges", type=float, default=90.0)
    args = ap.parse_args()

    ws = Path(args.workspace)
    source = (ws / "audio" / "source.wav").resolve()
    manifest = ws / "manifests" / args.manifest
    cues = load_cues(manifest)
    diar = json.loads(Path(args.diarization).read_text(encoding="utf-8"))
    turns = sorted(diar["exclusive"], key=lambda t: t["start"])
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(source)], check=True, capture_output=True, text=True).stdout.strip())

    start = args.start if args.start is not None else choose_window(cues, turns, args.length, args.skip_edges, total)
    start = max(0.0, snap(cues, start, -1))
    end = min(total, snap(cues, start + args.length, +1))
    length = end - start

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{ws.name}_{start:.0f}-{end:.0f}s"
    wav = out / f"{stem}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source),
                    "-c:a", "pcm_s16le", str(wav)], check=True)

    lines = [HEADER.format(title=stem, offset=start, length=length, source=source, manifest=manifest.resolve(),
                           audio=wav.resolve())]
    n = multi = 0
    for c in cues:
        if c["start"] < start or c["end"] > end:
            continue
        share = speaker_share(c["start"], c["end"], turns)
        actor = ("S" + max(share, key=share.get).rsplit("_", 1)[-1]) if share else "?"
        is_multi = sum(1 for v in share.values() if v >= 0.3) >= 2
        multi += is_multi
        n += 1
        text = c["text"].replace("\n", " ")
        lines.append(f"Dialogue: 0,{ass_time(c['start'] - start)},{ass_time(c['end'] - start)},Default,{actor},"
                     f"0,0,0,{'多人' if is_multi else ''},{text}")
    ass = out / f"{stem}.ass"
    ass.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"window {start:.2f}-{end:.2f}s ({length:.1f}s), {n} cues ({multi} multi-speaker) -> {ass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
