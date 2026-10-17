"""Build a one-file Aegisub review package for all 声ログ clips, so the recorder can be identified quickly.

Usage (whisperx venv, HF_HUB_OFFLINE=1):
  .venv-whisperx312/Scripts/python.exe experiments/speaker-change/koelog_review.py <clips.json> <out_dir>
      [--ref NAME=wav ... --check gold_pieces.json]

Output:
  review.wav  all clips concatenated, 2 s silence before each clip
  review.ass  per clip one header line (style 段头, Actor "?") showing the episode, members on air (from the
              episode's self-introduction), chunk self-consistency (single voice ≈0.45–0.6, groups ≈0.27–0.30)
              and what the studio said right after the clip; then one line per ASR cue (Actor "?",
              Effect "ep@episode_time"). Fill the header Actor (member, or 多人); per-line Actors are optional
              (useful to recover single-speaker stretches inside group clips).
  review_map.json  clip offsets in review.wav, to map labels back to episode time.

Voices are NOT guessed: references from other recordings (the hk快闪 videos, --ref) scored 31.8% on the 30期
gold pieces (3-way), i.e. cross-recording-condition centroids are useless. --ref/--check only re-run that test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from pyannote_diarize import MODEL, SAMPLE_RATE, load_window  # noqa: E402

# Members on air, read from each episode's self-introduction at ~17 s ("トゲナシトゲアリのリナとゆりです").
MEMBERS = {
    "24": "理名 夕莉", "25": "理名 夕莉 朱李", "26": "理名 夕莉 朱李", "27": "朱李 夕莉", "45": "理名 夕莉 朱李",
    "46": "理名 夕莉", "47": "理名 夕莉", "48": "理名 夕莉 朱李", "49": "朱李 夕莉", "50": "朱李 夕莉",
    "51": "理名 夕莉 朱李", "52": "理名 朱李", "53": "理名 朱李", "54": "理名 夕莉", "30": "理名 朱李",
}
GAP = 2.0
SINGLE, MULTI = 0.40, 0.31  # chunk self-consistency thresholds (from the clips: one voice ≈0.45–0.6, groups ≈0.27–0.30)

HEADER = """[Script Info]
; 声ログ review: fix the Actor of each 段头 line (member name, or 多人 / X). Per-line Actors are optional.
Title: koelog review
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[Aegisub Project Garbage]
Audio File: review.wav

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,56,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,50,1
Style: 段头,Microsoft YaHei,44,&H0000FFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,3,1,8,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ts(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def chunk_embed(emb, x: torch.Tensor, chunk: float = 3.0, hop: float = 1.5) -> np.ndarray:
    n, h = int(chunk * SAMPLE_RATE), int(hop * SAMPLE_RATE)
    starts = range(0, max(x.shape[1] - n, 0) + 1, h)
    E = np.asarray(emb(torch.stack([x[:, s:s + n] for s in starts])))
    return E / np.linalg.norm(E, axis=1, keepdims=True)



def load_cues(ws: str, a: float, b: float) -> list[tuple[float, float, str]]:
    data = json.loads(Path(ws, "manifests", "split_segments.json").read_text(encoding="utf-8-sig"))
    items = data.values() if isinstance(data, dict) else data
    return sorted((c["start_time"] / 1000, c["end_time"] / 1000, c["original_subtitle"]) for c in items
                  if a <= c["start_time"] / 1000 < b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("out_dir")
    ap.add_argument("--ref", action="append", default=[], help="NAME=single-person wav (only used with --check)")
    ap.add_argument("--check", default=None, help="labelled pieces json (embed_probe.py) to test the references")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    pipeline.to(torch.device("cuda"))
    emb = pipeline._embedding

    names, cents = [], []
    for spec in args.ref:
        name, path = spec.split("=", 1)
        x, _ = load_window(path, 0, 0)
        E = chunk_embed(emb, x)
        c = E.mean(0)
        sim = E @ (c / np.linalg.norm(c))
        keep = sim >= np.median(sim)
        c = E[keep].mean(0)
        names.append(name)
        cents.append(c / np.linalg.norm(c))
        print(f"ref {name}: {x.shape[1] / SAMPLE_RATE:.0f}s, {len(E)} chunks, self-cos {float(sim.mean()):.3f}")
    if cents:
        C = np.stack(cents)
        print("ref centroid cosine:", {f"{names[i]}-{names[j]}": round(float(C[i] @ C[j]), 3)
                                       for i in range(len(names)) for j in range(i + 1, len(names))})

    if cents and args.check:
        pieces = json.loads(Path(args.check).read_text(encoding="utf-8"))
        P = np.array([p["embedding"] for p in pieces])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        S = P @ C.T
        acc = np.mean([names[int(k)] == p["speaker"] for k, p in zip(S.argmax(1), pieces, strict=True)])
        both = [k for k, n in enumerate(names) if n in {p["speaker"] for p in pieces}]
        acc2 = np.mean([names[both[int(np.argmax(row[both]))]] == p["speaker"]
                        for row, p in zip(S, pieces, strict=True)])
        print(f"check on {len(pieces)} gold pieces: 3-way {acc:.1%}, restricted to gold speakers {acc2:.1%}")

    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    audio, lines, mapping, t = [], [], [], 0.0
    for c in clips:
        x, _ = load_window(c["audio"], c["start"], c["end"] - c["start"])
        offset = t + GAP
        audio += [np.zeros(int(GAP * SAMPLE_RATE), np.float32), x.numpy().ravel()]
        E = chunk_embed(emb, x)
        pair = E @ E.T
        self_cos = float((pair.sum() - len(E)) / max(len(E) * (len(E) - 1), 1))
        hint = "像单人" if self_cos >= SINGLE else ("像多人" if self_cos < MULTI else "不确定")
        head = (f"【第{c['episode']}期｜本期成员 {MEMBERS.get(c['episode'], '?')}｜一致性 {self_cos:.2f} {hint}】"
                f"录音后：{c['studio_after'][:70]}")
        lines.append(f"Dialogue: 0,{ts(offset)},{ts(offset + min(4.0, c['end'] - c['start']))},段头,?,0,0,0,"
                     f"ep{c['episode']},{head}")
        for a, b, text in load_cues(c["workspace"], c["start"], c["end"]):
            lines.append(f"Dialogue: 0,{ts(offset + a - c['start'])},{ts(offset + min(b, c['end']) - c['start'])},"
                         f"Default,?,0,0,0,ep{c['episode']}@{a:.2f},{text}")
        mapping.append({"episode": c["episode"], "workspace": c["workspace"], "review_offset": round(offset, 3),
                        "episode_start": c["start"], "episode_end": c["end"], "self_cos": round(self_cos, 4)})
        print(f"ep{c['episode']:>3} {c['end'] - c['start']:6.1f}s  self-cos {self_cos:.2f} {hint}")
        t = offset + (c["end"] - c["start"])

    pcm = np.clip(np.concatenate(audio), -1, 1)
    with wave.open(str(out / "review.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())
    (out / "review.ass").write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8-sig")
    (out / "review_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out / 'review.ass'} ({len(clips)} clips, {t / 60:.1f} min)")


if __name__ == "__main__":
    main()
