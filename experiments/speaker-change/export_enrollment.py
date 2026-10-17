"""Export enrolled ReDimNet-M member centroids for the production speaker track (qwen_asr/speaker/voiceprint.py).

Usage (.venv312):
  python experiments/speaker-change/export_enrollment.py samples/speaker-enroll-togeradi.json
      [--members 理名,夕莉,朱李] [--sets studio_ass,studio_ass_unproofed] [--exclude-ids "#30"]
      [--ass-styles "D:/字幕组work/radio/过往结果/【已校】#15 zepp巡演，朱李的手指，想拍的电影.ass"]

Centroid = mean of the L2-normalised ReDimNet-M embeddings of a member's enrollment segments (runs/bench/redimnet_m.npz,
rows in the order of runs/enroll/enroll_embeddings.json filtered to the bench sets), L2-normalised again — the same
computation as `enroll_diarize.py --model redimnet_m`.

--ass-styles copies the per-member styles of a proofread studio ASS (one style per member, named like "1.理名") and
its PlayResX/PlayResY, so the speaker ASS export (qwen_asr/speaker/ass_export.py) looks like the studio subtitles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH_SETS = {"studio_ass", "studio_ass_unproofed"}


def ass_styles(path: Path, members: list[str]) -> dict:
    """{"script_info": {...PlayRes...}, "styles": {member: "Style: ..."}} from a studio ASS."""
    info, styles = {}, {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key = line.split(":", 1)[0]
        if key in ("PlayResX", "PlayResY", "WrapStyle", "ScaledBorderAndShadow"):
            info[key] = line.split(":", 1)[1].strip()
        if line.startswith("Style:"):
            name = line[len("Style:"):].split(",", 1)[0].strip()
            member = name.split(".", 1)[-1]
            if member in members:
                styles[member] = line.strip()
    missing = [m for m in members if m not in styles]
    if missing:
        raise SystemExit(f"{path.name} has no style for {', '.join(missing)}")
    return {"source": path.name, "script_info": info, "styles": styles}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--members", default="理名,夕莉,朱李")
    ap.add_argument("--sets", default="studio_ass,studio_ass_unproofed")
    ap.add_argument("--exclude-ids", default="")
    ap.add_argument("--rows", default=str(HERE / "runs" / "enroll" / "enroll_embeddings.json"))
    ap.add_argument("--bench", default=str(HERE / "runs" / "bench" / "redimnet_m.npz"))
    ap.add_argument("--ass-styles", help="studio ASS to copy member styles (Style: N.<member>,...) from")
    args = ap.parse_args()

    rows = [r for r in json.loads(Path(args.rows).read_text(encoding="utf-8")) if r["set"] in BENCH_SETS]
    emb = np.load(args.bench)["E"]
    assert len(rows) == len(emb), "bench cache does not match the enrollment rows"
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sets, excluded = set(args.sets.split(",")), {x for x in args.exclude_ids.split(",") if x}
    keep = np.array([r["set"] in sets and r["id"] not in excluded for r in rows])
    spk = np.array([r["speaker"] for r in rows])
    members, counts = {}, {}
    for name in args.members.split(","):
        pick = keep & (spk == name)
        if not pick.any():
            raise SystemExit(f"no enrollment segments for {name}")
        c = emb[pick].mean(0)
        members[name] = [round(float(v), 6) for v in c / np.linalg.norm(c)]
        counts[name] = {"segments": int(pick.sum()),
                        "seconds": round(sum(r["end"] - r["start"] for r, k in zip(rows, pick, strict=True) if k), 1)}
    payload = {
        "model": "redimnet_m",
        "weights": "M-vb2+vox2+cnc-ft_mix.pt",
        "win": 0.75,
        "hop": 0.2,
        "source": f"experiments/speaker-change export_enrollment.py, sets {args.sets}"
                  + (f", excluding {args.exclude_ids}" if excluded else ""),
        "counts": counts,
        "members": members,
    }
    if args.ass_styles:
        payload["ass"] = ass_styles(Path(args.ass_styles), list(members))
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(", ".join(f"{k}: {v['segments']} segs / {v['seconds']}s" for k, v in counts.items()))


if __name__ == "__main__":
    main()
