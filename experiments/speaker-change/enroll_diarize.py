"""Diarize a window by enrolled voiceprints instead of clustering (sliding-window embedding + nearest centroid).

Usage (whisperx venv, HF_HUB_OFFLINE=1):
  .venv-whisperx312/Scripts/python.exe experiments/speaker-change/enroll_diarize.py <audio.wav> <out_dir>
      --enroll runs/enroll/enroll_embeddings.json --sets studio_ass,studio_ass_unproofed --members 理名,朱李
      [--model redimnet_m] [--vad <pyannote diarization.json>] [--start S --duration D] [--win 1.5] [--hop 0.25]
      [--smooth 0.5]

Speech activity: without --vad it is computed in-process from the pyannote segmentation model alone (frames with
speaker count > 0, the same speech region as the union of the full pipeline's "regular" turns, but skipping its
embedding + clustering, ~1 s instead of ~25 s per 27 min); with --vad it is the union of the given pyannote
turns. Who-is-talking is NOT taken from pyannote. Each window gets the nearest centroid among --members; every
10 ms frame takes the label of the windows covering it (sum of similarities); labels are median-smoothed over
--smooth seconds. Output diarization.json in the same format as pyannote_diarize.py ("exclusive" == "regular"),
so score_speakers.py can compare them directly. redimnet_m windows run batched on the GPU in fp16 (--fp32 off).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "third_party"))
from pyannote_diarize import MODEL, SAMPLE_RATE, load_window  # noqa: E402

STEP = 0.01
BENCH_SETS = {"studio_ass", "studio_ass_unproofed"}  # bench_spk_models.py default --sets


def load_pipeline():
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    return pipeline.to(torch.device("cuda"))


def speech_turns(pipeline, x: torch.Tensor) -> list[dict]:
    """Speech regions of the community-1 pipeline from its segmentation + speaker count only (count > 0)."""
    from pyannote.audio.utils.signal import Binarize
    from pyannote.core import SlidingWindowFeature
    seg = pipeline.get_segmentations({"waveform": x, "sample_rate": SAMPLE_RATE})
    count = pipeline.speaker_count(seg, pipeline._segmentation.model.receptive_field, warm_up=(0.0, 0.0))
    active = SlidingWindowFeature((count.data > 0).astype(np.float32), count.sliding_window)
    binarize = Binarize(onset=0.5, offset=0.5, min_duration_on=0.0,
                        min_duration_off=pipeline.segmentation.min_duration_off)
    return [{"start": s.start, "end": s.end} for s in binarize(active).get_timeline()]


def redimnet_windows(x: torch.Tensor, w: int, h: int, fp16: bool, bs: int = 32) -> np.ndarray:
    # RTX 4070S, 4000 windows of 0.75 s: fp32 8.3 s, fp16 4.8 s (bs 64) / 4.0 s (bs 32) / 3.7 s (+channels_last)
    from bench_spk_models import DEVICE, load_redimnet
    model = load_redimnet().to(memory_format=torch.channels_last)
    frames = x[0].to(DEVICE).unfold(0, w, h)  # (windows, w) view, no copy
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        for k in range(0, len(frames), bs):
            out.append(model(frames[k:k + bs].contiguous()).float())
    return torch.cat(out).cpu().numpy()


def median_labels(label: np.ndarray, k: int, m: int) -> np.ndarray:
    """Majority label over [i-k, i+k] (edge-padded), ties to the lowest index, like np.bincount().argmax()."""
    pad = np.pad(label, k, mode="edge")
    cs = np.concatenate([np.zeros((1, m), np.int32), np.cumsum(np.eye(m, dtype=np.int32)[pad], 0)])
    return (cs[2 * k + 1:] - cs[:-2 * k - 1]).argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("--enroll", required=True)
    ap.add_argument("--sets", default="studio_ass,studio_ass_unproofed")
    ap.add_argument("--exclude-ids", default="", help="comma-separated source ids left out (leave-one-episode-out)")
    ap.add_argument("--members", required=True, help="comma-separated speakers to choose from")
    ap.add_argument("--vad", help="pyannote diarization.json (speech regions only); default: segmentation in-process")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--win", type=float, default=1.5)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--model", default="pyannote34",
                    help="embedding model; others than pyannote34 need the bench_spk_models.py cache")
    ap.add_argument("--fp32", action="store_true", help="redimnet_m windows in fp32 (default fp16 autocast)")
    ap.add_argument("--save-scores", action="store_true",
                    help="also write frame_scores.npz (10 ms frames x members, mean cosine before smoothing)")
    ap.add_argument("--bench-dir", default=str(HERE / "runs" / "bench"))
    ap.add_argument("--bench-rows", default=str(HERE / "runs" / "enroll" / "enroll_embeddings.json"),
                    help="enrollment json the bench cache was built from")
    ap.add_argument("--adapt-iters", type=int, default=0, help="re-estimate centroids on this episode (0 = off)")
    ap.add_argument("--adapt-margin", type=float, default=0.05, help="min top1-top2 cosine margin to use a window")
    ap.add_argument("--adapt-prior", type=float, default=0.3, help="weight kept on the enrolled centroid")
    args = ap.parse_args()
    t0, timing = time.time(), {}

    def lap(name: str) -> None:
        timing[name] = round(time.time() - t0 - sum(timing.values()), 2)

    sets, members = set(args.sets.split(",")), args.members.split(",")
    rows = [r for r in json.loads(Path(args.enroll).read_text(encoding="utf-8")) if r["set"] in sets
            and r["id"] not in args.exclude_ids.split(",")]
    if args.model == "pyannote34":
        embs = np.array([r["embedding"] for r in rows])
    else:  # enrollment embeddings of other models come from the bench_spk_models.py cache (rows matched by span)
        z = np.load(Path(args.bench_dir) / f"{args.model}.npz")
        bench_rows = [r for r in json.loads(Path(args.bench_rows).read_text(encoding="utf-8"))
                      if r["set"] in BENCH_SETS]
        assert len(bench_rows) == len(z["E"]), "bench cache does not match --bench-rows"
        index = {(r["id"], r["start"], r["end"]): i for i, r in enumerate(bench_rows)}
        rows = [r for r in rows if (r["id"], r["start"], r["end"]) in index]
        embs = z["E"][[index[(r["id"], r["start"], r["end"])] for r in rows]]
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    spk = np.array([r["speaker"] for r in rows])
    C = np.stack([embs[spk == m].mean(0) for m in members])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    lap("enroll")

    x, _ = load_window(args.audio, args.start, args.duration)
    length = x.shape[1] / SAMPLE_RATE
    n = int(length / STEP) + 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h = int(args.win * SAMPLE_RATE), int(args.hop * SAMPLE_RATE)
    starts = np.arange(0, max(x.shape[1] - w, 0) + 1, h)
    tag = "" if args.model == "pyannote34" else f"-{args.model}"
    if args.model == "redimnet_m" and not args.fp32:
        tag += "-fp16"
    cache = out / f"windows-w{args.win}-h{args.hop}{tag}.npy"
    speech_cache = out / "speech.json"
    lap("decode")

    pipeline = None
    if args.vad:
        vad = [{"start": t["start"] - args.start, "end": t["end"] - args.start}
               for t in json.loads(Path(args.vad).read_text(encoding="utf-8"))["regular"]]
    elif speech_cache.exists() and json.loads(speech_cache.read_text(encoding="utf-8"))["offset"] == args.start:
        vad = json.loads(speech_cache.read_text(encoding="utf-8"))["speech"]
    else:
        pipeline = load_pipeline()
        lap("load pyannote")
        vad = speech_turns(pipeline, x)
        speech_cache.write_text(json.dumps({"offset": args.start, "speech": vad}), encoding="utf-8")
        lap("speech (segmentation)")
    speech = np.zeros(n, bool)
    for t in vad:
        if t["end"] > 0 and t["start"] < length:
            speech[max(int(t["start"] / STEP), 0):min(int(t["end"] / STEP), n)] = True

    if cache.exists():
        E = np.load(cache)
    else:
        if args.model == "pyannote34":
            emb = (pipeline or load_pipeline())._embedding
            E = np.concatenate([np.asarray(emb(torch.stack([x[:, s:s + w] for s in starts[k:k + 64]])))
                                for k in range(0, len(starts), 64)])
        elif args.model == "redimnet_m":
            E = redimnet_windows(x, w, h, fp16=not args.fp32)
        else:
            from bench_spk_models import load_model
            embed = load_model(args.model)
            E = np.concatenate([embed([x[:, s:s + w] for s in starts[k:k + 256]])
                                for k in range(0, len(starts), 256)])
        assert len(E) == len(starts)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        np.save(cache, E)
    lap("window embeddings")

    # per-episode adaptation: k-means started from the enrolled centroids, only on confident speech windows;
    # the enrolled centroid is kept as a prior (weight --adapt-prior) so a member cannot drift to another voice
    a = (starts / SAMPLE_RATE / STEP).astype(int)  # first frame of each window
    L = int(args.win / STEP)
    cs = np.concatenate([[0], np.cumsum(speech)])
    b = np.minimum(((starts + w) / SAMPLE_RATE / STEP).astype(int), n)
    win_speech = (cs[b] - cs[a]) / np.maximum(b - a, 1) >= 0.8
    C0 = C.copy()
    for it in range(args.adapt_iters):
        S = E @ C.T
        top = np.sort(S, 1)
        lab, margin = S.argmax(1), top[:, -1] - top[:, -2]
        sel = win_speech & (margin >= args.adapt_margin)
        for j in range(len(members)):
            pick = sel & (lab == j)
            if pick.sum() >= 20:
                mean = E[pick].mean(0)
                c = (1 - args.adapt_prior) * mean / np.linalg.norm(mean) + args.adapt_prior * C0[j]
                C[j] = c / np.linalg.norm(c)
        print(f"adapt {it + 1}: " + ", ".join(f"{m}={np.sum(sel & (lab == j)) * args.hop:.0f}s"
                                              for j, m in enumerate(members)), flush=True)
    S = E @ C.T  # (windows, members)

    # every frame sums the similarities of the windows covering it (difference array + cumsum)
    diff = np.zeros((n + 1, len(members)))
    np.add.at(diff, a, S)
    np.add.at(diff, np.minimum(a + L, n), -S)
    score = np.cumsum(diff, 0)[:n]
    label = score.argmax(1)
    if args.save_scores:  # per-frame mean cosine to each member (soft scores for word-level voting)
        cover = np.zeros(n + 1)
        np.add.at(cover, a, 1)
        np.add.at(cover, np.minimum(a + L, n), -1)
        mean = score / np.maximum(np.cumsum(cover)[:n], 1)[:, None]
        np.savez_compressed(out / "frame_scores.npz", scores=mean.astype(np.float16), speech=speech,
                            members=np.array(members), offset=args.start, step=STEP)
    k = max(int(args.smooth / STEP) // 2, 0)
    if k:
        label = median_labels(label, k, len(members))
    label[~speech] = -1

    edges = np.flatnonzero(np.diff(np.concatenate([[-1], label, [-1]])))  # run boundaries
    turns = [{"start": round(args.start + i * STEP, 3), "end": round(args.start + j * STEP, 3),
              "speaker": members[label[i]]} for i, j in zip(edges[:-1], edges[1:], strict=True) if label[i] >= 0]
    lap("assign")
    payload = {"model": f"{args.model} embedding + enrolled centroids ({args.sets})", "audio": str(Path(args.audio).resolve()),
               "offset": args.start, "duration": round(length, 3), "win": args.win, "hop": args.hop,
               "smooth": args.smooth, "adapt": [args.adapt_iters, args.adapt_margin, args.adapt_prior],
               "timing_s": timing, "regular": turns, "exclusive": turns,
               "speakers": {m: {"total_s": round(sum(t["end"] - t["start"] for t in turns if t["speaker"] == m), 2)}
                            for m in members}}
    (out / "diarization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(starts)} windows, {len(turns)} turns: "
          + ", ".join(f"{m}={v['total_s']}s" for m, v in payload["speakers"].items()))
    print("timing " + ", ".join(f"{k}={v}s" for k, v in timing.items()) + f", total={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
