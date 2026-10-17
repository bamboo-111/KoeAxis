"""Closed-set frame-level speaker classification for the fixed hosts (理名 / 夕莉 / 朱李).

Frozen WavLM base-plus features (the backbone inside .model-cache/speaker/wavlm-sv, 50 frames/s, several layers
mixed by learned weights) -> small BiLSTM -> per-frame speaker posteriors. Trained on the labelled studio episodes
of enroll_sources.json (ASS lines: frames covered by exactly one speaker's line get that label; silence, overlap and
uncovered frames are ignored) plus the confirmed 声ログ ranges. The gold episode (30) is never used for training.

Usage (whisperx venv, HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1):
  python frame_spk.py train --sources enroll_sources.json --out runs/framecls [--sets ...] [--steps 3000]
  python frame_spk.py infer <audio.wav> <out_dir> --model runs/framecls/model.pt --members 理名,朱李
      --vad <pyannote diarization.json> [--smooth 0.3]
infer writes diarization.json in the pyannote_diarize.py format (regular == exclusive), plus posteriors.npy.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pyannote_diarize import load_window  # noqa: E402

WAVLM = Path(".model-cache/speaker/wavlm-sv")
SPEAKERS = ["理名", "夕莉", "朱李"]
LAYERS = [2, 4, 6, 8]
FPS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wavlm():
    from transformers import WavLMForXVector
    return WavLMForXVector.from_pretrained(str(WAVLM)).wavlm.eval().to(DEVICE)


@torch.no_grad()
def features(wavlm, wave: torch.Tensor, chunk_s: float = 20.0, ctx_s: float = 1.0) -> np.ndarray:
    """(1, T) 16 kHz -> (frames, len(LAYERS), 768) float16, computed in chunks with context on both sides."""
    x = wave[0]
    x = (x - x.mean()) / (x.std() + 1e-7)
    n_frames = x.shape[0] // 320
    out = np.zeros((n_frames, len(LAYERS), 768), np.float16)
    step, ctx = int(chunk_s * FPS), int(ctx_s * FPS)
    for f0 in range(0, n_frames, step):
        a, b = max(f0 - ctx, 0), min(f0 + step + ctx, n_frames)
        seg = x[a * 320:b * 320 + 80]
        seg = torch.nn.functional.pad(seg, (0, (b - a) * 320 + 80 - seg.shape[0])).to(DEVICE).unsqueeze(0)
        hs = wavlm(seg, output_hidden_states=True).hidden_states
        h = torch.stack([hs[k][0] for k in LAYERS], 1).cpu().numpy().astype(np.float16)
        f1 = min(f0 + step, n_frames)
        out[f0:f1] = h[f0 - a:f0 - a + (f1 - f0)]
    return out


def ass_lines(path: str) -> list[tuple[float, float, str]]:
    rows = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if raw.startswith("Dialogue:"):
            f = raw.split(",", 9)
            spk = re.sub(r"^\d+\.", "", f[3]).strip()
            if f[9].strip():
                h = [sum(float(v) * m for v, m in zip(t.split(":"), (3600, 60, 1), strict=True)) for t in f[1:3]]
                rows.append((h[0], h[1], spk))
    return rows


def frame_labels(src: dict, n_frames: int) -> np.ndarray:
    """ASS source -> per-frame label: -1 = ignore; otherwise index into SPEAKERS."""
    lab = np.full(n_frames, -1, np.int64)
    cover = np.zeros((n_frames, len(SPEAKERS) + 1), bool)  # last column = any other / unknown style
    for s, e, spk in ass_lines(src["ass"]):
        cover[int(s * FPS):int(e * FPS), SPEAKERS.index(spk) if spk in SPEAKERS else -1] = True
    single = cover.sum(1) == 1
    idx = cover.argmax(1)
    ok = single & (idx < len(SPEAKERS))
    lab[ok] = idx[ok]
    return lab


class FrameSpk(nn.Module):
    def __init__(self, n_layers: int = len(LAYERS), dim: int = 768, hidden: int = 128, n_out: int = len(SPEAKERS)):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2))
        self.rnn = nn.LSTM(256, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, L, D) -> (B, T, n_out)
        x = (x * self.mix.softmax(0)[None, None, :, None]).sum(2)
        return self.head(self.rnn(self.proj(x))[0])


def cmd_train(args) -> None:
    out = Path(args.out)
    feats = Path(args.feats) if args.feats else out / "feats"
    feats.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    sets = set(args.sets.split(","))
    srcs = [s for s in json.loads(Path(args.sources).read_text(encoding="utf-8"))["sources"] if s["set"] in sets
            and s["id"] not in args.exclude_ids.split(",")]
    wavlm, data = None, []
    for s in srcs:
        f = feats / f"{re.sub(r'[^0-9A-Za-z]', '_', s['id'])}.npy"
        rng_src = s["kind"] == "range"  # only the confirmed range is decoded and labelled
        if not f.exists():
            wavlm = wavlm or load_wavlm()
            wave = load_window(s["audio"], s["start"], s["end"] - s["start"]) if rng_src else load_window(s["audio"], 0, 0)
            np.save(f, features(wavlm, wave[0]))
        X = np.load(f, mmap_mode="r")
        y = np.full(len(X), SPEAKERS.index(s["speaker"])) if rng_src else frame_labels(s, len(X))
        data.append((np.asarray(X), y))
        print(f"{s['id']:>4}: {len(y) / FPS:7.1f}s, labelled " +
              " ".join(f"{n}={np.sum(y == i) / FPS:.0f}s" for i, n in enumerate(SPEAKERS)), flush=True)
    del wavlm
    torch.cuda.empty_cache()

    counts = np.bincount(np.concatenate([y for _, y in data])[np.concatenate([y for _, y in data]) >= 0],
                         minlength=len(SPEAKERS))
    weight = torch.tensor(counts.sum() / (len(SPEAKERS) * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)
    model = FrameSpk().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps)
    lossf = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
    T = int(args.crop * FPS)
    lens = np.array([len(y) for _, y in data], float)
    rng = random.Random(0)
    torch.manual_seed(0)
    for step in range(1, args.steps + 1):
        xb, yb = [], []
        for _ in range(args.batch):
            X, y = data[rng.choices(range(len(data)), weights=lens)[0]]
            a = rng.randrange(0, max(len(y) - T, 1))
            xb.append(torch.from_numpy(X[a:a + T].astype(np.float32)))
            yb.append(torch.from_numpy(y[a:a + T]))
        L = min(len(v) for v in yb)
        x = torch.stack([v[:L] for v in xb]).to(DEVICE)
        x = x + args.noise * torch.randn_like(x) * x.std()
        y = torch.stack([v[:L] for v in yb]).to(DEVICE)
        loss = lossf(model(x).reshape(-1, len(SPEAKERS)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "speakers": SPEAKERS, "layers": LAYERS,
                "sets": sorted(sets), "excluded": args.exclude_ids, "steps": args.steps}, out / "model.pt")
    print(f"saved {out / 'model.pt'}; layer mix {model.mix.softmax(0).tolist()}")


@torch.no_grad()
def posteriors(model: FrameSpk, X: np.ndarray) -> np.ndarray:
    """Long-context BiLSTM over the whole episode: 60 s windows with 5 s context on each side."""
    model.eval()
    n, win, ctx = len(X), 60 * FPS, 5 * FPS
    P = np.zeros((n, len(SPEAKERS)), np.float32)
    for f0 in range(0, n, win):
        a, b = max(f0 - ctx, 0), min(f0 + win + ctx, n)
        p = model(torch.from_numpy(X[a:b].astype(np.float32))[None].to(DEVICE))[0].softmax(-1).cpu().numpy()
        f1 = min(f0 + win, n)
        P[f0:f1] = p[f0 - a:f0 - a + f1 - f0]
    return P


def adapt(model: FrameSpk, X: np.ndarray, y: np.ndarray, steps: int, crop: float = 8.0, batch: int = 16) -> None:
    """Fine-tune on this episode's own confident frames (pseudo-labels; no reference used)."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    counts = np.bincount(y[y >= 0], minlength=len(SPEAKERS))
    weight = torch.tensor(np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0), dtype=torch.float32,
                          device=DEVICE)
    lossf = nn.CrossEntropyLoss(weight=weight / weight[weight > 0].mean(), ignore_index=-1)
    T, rng = int(crop * FPS), random.Random(1)
    for _ in range(steps):
        starts = [rng.randrange(0, max(len(y) - T, 1)) for _ in range(batch)]
        x = torch.stack([torch.from_numpy(X[a:a + T].astype(np.float32)) for a in starts]).to(DEVICE)
        t = torch.stack([torch.from_numpy(y[a:a + T]) for a in starts]).to(DEVICE)
        loss = lossf(model(x).reshape(-1, len(SPEAKERS)), t.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def cmd_infer(args) -> None:
    ck = torch.load(args.model, map_location="cpu")
    model = FrameSpk().to(DEVICE).eval()
    model.load_state_dict(ck["state_dict"])
    members = args.members.split(",")
    cols = [ck["speakers"].index(m) for m in members]

    wave, _ = load_window(args.audio, args.start, args.duration)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "feats.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        X = features(load_wavlm(), wave)
        np.save(cache, X)
    n = len(X)
    speech = np.zeros(n, bool)
    for t in json.loads(Path(args.vad).read_text(encoding="utf-8"))["regular"]:
        a, b = int((t["start"] - args.start) * FPS), int((t["end"] - args.start) * FPS)
        if b > 0 and a < n:
            speech[max(a, 0):min(b, n)] = True
    P = posteriors(model, X)
    for r in range(args.adapt_rounds):
        Q = P[:, cols] / P[:, cols].sum(1, keepdims=True)
        y = np.full(n, -1, np.int64)
        ok = speech & (Q.max(1) >= args.adapt_conf)
        y[ok] = np.array(cols)[Q[ok].argmax(1)]
        print(f"adapt round {r + 1}: pseudo-labelled {ok.sum() / FPS:.0f}s of {speech.sum() / FPS:.0f}s speech "
              + " ".join(f"{m}={np.sum(y == c) / FPS:.0f}s" for m, c in zip(members, cols, strict=True)), flush=True)
        adapt(model, X, y, args.adapt_steps)
        P = posteriors(model, X)
    np.save(out / "posteriors.npy", P)

    label = P[:, cols].argmax(1)
    k = int(args.smooth * FPS) // 2
    if k:
        pad = np.pad(label, k, mode="edge")
        label = np.array([np.bincount(pad[i:i + 2 * k + 1], minlength=len(members)).argmax() for i in range(n)])
    label[~speech] = -1

    turns, cur = [], None
    for i in range(n + 1):
        lab = label[i] if i < n else -1
        if cur is not None and lab != cur[1]:
            turns.append({"start": round(args.start + cur[0] / FPS, 3), "end": round(args.start + i / FPS, 3),
                          "speaker": members[cur[1]]})
            cur = None
        if cur is None and lab >= 0:
            cur = (i, lab)
    payload = {"model": f"frame_spk WavLM-base-plus layers {ck['layers']} + BiLSTM ({','.join(ck['sets'])})",
               "audio": str(Path(args.audio).resolve()), "offset": args.start, "duration": round(n / FPS, 3),
               "smooth": args.smooth, "adapt_rounds": args.adapt_rounds, "regular": turns, "exclusive": turns,
               "speakers": {m: {"total_s": round(sum(t["end"] - t["start"] for t in turns if t["speaker"] == m), 2)}
                            for m in members}}
    (out / "diarization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(turns)} turns: " + ", ".join(f"{m}={v['total_s']}s" for m, v in payload["speakers"].items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--sources", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--sets", default="studio_ass,studio_ass_unproofed,koelog")
    t.add_argument("--feats", default="", help="feature cache dir (default <out>/feats)")
    t.add_argument("--exclude-ids", default="", help="comma-separated source ids left out (leave-one-episode-out)")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--crop", type=float, default=8.0)
    t.add_argument("--noise", type=float, default=0.05)
    i = sub.add_parser("infer")
    i.add_argument("audio")
    i.add_argument("out_dir")
    i.add_argument("--model", required=True)
    i.add_argument("--members", required=True)
    i.add_argument("--vad", required=True)
    i.add_argument("--start", type=float, default=0.0)
    i.add_argument("--duration", type=float, default=0.0)
    i.add_argument("--smooth", type=float, default=0.3)
    i.add_argument("--adapt-rounds", type=int, default=0, help="self-training rounds on this episode's confident frames")
    i.add_argument("--adapt-steps", type=int, default=300)
    i.add_argument("--adapt-conf", type=float, default=0.9)
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_infer(args)


if __name__ == "__main__":
    main()
