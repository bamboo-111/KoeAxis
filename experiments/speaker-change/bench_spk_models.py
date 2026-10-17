"""Benchmark speaker-embedding models on our data: enroll on old studio clips, test on the 30期 gold pieces.

Usage (whisperx venv, HF_HUB_OFFLINE=1):
  .venv-whisperx312/Scripts/python.exe experiments/speaker-change/bench_spk_models.py \
      --sources experiments/speaker-change/enroll_sources.json \
      --enroll experiments/speaker-change/runs/enroll/enroll_embeddings.json \
      --gold experiments/speaker-change/runs/cici30-full/probe/gold_pieces_0.5.json \
      --gold-audio workspaces/radio-cici-30-gemini38-20261017/audio/source.wav \
      --out experiments/speaker-change/runs/bench [--models pyannote34,resnet293,...] [--sets studio_ass,...]

The piece lists (time spans + speaker) are reused from enroll_eval.py / embed_probe.py outputs; every model
re-embeds the same audio. Embeddings are cached per model in <out>/<model>.npz. Back-ends compared, all
restricted to the speakers present in the gold (the on-air members are known per episode):
  centroid : cosine to per-speaker mean            knn5   : 5 nearest enrollment pieces vote
  logreg   : logistic regression on enrollment     lda    : LDA projection + centroid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "third_party"))
from pyannote_diarize import SAMPLE_RATE, load_window  # noqa: E402

CACHE = Path(".model-cache/speaker")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_MODELS = ["pyannote34", "resnet221", "resnet293", "cnceleb34", "campplus", "ecapa", "wavlm", "redimnet_m"]


def fbank(wave: torch.Tensor, cmn: bool = True) -> torch.Tensor:
    import torchaudio.compliance.kaldi as kaldi
    feats = kaldi.fbank(wave * 32768, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                        sample_frequency=SAMPLE_RATE, window_type="hamming")
    return feats - feats.mean(0, keepdim=True) if cmn else feats


def load_model(name: str):
    """Return embed(list of (1, T) float tensors at 16 kHz) -> np.ndarray (N, D)."""
    if name == "pyannote34":
        from pyannote.audio import Pipeline
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=os.environ.get("HF_TOKEN"))
        pipe.to(DEVICE)
        emb = pipe._embedding
        return lambda waves: np.concatenate([np.asarray(emb(w.unsqueeze(0))) for w in waves])
    if name in ("resnet221", "resnet293", "cnceleb34"):
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        sess = ort.InferenceSession(str(CACHE / name / "model.onnx"), providers=providers)
        inp = sess.get_inputs()[0].name

        def run(waves):
            return np.concatenate([sess.run(None, {inp: fbank(w).unsqueeze(0).numpy()})[0] for w in waves])
        return run
    if name == "campplus":
        from campplus.DTDNN import CAMPPlus
        model = CAMPPlus(feat_dim=80, embedding_size=192)
        model.load_state_dict(torch.load(CACHE / "campplus" / "campplus_cn_common.bin", map_location="cpu"))
        model.eval().to(DEVICE)

        def run(waves):
            with torch.no_grad():
                return np.concatenate([model(fbank(w).unsqueeze(0).to(DEVICE)).cpu().numpy() for w in waves])
        return run
    if name == "ecapa":
        from speechbrain.inference.speaker import EncoderClassifier
        clf = EncoderClassifier.from_hparams(source=str((CACHE / "ecapa").resolve()),
                                             savedir=str((CACHE / "ecapa").resolve()), run_opts={"device": str(DEVICE)})

        def run(waves):
            with torch.no_grad():
                return np.concatenate([clf.encode_batch(w.to(DEVICE)).squeeze(1).cpu().numpy() for w in waves])
        return run
    if name == "wavlm":
        from transformers import AutoFeatureExtractor, WavLMForXVector
        fe = AutoFeatureExtractor.from_pretrained(str(CACHE / "wavlm-sv"))
        model = WavLMForXVector.from_pretrained(str(CACHE / "wavlm-sv")).eval().to(DEVICE)

        def run(waves):
            out = []
            with torch.no_grad():
                for w in waves:
                    x = fe(w.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt").to(DEVICE)
                    out.append(model(**x).embeddings.cpu().numpy())
            return np.concatenate(out)
        return run
    if name == "redimnet_m":
        model = load_redimnet()

        def run(waves, bs: int = 64):
            # equal-length waves (sliding windows) are batched (~13x faster, identical output); others one by one
            out = []
            with torch.no_grad():
                if len({w.shape[-1] for w in waves}) == 1:
                    for k in range(0, len(waves), bs):
                        out.append(model(torch.cat(waves[k:k + bs]).to(DEVICE)).cpu().numpy())
                else:
                    out = [model(w.to(DEVICE)).cpu().numpy() for w in waves]
            return np.concatenate(out)
        return run
    raise ValueError(name)


def load_redimnet() -> torch.nn.Module:
    """ReDimNet-M (vox2 + CN-Celeb fine-tuned); input (B, T) 16 kHz waveform, output (B, 192)."""
    from redimnet.model import ReDimNetWrap
    full = torch.load(CACHE / "redimnet" / "M-vb2+vox2+cnc-ft_mix.pt", map_location="cpu")
    model = ReDimNetWrap(**full["model_config"])
    model.load_state_dict(full["state_dict"])
    return model.eval().to(DEVICE)


def gather(args) -> tuple[list[dict], list[dict]]:
    sources = {s["id"]: s for s in json.loads(Path(args.sources).read_text(encoding="utf-8"))["sources"]}
    sets = set(args.sets.split(","))
    enroll = [{k: r[k] for k in ("set", "id", "speaker", "start", "end")}
              for r in json.loads(Path(args.enroll).read_text(encoding="utf-8"))
              if r["set"] in sets and r["id"] not in args.exclude_ids.split(",")]
    for r in enroll:
        r["audio"] = sources[r["id"]]["audio"]
    gold = [{"speaker": g["speaker"], "start": g["start"], "end": g["end"], "audio": args.gold_audio}
            for g in json.loads(Path(args.gold).read_text(encoding="utf-8"))]
    return enroll, gold


def cut_all(pieces: list[dict]) -> list[torch.Tensor]:
    waves, cache = [], {}
    for p in pieces:
        if p["audio"] not in cache:
            cache[p["audio"]] = load_window(p["audio"], 0, 0)[0]
        x = cache[p["audio"]]
        a, b = int(p["start"] * SAMPLE_RATE), int(p["end"] * SAMPLE_RATE)
        waves.append(x[:, a:b].contiguous())
    return waves


def l2(E: np.ndarray) -> np.ndarray:
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def backends(E: np.ndarray, spk: np.ndarray, G: np.ndarray, members: list[str]) -> dict[str, np.ndarray]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    E, G = l2(E), l2(G)
    m = np.isin(spk, members)
    Em, sm = E[m], spk[m]
    C = l2(np.stack([Em[sm == n].mean(0) for n in members]))
    out = {"centroid": np.array(members)[(G @ C.T).argmax(1)]}
    K = G @ Em.T
    out["knn5"] = np.array([Counter(sm[np.argsort(r)[::-1][:5]]).most_common(1)[0][0] for r in K])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(E, spk)
    proba = lr.predict_proba(G)
    idx = [list(lr.classes_).index(n) for n in members]
    out["logreg"] = np.array(members)[proba[:, idx].argmax(1)]
    lda = LinearDiscriminantAnalysis().fit(E, spk)
    Lm, Lg = l2(lda.transform(Em)), l2(lda.transform(G))
    CL = l2(np.stack([Lm[sm == n].mean(0) for n in members]))
    out["lda"] = np.array(members)[(Lg @ CL.T).argmax(1)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--enroll", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--gold-audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--sets", default="studio_ass,studio_ass_unproofed")
    ap.add_argument("--exclude-ids", default="", help="enrollment source ids to leave out (the test episode)")
    ap.add_argument("--fuse", default="", help="comma-separated models to fuse (concatenated L2 embeddings)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    enroll, gold = gather(args)
    spk = np.array([r["speaker"] for r in enroll])
    truth = np.array([g["speaker"] for g in gold])
    members = sorted(set(truth))
    dur = np.array([g["end"] - g["start"] for g in gold])
    waves_e = waves_g = None
    print(f"enroll {len(enroll)} pieces {dict(Counter(spk.tolist()))}; gold {len(gold)} {dict(Counter(truth.tolist()))}")

    results = {}
    for name in args.models.split(","):
        f = out / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            E, G, secs = z["E"], z["G"], float(z["secs"])
        else:
            if waves_e is None:
                waves_e, waves_g = cut_all(enroll), cut_all(gold)
            try:
                embed = load_model(name)
            except Exception as e:  # noqa: BLE001 - report and continue with other models
                print(f"{name}: load failed: {type(e).__name__}: {e}")
                continue
            t0 = time.time()
            E, G = embed(waves_e), embed(waves_g)
            secs = time.time() - t0
            np.savez(f, E=E, G=G, secs=secs)
        results[name] = (E, G)
        preds = backends(E, spk, G, members)
        short = dur < 1.0
        row = " ".join(f"{k} {np.mean(v == truth):6.1%}" for k, v in preds.items())
        best = max(preds, key=lambda k: np.mean(preds[k] == truth))
        per = " ".join(f"{n}{np.mean(preds[best][truth == n] == n):.0%}" for n in members)
        print(f"{name:11} dim {E.shape[1]:4d} {secs:6.1f}s | {row} | best={best}: {per}, <1s {np.mean(preds[best][short] == truth[short]):.0%} (n={short.sum()})",
              flush=True)

    for combo in [c for c in args.fuse.split(";") if c]:
        names = [n for n in combo.split(",") if n in results]
        E = np.concatenate([l2(results[n][0]) for n in names], 1)
        G = np.concatenate([l2(results[n][1]) for n in names], 1)
        preds = backends(E, spk, G, members)
        print(f"fuse[{'+'.join(names)}] | " + " ".join(f"{k} {np.mean(v == truth):6.1%}" for k, v in preds.items()))


if __name__ == "__main__":
    main()
