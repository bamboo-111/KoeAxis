"""pyannote community-1 with its clustering step replaced by enrolled voiceprints.

Usage (whisperx venv, HF_HUB_OFFLINE=1):
  .venv-whisperx312/Scripts/python.exe experiments/speaker-change/segenroll_diarize.py <audio.wav> <out_dir>
      --enroll runs/enroll/enroll_embeddings.json --members 理名,朱李 [--sets studio_ass,studio_ass_unproofed]
      [--start S --duration D] [--backend centroid|logreg]

Everything pyannote does well is kept: the powerset segmentation model (local speakers per 10 s chunk, overlap
aware), speaker counting, per-chunk local embeddings and reconstruction. Only the global clustering (which merges
or splits our hosts wrongly) is replaced: in every chunk the active local speakers are matched one-to-one to the
on-air members (Hungarian on cosine similarity to enrolled centroids, or on logreg log-probabilities); extra local
speakers fall back to their best member. Output diarization.json has the same format as pyannote_diarize.py.
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
from pyannote.audio import Pipeline
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from pyannote_diarize import MODEL, annotation_rows, load_window  # noqa: E402


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class EnrolledAssignment:
    """Drop-in for pipeline.clustering: returns (hard_clusters, soft_clusters, centroids)."""

    expects_num_clusters = False

    def __init__(self, rows: list[dict], members: list[str], backend: str) -> None:
        E = l2(np.array([r["embedding"] for r in rows], dtype=np.float64))
        spk = np.array([r["speaker"] for r in rows])
        self.members = members
        self.centroids = l2(np.stack([E[spk == m].mean(0) for m in members]))
        self.lr = None
        if backend == "logreg":
            from sklearn.linear_model import LogisticRegression
            self.lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(E, spk)
            self.lr_idx = [list(self.lr.classes_).index(m) for m in members]
        self.stats = {"assigned": 0, "fallback": 0}

    def score(self, emb: np.ndarray) -> np.ndarray:
        if self.lr is not None:
            return np.log(self.lr.predict_proba(emb)[:, self.lr_idx] + 1e-9)
        return emb @ self.centroids.T

    def __call__(self, embeddings: np.ndarray, segmentations=None, **_) -> tuple:
        n_chunks, n_local, _ = embeddings.shape
        hard = np.full((n_chunks, n_local), -2, dtype=np.int8)
        soft = np.zeros((n_chunks, n_local, len(self.members)))
        for c in range(n_chunks):
            ok = ~np.any(np.isnan(embeddings[c]), axis=1)
            if segmentations is not None:  # only speakers active in this chunk compete for members
                ok &= segmentations.data[c].sum(0) > 0
            if not ok.any():
                continue
            local = np.where(ok)[0]
            S = self.score(l2(embeddings[c, local]))
            soft[c, local] = S
            rows, cols = linear_sum_assignment(-S)
            hard[c, local[rows]] = cols
            self.stats["assigned"] += len(rows)
            for i in set(range(len(local))) - set(rows):
                hard[c, local[i]] = S[i].argmax()
                self.stats["fallback"] += 1
        return hard, soft, self.centroids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("--enroll", required=True)
    ap.add_argument("--sets", default="studio_ass,studio_ass_unproofed")
    ap.add_argument("--members", required=True, help="comma-separated on-air members")
    ap.add_argument("--backend", default="centroid", choices=["centroid", "logreg"])
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    args = ap.parse_args()

    members, sets = args.members.split(","), set(args.sets.split(","))
    rows = [r for r in json.loads(Path(args.enroll).read_text(encoding="utf-8")) if r["set"] in sets]
    pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    pipeline.to(torch.device("cuda"))
    assign = EnrolledAssignment(rows, members, args.backend)
    pipeline.clustering = assign

    seen: list[list] = []  # integer cluster labels of each annotation, before pyannote renames them SPEAKER_xx
    to_annotation = pipeline.to_annotation

    def recording_to_annotation(*a, **kw):
        ann = to_annotation(*a, **kw)
        seen.append(sorted(ann.labels()))
        return ann
    pipeline.to_annotation = recording_to_annotation

    wave, sr = load_window(args.audio, args.start, args.duration)
    t0 = time.time()
    result = pipeline({"waveform": wave, "sample_rate": sr}, max_speakers=len(members))
    elapsed = time.time() - t0
    # pipeline maps the sorted integer labels of the regular annotation to SPEAKER_00, SPEAKER_01, ...
    name = {f"SPEAKER_{i:02d}": members[int(k)] for i, k in enumerate(seen[0])}
    regular = annotation_rows(result.speaker_diarization, args.start)
    exclusive = annotation_rows(result.exclusive_speaker_diarization, args.start)
    for r in regular + exclusive:
        r["speaker"] = name[r["speaker"]]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": f"{MODEL} segmentation+embedding, enrolled {args.backend} assignment ({args.sets})",
               "audio": str(Path(args.audio).resolve()), "offset": args.start, "duration": round(wave.shape[1] / sr, 3),
               "elapsed_s": round(elapsed, 2), "regular": regular, "exclusive": exclusive,
               "speakers": {m: {"total_s": round(sum(r["end"] - r["start"] for r in exclusive if r["speaker"] == m), 2)}
                            for m in members}}
    (out / "diarization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{elapsed:.1f}s, local speakers assigned {assign.stats}: "
          + ", ".join(f"{m}={v['total_s']}s" for m, v in payload["speakers"].items()), flush=True)


if __name__ == "__main__":
    main()
