# Pipeline

The registered stage order is:

```text
prepare -> transcribe -> correct -> align -> split -> translate -> mimo-proofread -> proofread-realign -> quality-gate -> normalize -> export
```

The minimal `run` path is `prepare -> transcribe -> quality-gate -> export`. Optional flags insert `correct`, `align`, `split`, `translate`, MiMo proofreading plus proofread realignment, and `normalize` in the registered order. The quality gate always runs before formal export and a failing gate stops the pipeline.

Production defaults use silence-first segmentation with a 15 second maximum segment window, project-local `.model-cache`, `local_files_only` enabled unless explicitly disabled, adaptive ASR batching, and stage checkpoints for resumable `transcribe` and `align`. The CLI leaves `batch_size` unset initially; adaptive resolution selects 3, 4, or 5 from the current segment-duration distribution, while fixed mode falls back to 5 when no explicit size is supplied.

Qwen ForcedAligner is the only production main aligner. Split has one production implementation, `rule`; the rejected LLM split variants and MFA full-alignment backend are absent from CLI, Web, and PipelineRunner. MFA local remains available only as an explicit, default-off proofread-realignment fallback. Translation LLM, MiMo, the generic LLM client, timing legality checks, and the mandatory pre-export quality gate remain supported.

After Align, the Web recovery layer routes every failed dialogue segment. Music regions are represented separately as `SKIPPED_MUSIC_REGION`; they remain inspectable but are excluded from dialogue recovery counts and the live alignment quality gate. A recovery action that accepts `completed_coarse` first backs up aligned manifest/checkpoint/event evidence, atomically updates shared alignment state, and leaves the prior quality FAIL visible until quality is rerun.

Web cue editing writes only `drafts/web-review.json`. Each edit validates cue identity, non-empty source text, positive timing, non-overlap, and expected revision; it backs up an existing draft and appends `reports/web_review_history.jsonl`. A dirty review draft invalidates quality/normalize/export presentation without silently replacing `normalized_segments.json` or exported subtitles. Undo restores the previous draft cue while preserving formal manifests and audit history.

## Artifacts

- `prepare`: writes `audio/source.wav`, `segments.json`, and optionally `audio/segments/*.wav` when eager segment export is enabled.
- `transcribe`: reads `segments.json`, writes `transcript_segments.json`, `transcript.txt`, `transcript_events.jsonl`, `transcript_checkpoint.json`, and optional `transcribe.profile.json`.
- `correct`: reads `transcript_segments.json`, writes `corrected_segments.json`, and updates `transcript_segments.json`.
- `align`: reads `transcript_segments.json`, writes `aligned_segments.json`, `aligned_events.jsonl`, and `aligned_checkpoint.json`.
- `mfa`: optional local proofread fallback plus standalone experiment tooling; MFA environments, corpora, pretrained models, and micromamba files are local state under `tools/` and are not tracked as source.
- `split`: rule-only production stage; reads `aligned_segments.json`, writes `split_segments.json` and `subtitles.split.srt`.
- `translate`: reads `split_segments.json`, writes `translated_segments.json` and `subtitles.translated.srt`.
- `proofread`: optional MiMo suspects-only flow; reads candidate/suspect inputs, writes checkpointed batch state, evidence reports, and protected edit outputs without changing the default subtitle schema.
- `quality`: optional quality gates; content, ASS, diff, final quality, and proofread-realign checks write reports while preserving source segment schema.
- `normalize`: reads the best configured subtitle source, writes `normalized_segments.json` and `subtitles.normalized.srt`.
- `export`: reads the selected subtitle source, writes `subtitles.srt` and/or `subtitles.vtt`.
- `web recovery`: writes recovery state/action evidence below `reports/` and, only for an explicitly accepted coarse fallback, backed-up aligned state.
- `web review`: writes `drafts/web-review.json`, `reports/web_review_history.jsonl`, and `reports/review-backups/`; it does not overwrite formal manifests.

## Progress

`progress.json` uses these stable fields:

```json
{
  "stage": "translate",
  "status": "running",
  "done": 3,
  "total": 8,
  "current": "batch 1",
  "updated_at": "2026-06-06T00:00:00Z",
  "summary": "3/8 translated subtitles"
}
```

Status values are `running`, `completed`, `failed`, and `skipped`.

## Local Verification

Use the project environment when available:

```powershell
.\scripts\local_check.ps1
```

The script runs `compileall`, the full pytest suite, Ruff, and `git diff --check`. Install the pinned development dependencies with `.venv312\Scripts\python.exe -m pip install -r requirements-dev.txt` before running it. `-SkipRuff` is diagnostic-only and is not accepted for final verification.

## Rebuilding The Optional MFA Environment

MFA is local experimental state and is not required for the default Qwen alignment path. The currently verified local versions are micromamba `2.8.1` and Montreal Forced Aligner `3.4.0`, with the `japanese_mfa` acoustic model and dictionary.

From the repository root on a new Windows machine:

```powershell
New-Item -ItemType Directory -Force tools\micromamba\extract, tools\mfa-root | Out-Null
Invoke-WebRequest https://micro.mamba.pm/api/micromamba/win-64/latest `
  -OutFile tools\micromamba\micromamba.tar.bz2
tar.exe -xjf tools\micromamba\micromamba.tar.bz2 -C tools\micromamba\extract

$mamba = Resolve-Path tools\micromamba\extract\Library\bin\micromamba.exe
& $mamba create -y -p tools\mfa-env -c conda-forge "montreal-forced-aligner=3.4.0"

$env:MFA_ROOT_DIR = (Resolve-Path tools\mfa-root).Path
& $mamba run -p tools\mfa-env mfa model download acoustic japanese_mfa
& $mamba run -p tools\mfa-env mfa model download dictionary japanese_mfa
& $mamba run -p tools\mfa-env mfa version
```

The final command must print `3.4.0`. Runtime discovery checks the micromamba environment first, then `tools\mfa-env\Scripts\mfa.exe`, then an `mfa` executable on `PATH`. `MFA_ROOT_DIR` should continue to point to `tools\mfa-root` so downloaded models and corpora remain project-local and ignored by Git.

These commands require network access to micromamba/conda-forge and the MFA model registry. Existing `tools\mfa-env`, `tools\mfa-root`, and `tools\micromamba` directories must not be overwritten, moved, or deleted merely to test the instructions.
