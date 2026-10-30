# Diagnostic Tools

This directory separates maintainable diagnostic source from machine-local tool
environments and generated data.

## Source Tools

### Split Content-Loss Diagnosis

`diagnose_split_content_loss.py` compares the effective pre-split text with
`split_segments.json`. It reports content conservation, protected short-response
counts, adjacent duplicates, source fallback reasons, file SHA-256 values, and
bounded diff examples.

Required input for each `--workdir`:

- `aligned_segments.json` or `manifests/aligned_segments.json`
- `transcript_segments.json` or `manifests/transcript_segments.json`
- `split_segments.json` or `manifests/split_segments.json`

Example:

```powershell
.venv312\Scripts\python.exe tools\diagnose_split_content_loss.py `
  --workdir workspaces\example `
  --output-json tmp_codex_pytest\split-content-loss.json `
  --output-md tmp_codex_pytest\split-content-loss.md
```

### Split Readability Diagnosis

`diagnose_split_readability.py` reads `split_segments.json` and reports short
ordinary subtitles, protected short responses, non-positive intervals, overlaps,
very long subtitles, adjacent duplicates, and likely merge directions.

Required input for each `--workdir`:

- `split_segments.json`

Example:

```powershell
.venv312\Scripts\python.exe tools\diagnose_split_readability.py `
  --workdir workspaces\example `
  --output-json tmp_codex_pytest\split-readability.json `
  --output-md tmp_codex_pytest\split-readability.md
```

Both tools accept repeated `--workdir` arguments and write UTF-8 JSON and
Markdown reports. Use an ignored temporary or workspace path for their outputs.

### Radio Highlight Clips

`radio_clips.py` scores highlight-clip candidates of a full radio episode with an
LLM, then cuts the ones you pick to mp3 (lossless stream copy, like LosslessCut).
Logic lives in `qwen_asr/clip_planner.py`; prompts are
`prompts/radio_clip_selection.md` (selection, scores, titles) and
`prompts/radio_clip_boundary.md` (second pass that checks each clip's first and
last cue).

```powershell
# 1. scored candidate list: <out>\45.candidates.md (+ 45.plan.json)
.venv312\Scripts\python.exe tools\radio_clips.py plan `
  --srt output\ep45.srt --media "D:\Users\下载\【源】刺刺广播第45期0811会员版.mp3" `
  --out output\radio-clips --label 45
# optional: redo only the boundary check (e.g. after a network error)
.venv312\Scripts\python.exe tools\radio_clips.py refine --plan output\radio-clips\45.plan.json
# 2. cut the picked numbers to <out>\clips\#45-01 标题.mp3; --adjust N:START-END moves a
#    candidate's cue range, --with-srt also slices the full-episode subtitle
.venv312\Scripts\python.exe tools\radio_clips.py cut `
  --plan output\radio-clips\45.plan.json --pick 1,3 --adjust 3:160-230 --with-srt
```

Cut rule (measured on 8 hand-cut clips, ~0.4s mean error): start 1.0s before the
first cue but never inside the previous cue (0.15s lead when they touch), end
1.0s after the last cue. `cut` recomputes times from the stored cue ranges, so a
changed rule also applies to old plans. `--with-srt` writes `【ai】<clip>.srt` next to
the mp3; a sentence cut by the clip end (heard only in the fade-out tail) is left
out of the subtitle.

## Local Dependencies

The following directories are machine-local state, not project source:

- `mfa-env/`: local Montreal Forced Aligner environment
- `mfa-root/`: local MFA models, corpora, and runtime state
- `micromamba/`: local environment bootstrap files
- `__pycache__/`: rebuildable Python bytecode cache

They remain in place and are ignored by Git. The complete micromamba, MFA 3.4.0,
and `japanese_mfa` model rebuild commands are documented under "Rebuilding The
Optional MFA Environment" in `docs/PIPELINE.md`. Moving, overwriting, or deleting
any existing local dependency still requires explicit path-level approval.
