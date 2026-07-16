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
