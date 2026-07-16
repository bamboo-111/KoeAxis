# Architecture

This project is split into stable entrypoints and implementation packages.

## Entrypoints

- `main.py` is the compatibility CLI wrapper.
- `webapp.py` is the compatibility WebUI wrapper.
- `qwen_asr/cli.py` owns argparse setup and command dispatch.
- `qwen_asr/web/server.py` owns the local HTTP server.

## Core Packages

- `qwen_asr/commands/` contains CLI stage implementations.
- `qwen_asr/stages.py` defines stage order, artifact inputs, outputs, and cleanup targets.
- `qwen_asr/artifact_state.py` evaluates artifact completeness, missing inputs, outdated outputs, and cleanup paths.
- `qwen_asr/progress.py` writes the stable `progress.json` payload.
- `qwen_asr/model_runtime.py` centralizes model cache, device, dtype, and cleanup helpers shared by ASR and align adapters.
- `qwen_asr/mfa_experiment.py` and its focused helpers isolate the optional MFA local proofread fallback from the default Qwen align path. The rejected full-alignment experiment lives at `tools/mfa_full_alignment.py` and is not imported by production modules.
- `qwen_asr/quality_suspects.py`, `qwen_asr/content_quality.py`, `qwen_asr/ass_quality.py`, `qwen_asr/ass_quality_diff.py`, and `qwen_asr/final_quality.py` provide layered quality gates for subtitle content, timing, ASS diff checks, and final acceptance.
- `qwen_asr/proofread_realign.py` and `qwen_asr/mimo_proofread.py` handle suspects-only proofreading, checkpointed LLM calls, protected edit application, and optional post-proofread realignment.
- `optimizer/` contains rule-only subtitle splitting, cleanup, translation, and shared LLM client utilities. The LLM client remains for translation and proofreading, not production split.
- `optimizer/splitter.py` remains the compatibility entrypoint for the sole production split implementation. Boundary, readability, display-duration, and timing helpers keep public input and output schemas stable.

## Local State Boundaries

The repository treats virtual environments, model caches, workspaces, backup directories, benchmark run artifacts, MFA corpora, and micromamba downloads as local runtime state. These paths are ignored precisely in `.gitignore`, inventoried through `scripts/project_inventory.py`, and must not be deleted or moved without explicit approval.

Source-level benchmark metadata remains in `benchmarks/`: environment notes, summaries, reproduction commands, and report scripts. Generated run directories, audio chunks, logs, and stage manifests under `benchmarks/**/runs/` are local artifacts or archive candidates.

## Web Workbench Boundaries

- `qwen_asr/web/workspace_api.py` owns versioned envelopes, workspace scope checks, quality/export inventories, and API error translation.
- `qwen_asr/web/stage_service.py`, `stage_start_service.py`, and `job_state.py` expose structured stage state, safe CLI payload preparation, and restart-reconciled job persistence.
- `qwen_asr/recovery_service.py` owns failed-dialogue recovery tasks and append-only action evidence; it reuses the production VAD adapter.
- `qwen_asr/review_service.py` combines cue/audio/reference data and owns `drafts/web-review.json`, automatic draft backups, revision checks, JSONL audit, and last-edit undo. Formal manifests are never silently overwritten by Web editing.
- `qwen_asr/alignment_state.py` is the shared authority for `completed_exact`, `completed_coarse`, `failed`, and `SKIPPED_MUSIC_REGION` across CLI, quality, recovery, and Web.
- `qwen_asr/artifact_state.py` remains the shared completion/outdated authority. A dirty review draft marks `quality-gate`, `normalize`, and `export` outdated without deleting formal artifacts.
- `qwen_asr/web/templates/workbench.html` plus `static/workbench.css` and `static/workbench.js` implement the default workbench. The legacy configuration UI remains isolated at `/legacy`.

All Web file access is inventory-based or constrained to the selected first-level workspace and manifest-linked media roots. API credentials are environment-only; versioned start requests reject credential fields before process creation.

## Compatibility Rules

- Keep `main.py` and `webapp.py` as importable wrappers.
- Keep `optimizer.text_utils` as a compatibility export layer for cleanup and text metric helpers.
- Do not rename workdir artifact files without a migration plan.
- Cleanup may delete only validated first-level project directories under `workspaces/`.
- Do not delete `work-*`, `backups`, `dist`, local virtual environments, or user media paths during cleanup.
- Do not change CLI defaults, checkpoint schemas, manifest filenames, or report JSON fields during module extraction unless a versioned migration is documented and tested.
- Production packages must not import experiment modules from `tools/`; tools may import production libraries for reproducibility.
