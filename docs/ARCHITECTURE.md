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
- `optimizer/` contains local subtitle splitting, cleanup, translation, and LLM client utilities.

## Compatibility Rules

- Keep `main.py` and `webapp.py` as importable wrappers.
- Keep `optimizer.text_utils` as a compatibility export layer for cleanup and text metric helpers.
- Do not rename workdir artifact files without a migration plan.
- Cleanup may delete only validated first-level project directories under `workspaces/`.
- Do not delete `work-*`, `backups`, `dist`, local virtual environments, or user media paths during cleanup.
