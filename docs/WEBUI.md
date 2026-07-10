# WebUI

The WebUI is intentionally local-only and defaults to:

```text
http://127.0.0.1:8765
```

## Modules

- `webapp.py`: compatibility wrapper.
- `qwen_asr/web/server.py`: HTTP routes, job lifecycle, process stop behavior.
- `qwen_asr/web/commands.py`: payload-to-CLI command construction.
- `qwen_asr/web/status.py`: artifact status, logs, progress summaries.
- `qwen_asr/web/static_html.py`: template loader.
- `qwen_asr/web/templates/index.html`: browser UI.

## API Routes

- `GET /`: returns the UI.
- `GET /api/status?workdir=...`: returns artifact state, counts, and recent logs.
- `GET /api/job`: returns current job and progress.
- `GET /api/suggest-workdir?media=...`: suggests a new `workspaces/NNNN-name` directory.
- `GET /api/workspaces`: lists deletable workspace cache directories.
- `POST /api/start`: starts one stage or the whole pipeline.
- `POST /api/stop`: stops the active job.
- `POST /api/pick-media`: opens a local file picker and returns the selected absolute path.
- `POST /api/delete-workspace`: deletes a first-level project directory under `workspaces/`.
- `POST /api/delete-workspaces`: deletes selected first-level project directories under `workspaces/`, or all listed workspace caches when no list is provided.

## Notes

The WebUI starts stages as subprocesses through the CLI entrypoint. This preserves GPU memory release behavior between model-heavy stages.

## Layout

The browser UI is organized as a local workbench:

- Top bar: current status, media path selection/manual input, workdir, single/all workspace cache deletion, and primary `Run` / `Resume` / `Stop` actions.
- Config rail/drawer: a 52px rail by default, expanding to configuration groups for input/segmentation, ASR/Align, LLM/translation/correction, and Normalize/Export.
- Workspace: job summary, pipeline timeline, artifact readiness, and stage-scoped logs.

The page still uses the same local storage key, `qwen3_asr_webui_settings_v2`, so existing browser-side settings continue to load.
The drawer open/closed UI state is stored separately in `qwen3_asr_webui_ui_state_v1`.

## Frontend Validation

Before starting a job, the WebUI checks the most common missing inputs:

- `prepare` and normal `run` require media path and workdir.
- custom export mode requires an export path.
- `translate`, `correct`, and full `run` with translate/correct enabled require LLM model, base URL, and API key.
- `split` can still run without LLM settings and falls back to the existing rule-based behavior.

Validation failures automatically open the configuration drawer and focus the field that needs attention.

## Runtime Behavior

- Existing browser settings default to a collapsed configuration rail so the workspace is the primary screen.
- First-time browser sessions open the configuration drawer so required settings are discoverable.
- While a job is running or stopping, configuration inputs and stage run buttons are disabled; stop, refresh, and log switching remain available.
- Logs use a light code-style panel consistent with the rest of the WebUI color system.
