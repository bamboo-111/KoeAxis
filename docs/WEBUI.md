# WebUI

The WebUI is intentionally local-only and defaults to:

```text
http://127.0.0.1:8765
```

## Modules

- `webapp.py`: compatibility wrapper.
- `qwen_asr/web/server.py`: HTTP routes, job lifecycle, process stop behavior.
- `qwen_asr/web/commands.py`: payload-to-CLI command construction.
- `qwen_asr/web/workspace_api.py`: versioned envelopes, workspace/path boundaries, recovery/review actions, and quality/export inventories.
- `qwen_asr/recovery_executor.py` and `qwen_asr/recovery_service.py`: shared real retry/undo execution plus transcript, language, VAD and coarse gates used by CLI and Web.
- `qwen_asr/web/stage_service.py`: structured stage state and evidence.
- `qwen_asr/web/stage_start_service.py`: non-secret saved settings plus project defaults to shared CLI payloads.
- `qwen_asr/web/job_state.py`: redacted, restart-reconciled job persistence.
- `qwen_asr/web/status.py`: artifact status, logs, progress summaries.
- `qwen_asr/web/static_html.py`: template loader.
- `qwen_asr/web/templates/workbench.html` and `static/workbench.*`: default workbench.
- `qwen_asr/web/templates/index.html`: legacy configuration UI at `/legacy`.

## API Routes

- `GET /`: returns the structured workbench; `GET /legacy` returns the compatibility configuration UI.
- `GET /api/v1/contract`, `/job`, `/workspaces`, `/workspace`, `/workspace/stages`, `/workspace/align`, `/workspace/recovery`, `/workspace/review`, `/workspace/quality`, and `/workspace/exports`: versioned structured state.
- `POST /api/v1/workspace/stage/start`: validates inputs/settings, rejects credential fields, and starts a shared CLI stage job.
- `POST /api/v1/workspace/recovery/action`: executes `verify_transcript`, `localize_vad`, `route_language`, real `retry_align`, guarded `accept_completed_coarse`, or target-level `undo_recovery` and returns the resulting workspace state/evidence.
- `POST /api/v1/workspace/review/edit` and `/review/undo`: revisioned draft editing with backup and audit.
- `GET /api/v1/workspace/media`: workspace or manifest-linked media with HTTP Range support.
- `GET /api/v1/workspace/quality-evidence`: inventory-scoped quality report preview.
- `GET /api/v1/workspace/export-file`: inventory-scoped subtitle preview/download.
- The unversioned routes below remain for compatibility:
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

The default browser UI is organized as a local workbench:

- Top bar: workspace selection, persisted job state, stop, and explicit refresh.
- Navigation: pipeline, failed-segment recovery, cue review, quality evidence, and exports.
- Pipeline: status/count/duration/evidence plus safe per-stage continue buttons and domain-view jumps.
- Recovery: all failed dialogues, short-response and root-cause filters, audio/context/evidence, transcript verification, real language-routed Qwen/MFA retry, VAD/coarse hard gates, before/after metrics, and auditable undo.
- Review: cue table with exact/coarse/failed/issues filters, full-audio seeking, editable draft fields, dirty/revision state, undo, and read-only ASS references.
- Quality/export: every WARN/FAIL has a recovery/cue/report target; exports retain `quality_gate_failed` while quality is FAIL.

The legacy config rail/drawer remains at `/legacy` for input/segmentation, ASR/Align, LLM/translation/correction, and Normalize/Export settings.

The page stores non-sensitive settings under `koeaxis_webui_settings_v1`. Existing `qwen3_asr_webui_settings_v3` and earlier v2 settings are migrated, sanitized, and then removed.
The drawer open/closed UI state is stored separately in `qwen3_asr_webui_ui_state_v1`.

Credential fields are intentionally absent from the HTML and request payloads. Existing browser settings are sanitized on load so legacy credential-like fields are removed before the settings are written back. MiMo uses `MIMO_API_KEY`, DeepSeek official endpoints use `DEEPSEEK_API_KEY`, and other compatible providers use `LLM_API_KEY`.

Recovery API semantics are intentionally strict: Qwen retry defaults to `original_transcript`; verified text requires a prior `verify_transcript` action and explicit opt-in. `mfa-local` requires a Japanese route. `accept_completed_coarse` requires verified text and a safe VAD selection; no-speech or ambiguous-region cases return a stable conflict error and leave the aligned manifest unchanged. A quality-gate process returning code 1 is represented as an honest stage failure when the report status is `FAIL`.

## Frontend Validation

Before starting a job, the WebUI checks the most common missing inputs:

- `prepare` and normal `run` require media path and workdir.
- custom export mode requires an export path.
- `translate`, `correct`, and full `run` with translate/correct enabled require an LLM model and base URL. Credentials are resolved only from the Web server process environment.
- `split` can still run without LLM settings and falls back to the existing rule-based behavior.

Validation failures automatically open the configuration drawer and focus the field that needs attention.

## Runtime Behavior

- Existing browser settings default to a collapsed configuration rail so the workspace is the primary screen.
- First-time browser sessions open the configuration drawer so required settings are discoverable.
- While a job is running or stopping, configuration inputs and stage run buttons are disabled; stop, refresh, and log switching remain available.
- Logs use a light code-style panel consistent with the rest of the WebUI color system.
- Workbench polling requests only `/api/v1/job`; it does not rebuild cue/stage DOM or reset scroll position every three seconds.
- Desktop and mobile navigation preserve keyboard names and focus outlines. At widths up to 560px, the bottom navigation hides visual labels while retaining `aria-label` names.
