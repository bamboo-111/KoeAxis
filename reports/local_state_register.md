# Local State Register

Generated: `2026-07-15T15:01:27.340398+00:00`

No path in this report was moved, archived, or deleted.

| Path | Size | Status | Owner | Purpose | Rebuild | Retention |
|---|---:|---|---|---|---|---|
| `.venv312` | 5.37 GB | KEEP_PRIMARY | Project development and production runtime | Primary Python environment for compile, tests, Ruff, CLI, and model execution. | Create with Python 3.12, then install requirements.txt and requirements-dev.txt. | KEEP_PRIMARY; do not move because virtual environments may contain absolute paths. |
| `.venv-whisperx` | 28.69 MB | UNKNOWN | Legacy WhisperX experiment | Python 3.14 virtual-environment shell; no WhisperX, faster-whisper, or torch distribution was detected. | No verified rebuild is required because no active code or document references this shell. | UNKNOWN; retain in place until a path-level decision is approved. |
| `.venv-whisperx312` | 7.82 GB | KEEP_LOCAL_EXPERIMENT | WhisperX/faster-whisper experiment | Functional Python 3.12 experiment environment for WhisperX 3.8.6 and faster-whisper 1.2.1; not used by the production Qwen path. | Create with Python 3.12 and install whisperx==3.8.6 with its compatible torch/faster-whisper stack. | KEEP_LOCAL_EXPERIMENT until its historical evidence is no longer required and a path-level decision is approved. |
| `.model-cache` | 15.27 GB | KEEP_REBUILDABLE | Project model runtime | Project-local Qwen, faster-whisper, and MDX model cache used to keep local_files_only execution deterministic. | Download the listed model IDs/revisions into .model-cache with network access explicitly enabled. | KEEP_REBUILDABLE; large downloads are ignored and never auto-deleted. |
| `tools/mfa-env` | 5.90 GB | KEEP_LOCAL_EXPERIMENT | Optional MFA experiment runtime | Micromamba prefix containing Montreal Forced Aligner and its native dependencies. | Follow docs/PIPELINE.md to create MFA 3.4.0 under tools/mfa-env. | KEEP_LOCAL_EXPERIMENT; ignored and not part of production source. |
| `tools/mfa-root` | 856.02 MB | KEEP_KEY_EVIDENCE | Optional MFA experiment runtime | Project-local MFA models, corpora, caches, configuration, and command history. | Set MFA_ROOT_DIR and download japanese_mfa acoustic and dictionary models as documented in docs/PIPELINE.md. | KEEP_KEY_EVIDENCE; command history and models support historical MFA reports. |
| `tools/micromamba` | 15.30 MB | KEEP_REBUILDABLE | Optional MFA experiment runtime | Project-local micromamba bootstrap and executable. | Download the current Windows micromamba package and extract it as documented in docs/PIPELINE.md. | KEEP_REBUILDABLE; ignored and not production source. |
| `workspaces` | 11.31 GB | MIXED_REVIEWED | Pipeline experiments and acceptance evidence | 127 independent workspaces containing current baselines, historical evidence, and unclassified experiments. | Individual workspaces require their recorded input media, commands, model revisions, and manifests; they are not globally rebuildable from source alone. | Mixed: CURRENT_BASELINE and KEY_EVIDENCE are retained; UNKNOWN remains in place pending review. |
| `backups` | 19.29 GB | KEEP_ROLLBACK | Rollback evidence for uncommitted cleanup work | 519 topic/timestamp backup directories created before file edits. | Not rebuildable by design; each backup preserves a pre-edit state. | KEEP_ROLLBACK until a stable commit exists and a later full regression passes. |
| `tmp_manual_tests` | 4.49 GB | MIXED_REVIEWED | Historical manual experiments | 96 manual-test directories containing mixed derived data and at least one still-referenced reliable ASS path. | Mixed and not globally reproducible; use per-directory evidence before considering archive. | Mixed KEY_EVIDENCE/UNKNOWN; the directory is not proven replaceable by formal workspaces. |

## Model Cache

| Model | Revision | Size | Status | Rebuild |
|---|---|---:|---|---|
| `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5` | 8.76 GB | KEEP_REBUILDABLE | Populate through the Qwen ASR loader with local_files_only disabled once, or download the pinned revision into .model-cache. |
| `Qwen/Qwen3-ForcedAligner-0.6B` | `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` | 3.43 GB | KEEP_REBUILDABLE | Populate through the Qwen aligner loader with local_files_only disabled once, or download the pinned revision into .model-cache. |
| `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` | 3.02 GB | KEEP_REBUILDABLE | Re-download with faster-whisper/WhisperX into .model-cache/faster-whisper when that experiment is explicitly needed. |
| `UVR-MDX-NET-Inst_HQ_3.onnx` | `local model filename` | 63.71 MB | KEEP_REBUILDABLE | Re-download through the configured audio-separator model provider into .model-cache/mdx-net. |

## Backup Directories

Count: `519`
