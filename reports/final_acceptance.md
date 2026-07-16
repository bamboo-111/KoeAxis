# Project Cleanup Final Acceptance

Status: `COMPLETE`

Generated: `2026-07-15`

This is the current authoritative acceptance report for `docs/项目文件与代码整理计划.md`. Earlier versions are preserved in timestamped backups. P0-P7 planned work and the final protected-environment resolution have passed.

## Executive Result

| Area | Result | Evidence |
|---|---|---|
| P0 baseline | PASS | Inventory, Git state, original 384-test baseline, compileall and CLI evidence recorded. |
| P1 classification | PASS | `source_test_mapping.*`, `change_groups.*`; 60 production modules mapped, 0 unmapped. |
| P2 audit | PASS | `p2_approval_status.json`; P2-D remains optional and unapproved. |
| P3 tools/state boundaries | PASS | `local_state_register.*`, `evidence_reference_map.*`, benchmark dry-run evidence. |
| P4 documentation | PASS | README, ARCHITECTURE, PIPELINE, STATUS, CLI and Web behavior reconciled. |
| P5 retirement/modularization | PASS | Four P5-0 manifests, helper audit, focused and full regression evidence. |
| P6 development gates | PASS | Ruff 0.15.21 pinned; source-only compileall, pytest, Ruff and diff check pass. |
| P7 closeout | PASS | `p7_closeout.*`, refreshed inventory/classification/retention reports and production smoke. |
| Final protected-environment acceptance | PASS | 8,931 exact approved cache files backed up, deleted and revalidated; 0 remain. |

Final acceptance: `PASS`

Goal complete: `true`

## Authorization And Safety

- P5-0 machine execution basis: `reports/rejected_feature_inventory.json`, approval revision 2.
- Approved SHA-256: `e2c3600888033a51c107037b072387a11f8921fe2378fcbf9101a01a4ff583b4`.
- Executed order: `P5-0-A-llm-split`, `P5-0-B-mfa-full`, `P5-0-D-align-variants`, `P5-0-E-diagnostic-tools`.
- Every batch received a separate pre-edit backup and passed its targeted acceptance before the next batch.
- P5-0-C remains `KEEP_LOCAL_FALLBACK`; MFA local proofread fallback was not deleted or moved.
- Historical workspaces, reports, products, imported data, models and virtual environments were not moved by the approved P5-0 actions.
- No Git stage, commit or push was performed.

## P0-P4 Evidence

### P0 Baseline

- Original full regression baseline: `384 passed`.
- Baseline inventory used for P7 comparison: `backups/inventory-refresh-20260715-122904/project_inventory.json`.
- Status taxonomy distinguishes `KEEP`, `ARCHIVE_CANDIDATE`, `DELETE_CANDIDATE`, and `UNKNOWN`; candidate labels do not authorize action.

### P1 Git And Review Scope

- Current review scope: 222 paths.
- New production modules: 60 mapped, 0 unmapped.
- One unclassified path remains intentionally explicit: `1/clip.json`.
- The path is a 137-byte word-tier JSON sample outside every approved destructive package. It remains in place pending owner classification.
- Change groups include purpose and rollback guidance; nothing was staged or committed.

### P2 Local State And Optional Approval

Current `reports/p2_approval_status.json`:

- Status: `AWAITING_OPTIONAL_APPROVAL`.
- `goal_blocking=false`.
- Candidates: 34,856.
- `DELETE_CANDIDATE`: 8,845.
- `ARCHIVE_CANDIDATE`: 128.
- `UNKNOWN`: 25,883.
- Approved: 0.
- Archived: 0.
- Deleted: 0.
- Every file candidate has SHA-256 evidence.

No P2-D archive or deletion is required for Goal completion. An empty optional approval set is valid.

### P3 Tools And Protected State

- 127 workspaces remain in place: 1 current baseline, 22 key evidence, 104 unknown.
- 96 manual-test directories remain in place: 1 key evidence, 95 unknown.
- 519 timestamped backup directories remain as rollback evidence.
- Missing referenced evidence paths: 0.
- `.venv312`, WhisperX environments, `.model-cache`, MFA state, micromamba, workspaces, backups and generated benchmark runs have explicit owner/rebuild/retention records.
- The documented MFA rebuild procedure remains optional and does not overwrite an existing environment.

### P4 Documentation

- README now describes rule-only production split; the rejected LLM split invocation has been removed.
- ARCHITECTURE records the rejected full MFA aligner at `tools/mfa_full_alignment.py` and forbids production imports from `tools/`.
- PIPELINE states that Qwen is the sole production main aligner, `rule` is the sole production split implementation, MFA local is default-off proofread fallback, and quality-gate remains mandatory before formal export.
- STATUS distinguishes current behavior from historical `COMPLETED_REJECTED` experiment documents.
- No stale `AWAITING_REAPPROVAL`, zero-approved P5-0 statement, `qwen_asr/mfa_backend.py` production path, or LLM split usage remains in current specifications.

## P5 Rejected Feature Retirement

### P5-0-A LLM Split

- Deleted the 7 approved LLM split implementation, prompt and dedicated-test paths.
- Removed approved CLI, Web and production branches.
- Preserved translation LLM, MiMo, generic `optimizer/llm_client.py`, rule split and historical evidence.
- Targeted acceptance: `177 passed in 2.64s`.
- Manifest: `reports/p5_0_a_execution_manifest.json`, status `PASS`.

### P5-0-B MFA Full Alignment

- Moved `qwen_asr/mfa_backend.py` to `tools/mfa_full_alignment.py`.
- Removed the rejected formal full-alignment backend from production CLI/Pipeline/Web paths.
- Preserved MFA local proofread fallback, environment detection, evidence and report compatibility.
- Targeted acceptance: `118 passed in 7.33s`.
- Manifest: `reports/p5_0_b_execution_manifest.json`, status `PASS`.

### P5-0-D Align Variants

- Deleted `qwen_asr/align_timing_repair.py` and its dedicated test.
- Removed only rejected experiment parameters.
- Preserved timing legality, coverage, zero-run, dense coverage, local-collapse and quality-gate checks that remain production protections.
- Targeted acceptance: `91 passed in 5.24s`.
- Manifest: `reports/p5_0_d_execution_manifest.json`, status `PASS`.

### P5-0-E Diagnostic Tools

- Moved tuning matrix, baseline snapshot, align diagnose and align split audit modules to `tools/`.
- Removed their production CLI registrations and production-package imports.
- Preserved `qwen_asr/ass_quality_diff.py`, `qwen_asr/pipeline_runner.py` and `qwen_asr/quality_suspects.py` production behavior.
- Targeted acceptance: `112 passed in 5.20s`.
- Manifest: `reports/p5_0_e_execution_manifest.json`, status `PASS`.

### Quantified Retirement

- Recorded actions: 9 deletes, 5 moves and 33 edits across 36 unique approved paths.
- Approved production/test source paths are currently 197,323 bytes smaller than their earliest approved baselines.
- `optimizer/` is 60,982 bytes smaller than the P0 inventory snapshot despite the new extracted rule helpers.
- Four-batch post-retirement gate: `553 passed`; compileall, Ruff, CLI/import/help, Web parameters and production smoke passed.

## P5 Modularization

- `commands/stages.py`: command handlers extracted while compatibility exports remain.
- `mimo_proofread.py`: candidates, requests, checkpoints, guards, inputs, audio, application and outputs separated with schema compatibility.
- `optimizer/splitter.py`: boundaries, readability, display duration and timing allocation separated; dead LLM matching code removed.
- Splitter rule parameters are grouped in frozen readability, display-duration and inline-timing dataclasses.
- `optimizer_bridge.py`: adapter, stage invocation and guard responsibilities separated while production imports remain stable.
- `final_quality.py`: independent checker modules plus shared, tested status/result/numeric helpers.
- MFA, proofread realignment and history glossary boundaries are split into focused modules with compatibility tests.
- Duplicate-helper audit scanned 962 production functions, found 12 structurally identical groups, and consolidated only helpers with the same semantics, ownership and tested contract. See `reports/p5_duplicate_helper_audit.json`.

Latest source/test scope:

- `qwen_asr`: 90 Python files, 19,716 lines.
- `optimizer`: 20 Python files, 4,021 lines.
- Tests: 79 Python files, 14,300 lines.

## P6 And Final Technical Gates

- Ruff version: `0.15.21`.
- Ruff hard gate: `E4/E7/E9`, `F`, `B`.
- Development dependency is pinned in `requirements-dev.txt` and `pyproject.toml`.
- mypy was evaluated but intentionally not made a hard gate for the current dynamic CLI/JSON/model-adapter surface.

Final checks:

- `scripts/local_check.ps1`: `557 passed in 10.71s`.
- Source-only compileall: PASS.
- Ruff: `All checks passed`.
- `git diff --check`: PASS.
- Inventory/local-state/review audit: `8 passed in 0.25s`.
- CLI/Web/quality-gate production smoke: `68 passed in 1.02s`.
- MFA environment: version `3.4.0`, PASS.
- MFA local fallback: `53 passed in 0.60s`.
- CLI/import/help for main, split, align and run: PASS.
- Review scope: 60 mapped, 0 unmapped.
- Production imports from `tools/`: 0.
- JSON reports parse and current Chinese documents decode as UTF-8.

The quality-gate regression explicitly proves that a failed final gate prevents formal output completion and stops export. Qwen main alignment and rule split remain the production defaults.

## P7 Closeout

- Refreshed project inventory, Git file classification, archive candidates and P2 approval status.
- Refreshed local-state register and evidence reference map.
- Compared current disk/file state with the P0 inventory.
- Reconciled README, ARCHITECTURE, PIPELINE, STATUS, CLI help and Web behavior.
- Generated `reports/p7_closeout.json` and `.md` with changes, retention, optional candidates and risks.
- Root paths `$env`, `None`, `.agents` and `1/clip.json` are explicitly explained and remain unmodified because no destructive action is approved.

The large ignored-file count is expected to include local environments, models and protected evidence. It is not a request or authorization to clean them.

## Resolved Environment Incident

An extra command recursively compiled `tools/` and generated or updated `.pyc` files inside the protected MFA environment. This was outside the intended source-only compile scope.

Approved incident package:

- Path: `reports/p5_0_compileall_environment_incident.json`.
- SHA-256: `2cd735bee6d6ba529bf879c48cf814103794abb5bf537d4dc20e7adf9cff7e5f`.
- Exact candidates: 8,931 `.pyc` paths under `tools/mfa-env/`.
- Approved action: delete only the 8,931 exact listed `DELETE_CANDIDATE` files.
- Preflight: 8,931 files and 148,496,525 bytes matched path, size and SHA-256 evidence.
- Backup: 8,931/8,931 files copied with original relative paths and rehashed successfully.
- Backup path: `backups/p5-0-compileall-incident-delete-20260715-224938`.
- Deleted: 8,931 files.
- Remaining approved paths: 0.
- Directories deleted: 0.
- Package-external files deleted: 0.
- Execution manifest: `reports/p5_0_compileall_environment_incident_execution.json`, status `PASS`.
- `goal_blocking=false`.

Post-deletion verification passed: MFA `3.4.0`, MFA local fallback `53 passed`, full `local_check` `557 passed`, Ruff, source-only compileall, `git diff --check`, CLI/import/help, Web parameters and production smoke. No approved path was recreated.

## Non-Blocking Environment Warning

`pip check` reports:

```text
torch 2.11.0+cu128 has requirement setuptools<82, but you have setuptools 83.0.0.
```

No setuptools downgrade was performed because it would modify the authorized primary model environment. This warning is documented separately from the MFA cache incident and is not treated as a code/test regression.

## Final Decision

- P0-P7 planned work: `PASS`.
- Code, tests, documentation and P7 evidence: `PASS`.
- Protected environment integrity acceptance: `PASS`.
- Overall final acceptance: `COMPLETE`.
- Goal status: `complete`.
