# P5-0 Rejected Feature Retirement Inventory

Generated: 2026-07-15 20:39 Asia/Shanghai

Status: AWAITING_REAPPROVAL (approval revision 2)

No files were deleted, moved, archived, staged, committed, or pushed. This package is an audit and approval artifact only.

## Approval Summary

- Approved paths: 0
- Executed deletions: 0
- Executed moves: 0
- Executed archives: 0
- Goal blocking: true. The updated project plan makes P5-0 a required retirement gate before further P5 modularization. The audit package is complete, but retirement execution cannot begin until the affected paths receive explicit path-level approval.

## Revision 2 Preflight Correction

Revision 1 was explicitly approved for A/B/D/E, but no execution began. Mandatory preflight invalidated that revision before any code change because ten unique affected files had no package fingerprint and several files are intentionally edited by more than one sequential batch.

Revision 2 does not change any `approval_batches` delete, move, edit, preserve, or keep path set. It adds:

- Baseline size and SHA-256 for every affected file.
- A first-touch rule requiring the revision-2 baseline fingerprint.
- A later-touch rule requiring a shared file to match the `after_sha256` recorded by the immediately preceding execution manifest that touched it.
- Explicit shared-path order for `qwen_asr/cli.py`, `qwen_asr/commands/align.py`, `qwen_asr/commands/stages.py`, `qwen_asr/align_split_audit.py`, `qwen_asr/tuning_matrix.py`, `tests/test_align_split_audit.py`, and `tests/test_cli_help.py`.
- One machine-readable execution manifest path per batch.

Revision 1 SHA-256: `f04afeaba9983f87bf4f4341e1a738f1e34de4c9d02ee13ab029335c3730e825`.

No A/B/D/E file has been edited, deleted, or moved. Revision 2 requires explicit reapproval before P5-0-A starts.

## Corrected Approval Batches

The first inventory omitted several real import, CLI, Web, and integration-test paths. The corrected machine-readable package now includes five exact decision batches. No approval is inferred from the plan text.

### P5-0-A LLM split

- Delete after approval: `optimizer/split_by_llm.py`, `optimizer/token_boundary_split.py`, the four `optimizer/prompts/split/*.md` LLM prompts, and `tests/test_token_boundary_split.py`.
- Edit after approval: `optimizer/splitter.py`, `qwen_asr/optimizer_bridge.py`, `qwen_asr/cli.py`, `qwen_asr/web/templates/index.html`, `qwen_asr/web/commands.py`, `qwen_asr/align_split_audit.py`, `qwen_asr/tuning_matrix.py`, `tests/test_optimizer_bridge_timing.py`, `tests/test_align_split_audit.py`, `tests/test_cli_help.py`, and `tests/test_webui.py`.
- Preserve: translation LLM, MiMo, generic `llm_client`, rule split, and all historical evidence.

### P5-0-B MFA full alignment

- Move after approval: `qwen_asr/mfa_backend.py` -> `tools/mfa_full_alignment.py`.
- Edit after approval: `qwen_asr/commands/align.py`, `qwen_asr/commands/stages.py`, `qwen_asr/cli.py`, `qwen_asr/pipeline_runner.py`, `tests/test_mfa_backend.py`, `tests/test_align_cleanup.py`, `tests/test_pipeline_runner.py`, and `tests/test_cli_help.py`.
- Preserve: Qwen production alignment, separately approved local-fallback helpers, and historical MFA evidence.

### P5-0-C MFA local fallback

- Resolved action: `KEEP_LOCAL_FALLBACK`; no destructive path approval is required for retaining it.
- Independent benefit evidence: `workspaces/p4-proofread-realign-20260713-195633/reports/p4_proofread_realign_status.md` records Madougushi02 id 219 and Konoato01 id 110 completed by MFA local after Qwen did not complete those local candidates.
- Final P4 acceptance in `docs/字幕流程完整改造计划与验收清单.md:758` records Madougushi02 3/3 and Konoato01 4/4 completed, with fallback 0 and failed 0; each dataset includes one MFA-local success.
- This does not reverse the full-MFA rejection: Qwen remains the only production main aligner, while MFA local remains explicit, default off, and guarded inside proofread-realign.
- Affected production scope: `qwen_asr/proofread_realign.py`, `qwen_asr/proofread_realign_strategy.py`, `qwen_asr/mfa_experiment.py`, `qwen_asr/mfa_candidates.py`, `qwen_asr/mfa_environment.py`, `qwen_asr/mfa_guards.py`, `qwen_asr/mfa_lab.py`, `qwen_asr/mfa_report.py`, `qwen_asr/mfa_runner.py`, `qwen_asr/mfa_words.py`, `qwen_asr/mfa_writeback.py`, and `qwen_asr/cli.py`.
- Affected tests: `tests/test_proofread_realign.py`, `tests/test_cli_help.py`, `tests/test_mfa_backend.py`, `tests/test_mfa_candidates.py`, `tests/test_mfa_environment.py`, `tests/test_mfa_experiment.py`, `tests/test_mfa_guards.py`, `tests/test_mfa_lab.py`, `tests/test_mfa_report.py`, `tests/test_mfa_runner.py`, `tests/test_mfa_words.py`, and `tests/test_mfa_writeback.py`.

### P5-0-D Align timing variants

- Delete after approval: `qwen_asr/align_timing_repair.py` and `tests/test_align_timing_repair.py`.
- Edit after approval: `qwen_asr/align.py`, `qwen_asr/commands/align.py`, `qwen_asr/commands/stages.py`, `qwen_asr/cli.py`, `tests/test_align_quality.py`, and `tests/test_cli_help.py`.
- Preserve: independent timing-legality and final-quality checks plus historical matrix evidence.

### P5-0-E Diagnostic tools

- Move after approval: `qwen_asr/tuning_matrix.py` -> `tools/tuning_matrix.py`; `qwen_asr/baseline_snapshot.py` -> `tools/baseline_snapshot.py`; `qwen_asr/align_diagnose.py` -> `tools/align_diagnose.py`; `qwen_asr/align_split_audit.py` -> `tools/align_split_audit.py`.
- Edit after approval: `qwen_asr/cli.py`, the CLI wrapper boundary in `qwen_asr/ass_quality_diff.py`, and the corresponding tool/CLI tests.
- Keep in production: `qwen_asr/ass_quality_diff.py`, because `qwen_asr/pipeline_runner.py` imports its report builder for production quality-suspect routing; also keep `qwen_asr/quality_suspects.py` and their tests.

## Direct Delete Conditions

1. COMPLETED_REJECTED evidence is present.
2. No current production default, PipelineRunner, Web, or formal CLI call remains.
3. No inseparable shared implementation remains.
4. Historical reports, metrics, commands, and conclusions are preserved.
5. Manifest, checkpoint, and report schemas remain readable.
6. A precise file/symbol/CLI/test deletion list has path-level approval.

## Feature Inventory

### LLM split modes

- ID: `llm_split_modes`
- Status: COMPLETED_REJECTED evidence found
- Candidate action: `DELETE`
- Production exposure: Formal CLI, Web form, qwen_asr.optimizer_bridge split_mode branches, and split prompts are still present.
- Audit result: Not safe for immediate deletion because production CLI/Web/run exposure remains and path-level approval is absent.
- Approved: false
- Executed: false

Evidence:
- docs/字幕流程完整改造计划与验收清单.md:752 records Madougushi02 LLM split probes rejected for content conservation or protected short-response failures.
- docs/字幕流程完整改造计划与验收清单.md:888 records Konoato01 LLM split probe evidence and concludes rule remains the only production default.
- docs/项目文件与代码整理计划.md:318 requests removal from production CLI/Web/prompt/run paths, preserving historical reports.

Code symbols:
- `qwen_asr.optimizer_bridge.split_aligned_manifest(split_mode=...)`
- `optimizer.split_by_llm.split_text_by_llm`
- `optimizer.token_boundary_split.split_aligned_payload_by_token_counts`
- `optimizer.token_boundary_split.split_aligned_payload_by_token_delimited_text`
- `optimizer.token_boundary_split.split_aligned_payload_by_token_boundaries`

CLI/Web entries:
- qwen_asr/cli.py --split-mode choices include text/token-counts/token-delimited/token-boundary/rule for split/run/batch-run.
- qwen_asr/web/templates/index.html splitMode select includes token-counts/token-delimited/token-boundary/text/rule.
- qwen_asr/web/commands.py forwards payload split_mode to split/run commands.

Path-level candidates:
- `delete_after_approval`: `optimizer/split_by_llm.py`
- `delete_after_approval`: `optimizer/token_boundary_split.py`
- `delete_after_approval`: `optimizer/prompts/split/sentence.md`
- `delete_after_approval`: `optimizer/prompts/split/token_counts.md`
- `delete_after_approval`: `optimizer/prompts/split/token_delimited.md`
- `delete_after_approval`: `optimizer/prompts/split/token_boundary.md`
- `edit_after_approval`: `qwen_asr/optimizer_bridge.py`
- `edit_after_approval`: `qwen_asr/cli.py`
- `edit_after_approval`: `qwen_asr/web/templates/index.html`
- `edit_after_approval`: `qwen_asr/web/commands.py`
- `delete_after_approval`: `tests/test_token_boundary_split.py`

### MFA full main alignment backend

- ID: `mfa_full_main_alignment`
- Status: COMPLETED_REJECTED evidence found
- Candidate action: `MOVE_TO_TOOLS`
- Production exposure: Formal align/run/batch-run expose --align-backend mfa; qwen_asr.commands.align can call run_mfa_full_alignment.
- Audit result: Not safe for immediate deletion because the formal CLI still exposes MFA as an align backend and local fallback shares MFA modules.
- Approved: false
- Executed: false

Evidence:
- docs/字幕流程完整改造计划与验收清单.md:652 records MFA full replacement as COMPLETED_REJECTED and Qwen remains main alignment.
- docs/字幕流程完整改造计划与验收清单.md:828 records double-dataset MFA/Qwen A-B rejection.
- docs/项目文件与代码整理计划.md:319 requests removing MFA full alignment from production default/composite pipeline and evaluating a tools move.

Code symbols:
- `qwen_asr.mfa_backend.run_mfa_full_alignment`
- `qwen_asr.commands.align.cmd_align align_backend == mfa branch`
- `qwen_asr.pipeline_runner.PipelineRunner.build_invocations passes align_backend through run/batch-run`

CLI/Web entries:
- qwen_asr/cli.py align/run/batch-run --align-backend choices include mfa.
- PipelineRunner still forwards align_backend into the align stage.

Path-level candidates:
- `move_to_tools_after_approval`: `qwen_asr/mfa_backend.py` -> `tools/mfa_full_alignment.py`
- `edit_after_approval`: `qwen_asr/commands/align.py`
- `edit_after_approval`: `qwen_asr/cli.py`
- `edit_after_approval`: `qwen_asr/pipeline_runner.py`
- `edit_after_approval`: `tests/test_mfa_backend.py`

### MFA proofread local fallback

- ID: `mfa_proofread_local_fallback`
- Status: independent local-benefit evidence found; retain as an explicit default-off fallback
- Candidate action: `KEEP_LOCAL_FALLBACK`
- Production exposure: Formal proofread-realign/run/batch-run expose MFA local fallback controls, default off.
- Audit result: Resolved as `KEEP_LOCAL_FALLBACK`. Madougushi02 id 219 and Konoato01 id 110 provide independent accepted local recoveries; the full-MFA backend remains rejected and separate.
- Approved: false
- Executed: false

Evidence:
- docs/字幕流程完整改造计划与验收清单.md:717 records MFA local only as explicit experiment/candidate fallback, not default.
- docs/字幕流程完整改造计划与验收清单.md:758 records final accepted MFA-local recoveries in both datasets with complete proofread-realign results.
- workspaces/p4-proofread-realign-20260713-195633/reports/p4_proofread_realign_status.md records the per-item methods and Qwen-to-MFA recovery reason.
- docs/项目文件与代码整理计划.md:320 says keep only if independent local-use evidence supports it, otherwise retire separately.

Code symbols:
- `qwen_asr.proofread_realign.run_proofread_realign_stage`
- `qwen_asr.proofread_realign._try_mfa_local_realign`
- `qwen_asr.mfa_experiment.run_local_mfa_alignment_experiments`

CLI/Web entries:
- qwen_asr/cli.py proofread-realign/run/batch-run --proofread-realign-mfa-fallback choices off/local.

Path-level candidates:
- `keep_or_retire_after_approval`: `qwen_asr/proofread_realign.py`
- `keep_or_retire_after_approval`: `qwen_asr/proofread_realign_strategy.py`
- `keep_or_move_shared_helpers_after_approval`: `qwen_asr/mfa_experiment.py`
- `edit_after_approval`: `qwen_asr/cli.py`
- `delete_or_retarget_after_approval`: `tests/test_proofread_realign.py`

### No-benefit align timing parameter variants

- ID: `align_parameter_variants`
- Status: COMPLETED_REJECTED evidence found for the experimental matrix
- Candidate action: `DELETE`
- Production exposure: Formal align/run/batch-run expose --align-timing-repair min-duration/local-interpolate.
- Audit result: Not safe for immediate deletion because the parameter is still read by production align code and tests cover timing legality behavior.
- Approved: false
- Executed: false

Evidence:
- docs/字幕流程完整改造计划与验收清单.md:381 records prior align timing repair A-B evidence superseded by final matrix.
- docs/字幕流程完整改造计划与验收清单.md:863 records 9 variants with no target metric improvement.
- docs/项目文件与代码整理计划.md:321 requests removing production CLI switches that only serve failed experiments while keeping valid defaults and checks.

Code symbols:
- `qwen_asr.align_timing_repair.AlignTimingRepairConfig`
- `qwen_asr.align_timing_repair.repair_token_timing`
- `qwen_asr.align.QwenAligner.run_segment timing_repair_config`
- `qwen_asr.cli._add_align_timing_arguments`

CLI/Web entries:
- qwen_asr/cli.py align/run/batch-run --align-timing-repair choices off/min-duration/local-interpolate.

Path-level candidates:
- `delete_after_approval`: `qwen_asr/align_timing_repair.py`
- `edit_after_approval`: `qwen_asr/align.py`
- `edit_after_approval`: `qwen_asr/commands/align.py`
- `edit_after_approval`: `qwen_asr/commands/stages.py`
- `edit_after_approval`: `qwen_asr/cli.py`
- `delete_after_approval`: `tests/test_align_timing_repair.py`
- `edit_after_approval`: `tests/test_align_quality.py`

### Parameter matrix, A/B, and diagnostic reporters

- ID: `experiment_diagnostic_tools`
- Status: experiment reproduction tools only
- Candidate action: `MOVE_TO_TOOLS_WITH_KEEP_PRODUCTION_CORE`
- Production exposure: Formal CLI exposes diagnostic subcommands from qwen_asr; PipelineRunner imports the `ass_quality_diff` report builder for production quality-suspect routing.
- Audit result: Move the four standalone experiment tools only. Keep the `ass_quality_diff` production report core and retire or relocate only its formal diagnostic CLI wrapper after approval.
- Approved: false
- Executed: false

Evidence:
- docs/项目文件与代码整理计划.md:322 requests moving parameter matrix, A-B, and diagnostic reporters to tools or archive.
- docs/Goal模式完整任务提示词.md:76 lists these modules as existing quality/diagnostic modules.

Code symbols:
- `qwen_asr.tuning_matrix.cmd_tuning_matrix`
- `qwen_asr.baseline_snapshot.cmd_baseline_snapshot`
- `qwen_asr.align_diagnose.cmd_align_diagnose`
- `qwen_asr.align_split_audit.cmd_align_split_audit`
- `qwen_asr.ass_quality_diff.cmd_ass_quality_diff`

CLI/Web entries:
- qwen_asr/cli.py subcommands tuning-matrix, baseline-snapshot, align-diagnose, align-split-audit, ass-quality-diff.

Path-level candidates:
- `move_to_tools_after_approval`: `qwen_asr/tuning_matrix.py`
- `move_to_tools_after_approval`: `qwen_asr/baseline_snapshot.py`
- `move_to_tools_after_approval`: `qwen_asr/align_diagnose.py`
- `move_to_tools_after_approval`: `qwen_asr/align_split_audit.py`
- `move_to_tools_after_approval`: `qwen_asr/ass_quality_diff.py`
- `edit_after_approval`: `qwen_asr/cli.py`

### Automatic publish or quality-gate bypass paths

- ID: `quality_gate_bypass_or_auto_publish`
- Status: no approval to add bypass; existing quality gate blocks remain active
- Candidate action: `KEEP_EVIDENCE_ONLY`
- Production exposure: PipelineRunner explicitly runs final quality before formal export completion and stops on non-zero status.
- Audit result: No deletion candidate. Preserve and test the blocking gate while documenting that bypass is not approved.
- Approved: false
- Executed: false

Evidence:
- docs/项目文件与代码整理计划.md:323 says no bypass path; quality-gate continues to block formal output.
- reports/final_acceptance.md records pipeline tests covering export/normalize blocks when quality gate fails.

Code symbols:
- `qwen_asr.pipeline_runner.PipelineRunner.run`
- `qwen_asr.pipeline_runner.PipelineRunner._run_final_gate`
- `qwen_asr.commands.quality.cmd_quality_gate`

CLI/Web entries:
- Common --skip-preflight and --dry-run-check exist but are not quality-gate completion bypass approvals.
- qwen_asr/preflight.py dry_run_reserved warning is diagnostic only.

Path-level candidates:
- None

## File Fingerprints

- `optimizer/split_by_llm.py` size=13382 sha256=4f270e77f3f514902671bb584506ae5e5a36a543b3c389f896f047f425fb27f1
- `optimizer/token_boundary_split.py` size=27977 sha256=9c1c65b994be64f4ad945dde93cf4e4b43b93d9fa682f7591482c4c5b2a0806f
- `optimizer/prompts/split/sentence.md` size=4420 sha256=638e89a185f61bbe2605b3ff744200db5f26994fb38f23d7e535047b94c84607
- `optimizer/prompts/split/token_counts.md` size=1065 sha256=21770366ccd63051bf5673eec2e636f3691a15cdaa11678b6626185110a2b4f2
- `optimizer/prompts/split/token_delimited.md` size=1261 sha256=bdabda5bf97e6116e2d884fa48fae3ee9e5823bdc811a406d907575d21980fb3
- `optimizer/prompts/split/token_boundary.md` size=1732 sha256=397859e4d3a95b9f42161cd0d9788e6e1cb884a3be6e2c8309cab40224845a05
- `qwen_asr/optimizer_bridge.py` size=20156 sha256=6278c2ad5e1d19f458457c242fdebf9ef96fb6d0bc0ecea3b73377c64356f15b
- `qwen_asr/commands/split.py` size=1890 sha256=ccadc3f6fb486339a8f8d365629b9b3f2baee44d638caf73327171f5834adae5
- `qwen_asr/mfa_backend.py` size=11176 sha256=aa94176872043f998d35dfc4111cfe5fe6d05438d9b72bca00e6432bcd086ded
- `qwen_asr/mfa_experiment.py` size=11634 sha256=10ff489243fb0d3ce5afe413a0187e1dc2c27e5be55d25984a734707536297f5
- `qwen_asr/mfa_candidates.py` size=8872 sha256=b38a2b3778887f926894c9c4341c30a84db29625d3c5886ce8a5c6fea4123c6c
- `qwen_asr/mfa_environment.py` size=3703 sha256=25b1b77291ac55892bfe07b25c5a367e3b479a348d67f6ae35d12da872b912c5
- `qwen_asr/mfa_guards.py` size=7081 sha256=b8f280232c28f18e55fddfb8f5d25545e965ec7bc18eaf8ff77dad13625be7be
- `qwen_asr/mfa_lab.py` size=4110 sha256=3334bb52a996611d48a6bb8f22ca11980e42a7a1f66bea2b582b32988c349824
- `qwen_asr/mfa_report.py` size=5436 sha256=0ed4733efc6e86cbbfeb3fe9e0b4a6c12cc35b297c1d483a688b2ea720bbd6e1
- `qwen_asr/mfa_runner.py` size=8176 sha256=3193c2e2b08dfc9da3f15ebebc467d4b5fa4ffebefc48019960f221f59a99707
- `qwen_asr/mfa_words.py` size=2548 sha256=928624e8824483101031ec5ce6f9b8cb30a4be7284cc345d56474b2b831fa0a4
- `qwen_asr/mfa_writeback.py` size=6980 sha256=bbe6a77ff5e95b338f001715e68e6cd3e804836f326f24dd3516365c8461b213
- `qwen_asr/proofread_realign.py` size=27593 sha256=fac9ebb05c67ff44c2428fb8a96f8d591564fefc6df7c219c5752f3bc4a7e239
- `qwen_asr/proofread_realign_strategy.py` size=10303 sha256=c1c707da50e10d58c211b2eff099e9d3ccf88bf865c7d3916580a8c2d030f6d2
- `qwen_asr/align_timing_repair.py` size=3948 sha256=fbd136f43748726a4641d40933bc416e460af718bb2755d0739a117c9796fce7
- `qwen_asr/tuning_matrix.py` size=13715 sha256=b92cea7b0b1050f522be52a94bf75e3c40acebe6f1ec82027b816477f1316927
- `qwen_asr/baseline_snapshot.py` size=10884 sha256=10dd9a1160d2ed9785594eceb11ebbff901da104794317b9e30d6a1aecdcd75c
- `qwen_asr/align_diagnose.py` size=29662 sha256=c9c049ae7c29d0a18b398df3e7558849b0b25bef04f905fa1ae428924ebe703d
- `qwen_asr/align_split_audit.py` size=22107 sha256=a1b6662e044a1a25a4687a570bc5b66657b8f27c99be8a280cd419a90cadd917
- `qwen_asr/ass_quality_diff.py` size=15617 sha256=8d2d2e546ffca952e8332055a04eadc6c92ed6df5c356d8bdaca9d26aa668588
- `qwen_asr/cli.py` size=43369 sha256=c16002b9374df6180e88f1f2faf832a9b4c6965149f1cd03b0c36200a1489fe6
- `qwen_asr/web/commands.py` size=26416 sha256=275097334407d148aadea312dd61d84d06daa4a77fd806ebc27053df501e94a2
- `qwen_asr/web/templates/index.html` size=82101 sha256=02a79238557b4fad3d43025d075f2815c7aa20101079348413e1da0cd34103a9
