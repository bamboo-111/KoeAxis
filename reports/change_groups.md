# Review Change Groups

This is a review and rollback plan only. No file has been staged, committed, pushed, moved, archived, or deleted.

Review paths: `222`
Groups: `8`
Unclassified paths: `1`

## ASR, align, and runtime infrastructure

- ID: `asr_align_runtime`
- Purpose: Model runtime, transcription, alignment, artifact state, and core stage behavior.
- Rollback: Restore edited runtime files from their topic backups; newly created modules remain review-only until explicit deletion approval.
- Files: `25`

| State | Path |
|---|---|
| modified | `qwen_asr/align.py` |
| modified | `qwen_asr/artifact_state.py` |
| modified | `qwen_asr/asr.py` |
| untracked | `qwen_asr/commands/align.py` |
| untracked | `qwen_asr/commands/prepare.py` |
| untracked | `qwen_asr/commands/transcribe.py` |
| modified | `qwen_asr/corrector.py` |
| modified | `qwen_asr/defaults.py` |
| modified | `qwen_asr/history_glossary.py` |
| untracked | `qwen_asr/history_glossary_ass.py` |
| untracked | `qwen_asr/history_glossary_matching.py` |
| untracked | `qwen_asr/history_glossary_rules.py` |
| untracked | `qwen_asr/model_runtime.py` |
| modified | `qwen_asr/models.py` |
| modified | `qwen_asr/optimizer_bridge.py` |
| untracked | `qwen_asr/optimizer_bridge_adapter.py` |
| untracked | `qwen_asr/optimizer_bridge_guards.py` |
| untracked | `qwen_asr/optimizer_bridge_stages.py` |
| modified | `qwen_asr/segmenter.py` |
| modified | `qwen_asr/stages.py` |
| modified | `qwen_asr/vendor_qwen.py` |
| modified | `tests/test_align_cleanup.py` |
| modified | `tests/test_artifact_state.py` |
| untracked | `tests/test_model_runtime.py` |
| untracked | `tests/test_transcribe_profile.py` |

## Split and readability protection

- ID: `split_readability`
- Purpose: Rule split boundaries, timing allocation, display duration, and protected short-response behavior.
- Rollback: Restore splitter and prompt files from their topic backups together with the matching split tests.
- Files: `20`

| State | Path |
|---|---|
| modified | `optimizer/prompts/split/sentence.md` |
| modified | `optimizer/prompts/split/token_boundary.md` |
| modified | `optimizer/prompts/split/token_counts.md` |
| modified | `optimizer/prompts/split/token_delimited.md` |
| modified | `optimizer/split_by_llm.py` |
| modified | `optimizer/splitter.py` |
| untracked | `optimizer/splitter_boundaries.py` |
| untracked | `optimizer/splitter_display.py` |
| untracked | `optimizer/splitter_readability.py` |
| untracked | `optimizer/splitter_timing.py` |
| modified | `optimizer/token_boundary_split.py` |
| untracked | `qwen_asr/commands/split.py` |
| untracked | `tests/test_align_split_audit.py` |
| untracked | `tests/test_final_quality_readability.py` |
| untracked | `tests/test_splitter_boundaries.py` |
| modified | `tests/test_splitter_dialogue.py` |
| untracked | `tests/test_splitter_display.py` |
| untracked | `tests/test_splitter_readability.py` |
| untracked | `tests/test_splitter_timing.py` |
| modified | `tests/test_token_boundary_split.py` |

## Translation and MiMo suspects-only flow

- ID: `translation_mimo`
- Purpose: Translation guards plus MiMo candidate, request, checkpoint, application, audio, and output boundaries.
- Rollback: Restore translator/MiMo files and their paired tests from the corresponding extraction backups.
- Files: `27`

| State | Path |
|---|---|
| modified | `optimizer/prompts/translate/standard.md` |
| modified | `optimizer/translator.py` |
| untracked | `qwen_asr/commands/mimo_proofread.py` |
| untracked | `qwen_asr/commands/translate.py` |
| untracked | `qwen_asr/final_quality_mimo.py` |
| untracked | `qwen_asr/mimo_application.py` |
| untracked | `qwen_asr/mimo_audio.py` |
| untracked | `qwen_asr/mimo_candidates.py` |
| untracked | `qwen_asr/mimo_checkpoints.py` |
| untracked | `qwen_asr/mimo_guards.py` |
| untracked | `qwen_asr/mimo_inputs.py` |
| untracked | `qwen_asr/mimo_outputs.py` |
| modified | `qwen_asr/mimo_proofread.py` |
| untracked | `qwen_asr/mimo_requests.py` |
| untracked | `qwen_asr/quality_suspects.py` |
| untracked | `tests/test_final_quality_mimo.py` |
| untracked | `tests/test_final_quality_translation.py` |
| untracked | `tests/test_mimo_application.py` |
| untracked | `tests/test_mimo_audio.py` |
| untracked | `tests/test_mimo_candidates.py` |
| untracked | `tests/test_mimo_checkpoints.py` |
| untracked | `tests/test_mimo_guards.py` |
| untracked | `tests/test_mimo_inputs.py` |
| untracked | `tests/test_mimo_outputs.py` |
| untracked | `tests/test_mimo_proofread.py` |
| untracked | `tests/test_mimo_requests.py` |
| untracked | `tests/test_translator_suspects.py` |

## Proofread realignment and quality gates

- ID: `quality_realign`
- Purpose: Content, ASS, final-quality, proofread-realign, normalize, and export protection.
- Rollback: Restore quality and realignment modules with their tests; do not bypass the pre-export quality gate.
- Files: `36`

| State | Path |
|---|---|
| untracked | `qwen_asr/ass_quality.py` |
| untracked | `qwen_asr/commands/export.py` |
| untracked | `qwen_asr/commands/normalize.py` |
| untracked | `qwen_asr/commands/proofread_realign.py` |
| untracked | `qwen_asr/commands/quality.py` |
| untracked | `qwen_asr/content_quality.py` |
| untracked | `qwen_asr/final_quality.py` |
| untracked | `qwen_asr/final_quality_alignment.py` |
| untracked | `qwen_asr/final_quality_ass.py` |
| untracked | `qwen_asr/final_quality_common.py` |
| untracked | `qwen_asr/final_quality_content.py` |
| untracked | `qwen_asr/final_quality_postproofread.py` |
| untracked | `qwen_asr/final_quality_readability.py` |
| untracked | `qwen_asr/final_quality_realign.py` |
| untracked | `qwen_asr/final_quality_srt.py` |
| untracked | `qwen_asr/final_quality_stage.py` |
| untracked | `qwen_asr/final_quality_translation.py` |
| untracked | `qwen_asr/proofread_realign.py` |
| untracked | `qwen_asr/proofread_realign_strategy.py` |
| modified | `tests/test_align_quality.py` |
| untracked | `tests/test_ass_quality.py` |
| untracked | `tests/test_ass_quality_diff.py` |
| untracked | `tests/test_content_quality.py` |
| modified | `tests/test_export_paths.py` |
| untracked | `tests/test_final_quality.py` |
| untracked | `tests/test_final_quality_alignment.py` |
| untracked | `tests/test_final_quality_ass.py` |
| untracked | `tests/test_final_quality_common.py` |
| untracked | `tests/test_final_quality_content.py` |
| untracked | `tests/test_final_quality_postproofread.py` |
| untracked | `tests/test_final_quality_realign.py` |
| untracked | `tests/test_final_quality_srt.py` |
| untracked | `tests/test_final_quality_stage.py` |
| untracked | `tests/test_proofread_realign.py` |
| untracked | `tests/test_proofread_realign_strategy.py` |
| untracked | `tests/test_quality_suspects.py` |

## MFA, diagnostics, tools, and benchmarks

- ID: `mfa_diagnostics_benchmarks`
- Purpose: Experimental MFA boundaries, diagnostic tools, benchmark definitions, and reproducibility metadata.
- Rollback: Restore source/report files from backups; local environments, corpora, models, and run artifacts remain untouched.
- Files: `38`

| State | Path |
|---|---|
| untracked | `benchmarks/benchmark_results.csv` |
| untracked | `benchmarks/environment.txt` |
| untracked | `benchmarks/reproduction_commands.md` |
| untracked | `benchmarks/resume-audit-20260713-232937/commands.txt` |
| untracked | `benchmarks/resume-audit-20260713-232937/generate_reports.py` |
| untracked | `benchmarks/resume-audit-20260713-232937/raw_run_rows.json` |
| untracked | `benchmarks/resume_audit.md` |
| untracked | `qwen_asr/ass_quality_diff.py` |
| untracked | `qwen_asr/mfa_candidates.py` |
| untracked | `qwen_asr/mfa_environment.py` |
| untracked | `qwen_asr/mfa_experiment.py` |
| untracked | `qwen_asr/mfa_guards.py` |
| untracked | `qwen_asr/mfa_lab.py` |
| untracked | `qwen_asr/mfa_report.py` |
| untracked | `qwen_asr/mfa_runner.py` |
| untracked | `qwen_asr/mfa_words.py` |
| untracked | `qwen_asr/mfa_writeback.py` |
| untracked | `tests/test_align_diagnose.py` |
| untracked | `tests/test_baseline_snapshot.py` |
| untracked | `tests/test_local_state_audit.py` |
| untracked | `tests/test_mfa_backend.py` |
| untracked | `tests/test_mfa_candidates.py` |
| untracked | `tests/test_mfa_environment.py` |
| untracked | `tests/test_mfa_experiment.py` |
| untracked | `tests/test_mfa_guards.py` |
| untracked | `tests/test_mfa_lab.py` |
| untracked | `tests/test_mfa_report.py` |
| untracked | `tests/test_mfa_runner.py` |
| untracked | `tests/test_mfa_words.py` |
| untracked | `tests/test_mfa_writeback.py` |
| untracked | `tests/test_tuning_matrix.py` |
| untracked | `tools/align_diagnose.py` |
| untracked | `tools/align_split_audit.py` |
| untracked | `tools/baseline_snapshot.py` |
| untracked | `tools/diagnose_split_content_loss.py` |
| untracked | `tools/diagnose_split_readability.py` |
| untracked | `tools/mfa_full_alignment.py` |
| untracked | `tools/tuning_matrix.py` |

## CLI, WebUI, and pipeline wiring

- ID: `cli_web_wiring`
- Purpose: Argument parsing, command dispatch, PipelineRunner orchestration, Web command construction, and status presentation.
- Rollback: Restore CLI/Web/pipeline files as one wiring set, then rerun CLI-help, WebUI, and PipelineRunner tests.
- Files: `14`

| State | Path |
|---|---|
| modified | `qwen_asr/cli.py` |
| modified | `qwen_asr/commands/__init__.py` |
| untracked | `qwen_asr/commands/correct.py` |
| untracked | `qwen_asr/commands/preflight.py` |
| untracked | `qwen_asr/commands/run.py` |
| modified | `qwen_asr/commands/stages.py` |
| untracked | `qwen_asr/commands/transcribe_profile.py` |
| modified | `qwen_asr/pipeline_runner.py` |
| modified | `qwen_asr/web/commands.py` |
| modified | `qwen_asr/web/status.py` |
| modified | `qwen_asr/web/templates/index.html` |
| modified | `tests/test_cli_help.py` |
| modified | `tests/test_pipeline_runner.py` |
| modified | `tests/test_webui.py` |

## Tests, documentation, configuration, and acceptance

- ID: `tests_docs_acceptance`
- Purpose: Regression coverage, developer tooling, documentation, inventory, and acceptance evidence.
- Rollback: Restore edited documents/configuration from backups; retain generated evidence until its owning change is reviewed.
- Files: `61`

| State | Path |
|---|---|
| modified | `.gitignore` |
| modified | `README.md` |
| modified | `docs/ARCHITECTURE.md` |
| untracked | `docs/Goal模式完整任务提示词.md` |
| modified | `docs/PIPELINE.md` |
| untracked | `docs/STATUS.md` |
| untracked | `docs/字幕流程完整改造计划与验收清单.md` |
| untracked | `docs/项目文件与代码整理计划.md` |
| modified | `pyproject.toml` |
| untracked | `reports/archive_candidates.json` |
| untracked | `reports/archive_candidates.md` |
| untracked | `reports/change_groups.json` |
| untracked | `reports/change_groups.md` |
| untracked | `reports/evidence_reference_map.json` |
| untracked | `reports/evidence_reference_map.md` |
| untracked | `reports/final_acceptance.md` |
| untracked | `reports/git_file_classification.json` |
| untracked | `reports/git_file_classification.md` |
| untracked | `reports/local_state_register.json` |
| untracked | `reports/local_state_register.md` |
| untracked | `reports/p2_approval_status.json` |
| untracked | `reports/p5_0_a_execution_manifest.json` |
| untracked | `reports/p5_0_b_execution_manifest.json` |
| untracked | `reports/p5_0_compileall_environment_incident.json` |
| untracked | `reports/p5_0_compileall_environment_incident_execution.json` |
| untracked | `reports/p5_0_d_execution_manifest.json` |
| untracked | `reports/p5_0_e_execution_manifest.json` |
| untracked | `reports/p5_duplicate_helper_audit.json` |
| untracked | `reports/p5_duplicate_helper_audit.md` |
| untracked | `reports/p7_closeout.json` |
| untracked | `reports/p7_closeout.md` |
| untracked | `reports/project_inventory.json` |
| untracked | `reports/project_inventory.md` |
| untracked | `reports/rejected_feature_inventory.json` |
| untracked | `reports/rejected_feature_inventory.md` |
| untracked | `reports/source_test_mapping.json` |
| untracked | `reports/source_test_mapping.md` |
| untracked | `requirements-dev.txt` |
| untracked | `samples/radio-glossary.normalized.xlsx` |
| untracked | `samples/test.mp3` |
| untracked | `samples/test.srt` |
| untracked | `samples/无名字幕组任务分划.xlsx` |
| untracked | `samples/翻译对照.normalized-2.xlsx` |
| untracked | `samples/翻译对照.normalized.improved.xlsx` |
| untracked | `samples/翻译对照.normalized.xlsx` |
| untracked | `samples/翻译对照.xlsx` |
| untracked | `scripts/local_check.ps1` |
| untracked | `scripts/local_state_audit.py` |
| untracked | `scripts/project_inventory.py` |
| untracked | `scripts/review_scope.py` |
| modified | `tests/test_credentials.py` |
| modified | `tests/test_history_glossary.py` |
| untracked | `tests/test_history_glossary_ass.py` |
| untracked | `tests/test_history_glossary_matching.py` |
| untracked | `tests/test_history_glossary_rules.py` |
| untracked | `tests/test_optimizer_bridge_adapter.py` |
| untracked | `tests/test_optimizer_bridge_guards.py` |
| modified | `tests/test_optimizer_bridge_timing.py` |
| untracked | `tests/test_project_inventory.py` |
| untracked | `tests/test_review_scope.py` |
| untracked | `tools/README.md` |

## Unclassified review paths

- ID: `unclassified`
- Purpose: Paths that do not match an established source, test, documentation, or local-state boundary.
- Rollback: No action is authorized. Inspect origin and purpose before editing, ignoring, moving, archiving, or deleting.
- Files: `1`

| State | Path |
|---|---|
| untracked | `1/clip.json` |
