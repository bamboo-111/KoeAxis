# Source And Test Mapping

Production modules: `60`
Mapped: `60`
Unmapped: `0`

| Production module | Coverage | Tests | Status | Reason |
|---|---|---|---|---|
| `optimizer/splitter_boundaries.py` | direct | `tests/test_splitter_boundaries.py` | MAPPED | A same-name focused test module exists. |
| `optimizer/splitter_display.py` | direct | `tests/test_splitter_display.py` | MAPPED | A same-name focused test module exists. |
| `optimizer/splitter_readability.py` | direct | `tests/test_splitter_readability.py` | MAPPED | A same-name focused test module exists. |
| `optimizer/splitter_timing.py` | direct | `tests/test_splitter_timing.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/ass_quality.py` | direct | `tests/test_ass_quality.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/ass_quality_diff.py` | direct | `tests/test_ass_quality_diff.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/commands/align.py` | integration | `tests/test_pipeline_runner.py`<br>`tests/test_align_cleanup.py`<br>`tests/test_align_quality.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/commands/correct.py` | integration | `tests/test_cli_help.py`<br>`tests/test_pipeline_runner.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/commands/export.py` | integration | `tests/test_export_paths.py`<br>`tests/test_pipeline_runner.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/commands/mimo_proofread.py` | direct | `tests/test_mimo_proofread.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/commands/normalize.py` | direct | `tests/test_normalize.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/commands/preflight.py` | direct | `tests/test_preflight.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/commands/prepare.py` | integration | `tests/test_cli_help.py`<br>`tests/test_pipeline_runner.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/commands/proofread_realign.py` | direct | `tests/test_proofread_realign.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/commands/quality.py` | integration | `tests/test_final_quality.py`<br>`tests/test_pipeline_runner.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/commands/run.py` | integration | `tests/test_cli_help.py`<br>`tests/test_pipeline_runner.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/commands/split.py` | integration | `tests/test_optimizer_bridge_timing.py`<br>`tests/test_pipeline_runner.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/commands/transcribe.py` | integration | `tests/test_pipeline_runner.py`<br>`tests/test_transcribe_profile.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/commands/transcribe_profile.py` | direct | `tests/test_transcribe_profile.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/commands/translate.py` | integration | `tests/test_pipeline_runner.py`<br>`tests/test_translator_suspects.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/content_quality.py` | direct | `tests/test_content_quality.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality.py` | direct | `tests/test_final_quality.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_alignment.py` | direct | `tests/test_final_quality_alignment.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_ass.py` | direct | `tests/test_final_quality_ass.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_common.py` | direct | `tests/test_final_quality_common.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_content.py` | direct | `tests/test_final_quality_content.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_mimo.py` | direct | `tests/test_final_quality_mimo.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_postproofread.py` | direct | `tests/test_final_quality_postproofread.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_readability.py` | direct | `tests/test_final_quality_readability.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_realign.py` | direct | `tests/test_final_quality_realign.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_srt.py` | direct | `tests/test_final_quality_srt.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_stage.py` | direct | `tests/test_final_quality_stage.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/final_quality_translation.py` | direct | `tests/test_final_quality_translation.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/history_glossary_ass.py` | direct | `tests/test_history_glossary_ass.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/history_glossary_matching.py` | direct | `tests/test_history_glossary_matching.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/history_glossary_rules.py` | direct | `tests/test_history_glossary_rules.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_candidates.py` | direct | `tests/test_mfa_candidates.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_environment.py` | direct | `tests/test_mfa_environment.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_experiment.py` | direct | `tests/test_mfa_experiment.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_guards.py` | direct | `tests/test_mfa_guards.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_lab.py` | direct | `tests/test_mfa_lab.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_report.py` | direct | `tests/test_mfa_report.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_runner.py` | direct | `tests/test_mfa_runner.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_words.py` | direct | `tests/test_mfa_words.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mfa_writeback.py` | direct | `tests/test_mfa_writeback.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mimo_application.py` | direct | `tests/test_mimo_application.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mimo_audio.py` | direct | `tests/test_mimo_audio.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mimo_candidates.py` | direct | `tests/test_mimo_candidates.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mimo_checkpoints.py` | direct | `tests/test_mimo_checkpoints.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mimo_guards.py` | direct | `tests/test_mimo_guards.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mimo_inputs.py` | direct | `tests/test_mimo_inputs.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mimo_outputs.py` | direct | `tests/test_mimo_outputs.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/mimo_requests.py` | direct | `tests/test_mimo_requests.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/model_runtime.py` | direct | `tests/test_model_runtime.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/optimizer_bridge_adapter.py` | direct | `tests/test_optimizer_bridge_adapter.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/optimizer_bridge_guards.py` | direct | `tests/test_optimizer_bridge_guards.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/optimizer_bridge_stages.py` | integration | `tests/test_optimizer_bridge_timing.py`<br>`tests/test_pipeline_runner.py` | MAPPED | Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests. |
| `qwen_asr/proofread_realign.py` | direct | `tests/test_proofread_realign.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/proofread_realign_strategy.py` | direct | `tests/test_proofread_realign_strategy.py` | MAPPED | A same-name focused test module exists. |
| `qwen_asr/quality_suspects.py` | direct | `tests/test_quality_suspects.py` | MAPPED | A same-name focused test module exists. |
