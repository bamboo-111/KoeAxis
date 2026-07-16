# P5 Duplicate Helper Audit

## Scope And Method

- Scope: production Python under `qwen_asr/` and `optimizer/`; tests, `tools/`, local environments, workspaces, reports, and backups are excluded.
- Method: parse with `utf-8-sig`, normalize only function names, and compare the complete AST including arguments, defaults, annotations, and body.
- Functions scanned: `962`.
- Structurally identical groups: `12` groups / `24` functions.

## Consolidated

AST-identical final-quality status normalization, PASS/WARN/FAIL/SKIP result constructors, and optional numeric conversion helpers were consolidated in `qwen_asr/final_quality_common.py`. Compatibility aliases and focused tests preserve the former module-level contracts.

## Deliberately Retained

| Members | Decision | Reason |
|---|---|---|
| `align.py close`; `asr.py close` | KEEP_SEPARATE | Different runtime lifecycle owners. |
| `ass_quality.py _zh_status`; `ass_quality_diff.py _zh_status` | KEEP_SEPARATE | Independent production report cores; sharing a trivial label helper would add coupling. |
| `ass_quality.py _srt_ms`; `content_quality.py _srt_ms` | KEEP_SEPARATE | Private helpers for different report schemas. |
| `final_quality.py _manifest_key_sort`; `mimo_inputs.py manifest_key_sort` | KEEP_COMPATIBILITY_WRAPPER | Final-quality compatibility surface and MiMo input ownership differ. |
| `final_quality_common.py float_or_none`; `mfa_guards.py float_or_none` | KEEP_SEPARATE | MFA local fallback remains independently usable. |
| `final_quality_common.py int_or_none`; `optimizer_bridge_guards.py segment_time_ms` | KEEP_SEPARATE | Structurally equal, semantically different contracts. |
| `history_glossary_rules.py has_cjk`; `subtitle.py _contains_cjk` | KEEP_SEPARATE | Glossary filtering and subtitle parsing have separate ownership. |
| `mfa_candidates.py _int_or_none`; `mfa_guards.py int_or_none` | KEEP_SEPARATE | Candidate collection and writeback guards are isolated MFA boundaries. |
| `mfa_guards.py normalize_local_match_text`; `proofread_realign_strategy.py normalize_mfa_content` | KEEP_SEPARATE | Same current body, different policy boundaries. |
| `optimizer_bridge_adapter.py _normalize_content`; `commands/align.py _normalize_alignment_content` | KEEP_SEPARATE | Payload adaptation and command comparison are separate contracts. |
| `commands/export.py _load_project_metadata`; `commands/stages.py _load_project_metadata` | KEEP_COMPATIBILITY_WRAPPER | Stage facade compatibility and standalone export imports must both remain stable. |
| `commands/stages.py _read_completed_list_count`; `web/status.py _read_completed_list_count` | KEEP_SEPARATE | CLI and Web presentation boundaries should not depend on each other. |

## Acceptance

- Only helpers with the same semantics, ownership, and tested contract were merged.
- No production module imports from `tools/`.
- Trivial duplication was retained where consolidation would increase coupling or weaken compatibility.
- Result: `PASS`.
