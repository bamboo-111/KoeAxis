# Align 恢复 22 条失败执行轨迹

已覆盖 22/22 条非 OP/ED 对白失败；18/18 个规范化字符数不超过 4 的条目已标记为 `short_response`。22 条均已有 Qwen 主 Align 和 short-window 尝试证据。

| segment | 文本 | 字符 | 路由 | token | coverage | 主根因 | 置信度 |
|---|---|---|---|---|---|---|---|
| segment_000013 | うん。 | 2 | short_response | 1 | 0.028815 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000019 | うん。 | 2 | short_response | 1 | 0.024394 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000026 | お父様。 | 3 | short_response | 3 | 0.044202 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000031 | あれ、ご飯。 | 4 | short_response | 2 | 0.121468 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000036 | わけがわからないわ。 | 9 | standard | 5 | 0.196759 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000044 | くら。 | 2 | short_response | 1 | 0.047163 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000054 | 絶対取り戻さなきゃ。 | 9 | standard | 3 | 0.187185 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000067 | え。 | 1 | short_response | 1 | 0.093403 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000076 | 俺を。 | 2 | short_response | 2 | 0.0 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000086 | 何？足が息が苦しい。 | 8 | standard | 6 | 0.450583 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000093 | はい。 | 2 | short_response | 1 | 0.022883 | qwen_token_coverage_below_safety_threshold | HIGH |
| segment_000094 | 肉台、早く。 | 4 | short_response | 2 | 0.476923 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000096 | うん。うん。 | 4 | short_response | 2 | 0.005129 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000097 | はい。 | 2 | short_response | 1 | 0.0 | qwen_token_coverage_below_safety_threshold | HIGH |
| segment_000098 | うん。 | 2 | short_response | 1 | 0.010622 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000099 | 不。 | 1 | short_response | 1 | 0.010569 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000100 | うん。 | 2 | short_response | 1 | 0.0 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000101 | うん、うん。 | 4 | short_response | 2 | 0.012459 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000102 | うん。 | 2 | short_response | 1 | 0.0 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000104 | おいしい。 | 4 | short_response | 1 | 0.06958 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |
| segment_000110 | Searching where we are coming from. | 29 | standard | 6 | 0.916064 | language_route_mismatch | HIGH |
| segment_000127 | あれは。 | 3 | short_response | 2 | 0.024309 | qwen_timing_failure_then_short_window_content_guard_rejection | HIGH |

所有 transcript 在 coarse 或 verified-text retry 前仍需音频核验；参考 ASS 未作为 transcript、Qwen 或 MFA 输入。未执行的 MFA local、VAD/coarse 和语言路由均记录了明确 skip reason。
