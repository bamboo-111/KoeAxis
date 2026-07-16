# Align 恢复既有能力与生产可达性矩阵

| 能力 | 结论 | 入口/默认 | 真实可达性 |
|---|---|---|---|
| qwen_primary_align | IMPLEMENTED_REACHABLE | CLI align/run/batch-run with align_backend=qwen / qwen | production entry executes the backend; 22 non-music failures remain |
| asr_short_window_fallback | IMPLEMENTED_REACHABLE | align_fallback=asr-short-window / off | content preservation guard rejects transcript rewrites and must remain strict |
| align_parameter_matrix | REJECTED_BY_EVIDENCE | explicit align timing validation and repair flags / current defaults retained | reachable but broad rerun is prohibited without a new falsifiable hypothesis |
| mfa_full_backend | REJECTED_BY_EVIDENCE | align_backend=mfa / qwen | implemented but not eligible for Goal A broad rerun |
| mfa_local_fallback | IMPLEMENTED_NOT_REACHABLE | proofread-realign experiment path only / not connected to initial Align failed-dialogue recovery | no shared first-Align recovery executor currently dispatches it |
| recovery_retry_align | IMPLEMENTED_NOT_REACHABLE | Web/API action retry_align / strategy=qwen | does not invoke any align backend |
| recovery_language_route | IMPLEMENTED_NOT_REACHABLE | Web/API action route_language / none | does not affect backend dispatch |
| vad_coarse_writeback | IMPLEMENTED_REACHABLE | localize_vad then accept_completed_coarse / pyannote_onnx_v3, threshold=0.5 | reachable, but transcript verification, multi-region and neighbor hard gates are missing |
| proofread_realign | NOT_APPLICABLE | post-edit proofread-realign stage / post-translation/post-edit only | not a replacement for initial Align failure recovery |

结论：Qwen 主 Align、short-window 和 VAD/coarse 代码路径可达；MFA local、retry_align 与 route_language 尚未由共享首次 Align 恢复执行器真实派发。P6 九变体和 MFA 全量替代已有否定证据，不重复运行。
