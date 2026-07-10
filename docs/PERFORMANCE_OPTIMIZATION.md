# Performance Optimization Notes

This document summarizes the current ASR-side performance work, the measured tradeoffs, and the adaptive batching behavior now implemented in the project.

## Scope

The focus of this round was the `transcribe` stage:

- measure end-to-end stage time and hardware usage
- identify whether the bottleneck was Python orchestration, I/O, or model inference
- validate larger ASR batch sizes on the local RTX 4070 SUPER 12GB machine
- add an adaptive fallback strategy for OOM-like failures

The changes in this round do not attempt to keep bit-identical ASR text output relative to `batch_size=1`. Small transcription drift is accepted by current project direction.

## Key Findings

Baseline measurements on cached sample audio showed:

- `prepare` is mainly CPU / disk work and is not the first optimization target
- `transcribe` is the primary bottleneck
- `align` is already much faster than `transcribe`; its main overhead is model load cost

Function-level profiling of `transcribe` with `batch_size=1` showed:

- model load was significant but secondary
- JSON / transcript file writes were negligible
- the dominant cost was repeated single-segment inference

## Batch Sweep

Test machine:

- GPU: RTX 4070 SUPER 12GB
- Python: `.venv312`

Measured `transcribe` results on the same sample workload:

| batch_size | elapsed_s | peak_vram_mb | avg_vram_mb | peak_gpu_util | avg_gpu_util |
|---|---:|---:|---:|---:|---:|
| 1 | 215.45 | 6292 | 5901.49 | 100 | 49.00 |
| 2 | 128.28 | 7659 | 6766.30 | 94 | 54.00 |
| 3 | 98.08 | 8469 | 7063.96 | 90 | 55.53 |
| 4 | 78.75 | 8929 | 7407.18 | 100 | 66.36 |
| 5 | 70.97 | 9365 | 7659.28 | 100 | 66.64 |
| 6 | 73.49 | 9525 | 7354.73 | 100 | 59.62 |
| 7 | 67.70 | 10381 | 7979.26 | 92 | 64.21 |
| 8 | 70.35 | 11187 | 8598.91 | 100 | 66.80 |

Practical reading:

- `4` is a conservative speed / VRAM improvement over legacy `1`
- `5` is a strong default on 12GB-class cards
- `7` was fastest in this sample, but VRAM headroom is much tighter
- above this range, the performance curve is no longer monotonic, so bigger is not always better

## Output Drift

Comparing `batch_size=4` against `batch_size=1` on the measured sample:

- 9 / 20 segments had text differences
- overall estimated changed-character rate was about `0.24%`
- most differences were very small

This drift comes from upstream `qwen-asr` batch inference behavior, not from manifest ordering or local merge logic. In the upstream transformers path, batched requests are tokenized with `padding=True` and decoded together, so the batch path is not guaranteed to match the single-item path exactly.

## Adaptive Batching Strategy

Current implementation behavior:

1. Start `transcribe` with the requested batch size.
2. If a batch succeeds, continue normally.
3. If a batch throws an OOM-like error:
   - reduce batch size by 1
   - retry the same batch from the same first segment
   - keep using the reduced batch size for subsequent batches
4. If batch size reaches 1, normal single-segment execution remains available.
5. Non-OOM batch failures still fall back to per-segment transcription rather than being misclassified as OOM.

OOM-like detection currently checks exception chains for messages such as:

- `out of memory`
- `cuda out of memory`
- `cublas_status_alloc_failed`

## Current Defaults

The project now defaults to:

- CLI `transcribe --batch-size 5`
- CLI `run --batch-size 5`
- WebUI transcribe / run command construction also defaults to `5`

Users can still explicitly set:

- `--batch-size 1` for stricter compatibility with the old behavior
- lower values for smaller GPUs
- higher values for experimentation on larger cards

## OOM Pressure Test Status

Real OOM fallback was stress-tested after implementation.

Observed results on the current sample set:

- `batch_size=12` completed successfully
- `batch_size=20` completed successfully
- a synthetic 40-segment stress case also completed on this machine, so a clean reproducible CUDA OOM was not triggered from this workload

One interrupted synthetic pressure run temporarily showed very large host-side memory growth during execution, but after cleanup there were no persistent background Python processes, and the generated workdir contained a completed 40-segment transcript manifest.

Conclusion:

- the adaptive fallback path is implemented and covered by unit test
- on the current local workload, the model / driver stack is more tolerant than expected, so real OOM was not cleanly reproduced

## Why Not Keep Increasing Batch Size

Based on the current investigation, the next useful step is not to keep pushing `batch_size` upward blindly.

Reason:

- larger batch size changes more than just GPU occupancy
- upstream batch inference uses padded multi-sample processing, so memory pressure is shaped by:
  - number of samples in the batch
  - audio duration distribution inside the batch
  - padding to the longest sample
  - host-side preprocessing and tensor staging before generation
- in at least one synthetic pressure run, Python process memory grew very large even though a clean CUDA OOM was not observed

This means a future failure may come from:

- VRAM pressure
- CPU RAM pressure
- pagefile / host memory pressure
- a combination of the above

So simply increasing batch size further would give low-quality information. It would tell us that "something became unstable", but not where the real bottleneck is.

## Recommended Memory Profiling Plan

Instead of another blind batch sweep, the recommended next step is batch-internal memory profiling.

The goal is to identify whether memory growth is dominated by:

- audio normalization / loading
- processor padding and feature packing
- CPU-to-GPU tensor transfer
- model generation itself
- delayed cleanup

### Suggested Probe Points

For one batch run, record memory at these points:

1. Before batch preparation
2. After audio normalization
3. After `processor(..., padding=True)`
4. After `inputs.to(device)` / tensor transfer
5. After `model.generate(...)`
6. After batch cleanup

### Suggested Metrics

At each probe point, record:

- GPU VRAM used
- GPU utilization
- Python process private memory / RSS
- system available RAM
- optional pagefile growth if easy to capture on Windows

### Why This Matters

This profiling would answer the questions that batch-size sweep cannot:

- whether `batch_size=10+` is really VRAM-limited on this workload
- whether the dominant growth is actually on the host side
- whether `padding=True` is the main memory amplifier
- whether the model generate step or the preprocessing step is the real risk

Only after that data is available should the project consider:

- raising the default batch size again
- introducing VRAM-aware dynamic batch targeting
- changing preprocessing strategy
- adding stronger host-memory guards

## Tests Added / Used

Relevant validation performed:

- `py_compile` on changed modules
- existing CLI / pipeline / vendor / WebUI tests
- new unit test covering adaptive OOM batch reduction

New regression coverage:

- `tests/test_transcribe_batching.py`

## Recommended Next Steps

1. Add low-overhead structured stage metrics to production logs, especially:
   - model load time
   - per-batch inference time
   - effective batch size after fallback
2. Consider exposing `asr_batch_size` in the WebUI payload rather than only via backend default.
3. Revisit `align` cleanup frequency; it is still more conservative than necessary for throughput.
4. If future workloads include much longer segments than the current sample, re-run the batch sweep because the optimal size may move downward.
