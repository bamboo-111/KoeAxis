# Reproduction Commands

Benchmark root: `E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937`

Input media was created with:

```powershell
ffmpeg -y -hide_banner -loglevel error -t 180 -i samples\test.mp3 -c copy E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937\input_180s.mp3
```

Executed stage commands:

```powershell
﻿.venv312\Scripts\python.exe main.py prepare --media benchmarks\resume-audit-20260713-232937\input_180s.mp3 --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_a_fixed_b1_r1 --skip-preflight --max-segment-seconds 60 --min-segment-seconds 2 --eager-segment-export
.venv312\Scripts\python.exe main.py transcribe --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_a_fixed_b1_r1 --skip-preflight --local-files-only --model-cache-dir .model-cache --batch-mode fixed --batch-size 1 --profile-batches --language Japanese --target-batch-audio-seconds 180 --single-long-segment-threshold 999
.venv312\Scripts\python.exe main.py align --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_a_fixed_b1_r1 --skip-preflight --local-files-only --model-cache-dir .model-cache --model Qwen/Qwen3-ForcedAligner-0.6B
.venv312\Scripts\python.exe main.py prepare --media benchmarks\resume-audit-20260713-232937\input_180s.mp3 --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_a_fixed_b1_r2 --skip-preflight --max-segment-seconds 60 --min-segment-seconds 2 --eager-segment-export
.venv312\Scripts\python.exe main.py transcribe --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_a_fixed_b1_r2 --skip-preflight --local-files-only --model-cache-dir .model-cache --batch-mode fixed --batch-size 1 --profile-batches --language Japanese --target-batch-audio-seconds 180 --single-long-segment-threshold 999
.venv312\Scripts\python.exe main.py align --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_a_fixed_b1_r2 --skip-preflight --local-files-only --model-cache-dir .model-cache --model Qwen/Qwen3-ForcedAligner-0.6B
.venv312\Scripts\python.exe main.py prepare --media benchmarks\resume-audit-20260713-232937\input_180s.mp3 --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_b_fixed_b2_r1 --skip-preflight --max-segment-seconds 60 --min-segment-seconds 2 --eager-segment-export
.venv312\Scripts\python.exe main.py transcribe --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_b_fixed_b2_r1 --skip-preflight --local-files-only --model-cache-dir .model-cache --batch-mode fixed --batch-size 2 --profile-batches --language Japanese --target-batch-audio-seconds 180 --single-long-segment-threshold 999
.venv312\Scripts\python.exe main.py align --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_b_fixed_b2_r1 --skip-preflight --local-files-only --model-cache-dir .model-cache --model Qwen/Qwen3-ForcedAligner-0.6B
.venv312\Scripts\python.exe main.py prepare --media benchmarks\resume-audit-20260713-232937\input_180s.mp3 --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_b_fixed_b2_r2 --skip-preflight --max-segment-seconds 60 --min-segment-seconds 2 --eager-segment-export
.venv312\Scripts\python.exe main.py transcribe --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_b_fixed_b2_r2 --skip-preflight --local-files-only --model-cache-dir .model-cache --batch-mode fixed --batch-size 2 --profile-batches --language Japanese --target-batch-audio-seconds 180 --single-long-segment-threshold 999
.venv312\Scripts\python.exe main.py align --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_b_fixed_b2_r2 --skip-preflight --local-files-only --model-cache-dir .model-cache --model Qwen/Qwen3-ForcedAligner-0.6B
.venv312\Scripts\python.exe main.py prepare --media benchmarks\resume-audit-20260713-232937\input_180s.mp3 --workdir benchmarks\resume-audit-20260713-232937\runs\current_adaptive_auto_r1 --skip-preflight --max-segment-seconds 60 --min-segment-seconds 2 --eager-segment-export
.venv312\Scripts\python.exe main.py transcribe --workdir benchmarks\resume-audit-20260713-232937\runs\current_adaptive_auto_r1 --skip-preflight --local-files-only --model-cache-dir .model-cache --batch-mode adaptive --batch-size 5 --profile-batches --language Japanese
.venv312\Scripts\python.exe main.py align --workdir benchmarks\resume-audit-20260713-232937\runs\current_adaptive_auto_r1 --skip-preflight --local-files-only --model-cache-dir .model-cache --model Qwen/Qwen3-ForcedAligner-0.6B
.venv312\Scripts\python.exe main.py prepare --media benchmarks\resume-audit-20260713-232937\input_180s.mp3 --workdir benchmarks\resume-audit-20260713-232937\runs\current_adaptive_auto_r2 --skip-preflight --max-segment-seconds 60 --min-segment-seconds 2 --eager-segment-export
.venv312\Scripts\python.exe main.py transcribe --workdir benchmarks\resume-audit-20260713-232937\runs\current_adaptive_auto_r2 --skip-preflight --local-files-only --model-cache-dir .model-cache --batch-mode adaptive --batch-size 5 --profile-batches --language Japanese
.venv312\Scripts\python.exe main.py align --workdir benchmarks\resume-audit-20260713-232937\runs\current_adaptive_auto_r2 --skip-preflight --local-files-only --model-cache-dir .model-cache --model Qwen/Qwen3-ForcedAligner-0.6B
```

Resume test:

```powershell
.venv312\Scripts\python.exe main.py transcribe --workdir benchmarks\resume-audit-20260713-232937\runs\baseline_a_fixed_b1_r1 --skip-preflight --resume --local-files-only --model-cache-dir .model-cache --batch-mode fixed --batch-size 1 --profile-batches
# exit_code=0, elapsed_s=0.787
# log: benchmarks\resume-audit-20260713-232937\runs\baseline_a_fixed_b1_r1\resume_test.log
```

Report-input dry-run (reads the existing benchmark manifests and profiles without rewriting summaries):

```powershell
.venv312\Scripts\python.exe benchmarks\resume-audit-20260713-232937\generate_reports.py --check
```
