"""Run one pipeline phase of a CI run inside its workspace (subprocess entry used by the worker).

    python -m qwen_asr.ci phase --spec <workdir>/ci/spec.json --phase gpu|api [--gpu-lock PATH]

`spec.json` (written by the worker)::

    {"run_id": "...", "argv": [...arguments of `main.py run`...], "reset_from": "translate" | null}

GPU stages hold a cross-process file lock (one GPU job per lock file), CPU-only MFA prefetch runs
outside it. Progress is reported on stdout as lines ``@@KOEAXIS {json}`` (see `EVENT_PREFIX`); every
other line is ordinary log output. Exit code 0 = phase succeeded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_asr.ci.pipeline import API_PHASE, GPU_PHASE, PIPELINE_API_STAGES, PIPELINE_GPU_STAGES

EVENT_PREFIX = "@@KOEAXIS "


def emit(event: str, **payload: Any) -> None:
    payload = {"event": event, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def log(*parts: object) -> None:
    print(f"[{datetime.now():%H:%M:%S}]", *parts, flush=True)


def _gpu_lock_factory(path: str | None):
    if not path:
        return nullcontext
    from filelock import FileLock

    lock = FileLock(path)

    def acquire():
        return lock

    return acquire


def _wrap(name: str, handler, gpu_lock, needs_gpu=None):
    def run(stage_args, work_paths):
        emit("stage", stage=name, status="running")
        started = time.monotonic()
        try:
            use_lock = gpu_lock if needs_gpu is None or needs_gpu(work_paths) else nullcontext
            with use_lock():
                code = handler(stage_args, work_paths)
        except BaseException as exc:
            emit("stage", stage=name, status="failed", elapsed_s=round(time.monotonic() - started, 1),
                 error=f"{type(exc).__name__}: {exc}")
            raise
        emit("stage", stage=name, status="completed" if not code else "failed",
             elapsed_s=round(time.monotonic() - started, 1), return_code=code)
        return code

    return run


def build_args(argv: list[str], workdir: Path, phase: str):
    from qwen_asr.cli import _apply_model_cache_default, build_parser
    from qwen_asr.credentials import resolve_llm_api_key, resolve_llm_base_url

    args = build_parser().parse_args(["run", *argv])
    args.command = "run"
    args.llm_api_key = resolve_llm_api_key(getattr(args, "llm_api_key", None), args.llm_base_url) or None
    args.llm_base_url = resolve_llm_base_url(args.llm_base_url, args.llm_api_key)
    _apply_model_cache_default(args)
    args.performance_output = str(workdir / "reports" / f"pipeline_performance.{phase}.json")
    return args


def reset_from(work_paths, stage: str) -> list[str]:
    """Discard the outputs of `stage` and everything downstream so the resumed run recomputes them."""
    from qwen_asr.artifact_state import ArtifactState
    from qwen_asr.commands import stages as stage_helpers

    state = ArtifactState(work_paths)
    removed = [str(path) for path in [*state.force_delete_paths(stage), *state.downstream_existing_outputs(stage)]
               if path.exists()]
    state.delete_stage_outputs(stage)
    state.delete_downstream_outputs(stage)
    # Every retry stage is upstream of (or is) normalize: also drop normalize outputs and the export cache.
    stage_helpers._clear_normalize_outputs(work_paths)  # noqa: SLF001
    return removed


def run_phase(spec: dict[str, Any], phase: str, gpu_lock_path: str | None) -> int:
    from qwen_asr.commands import stages as stage_helpers
    from qwen_asr.defaults import DEFAULT_MODEL_CACHE_DIR
    from qwen_asr.logging_utils import setup_logging
    from qwen_asr.models import WorkPaths
    from qwen_asr.pipeline_runner import PipelineRunner
    from qwen_asr.proofread_realign import has_unrealigned_proofread_changes

    argv = [str(item) for item in spec["argv"]]
    workdir = Path(argv[argv.index("--workdir") + 1])
    for directory in ("logs", "reports"):
        (workdir / directory).mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=workdir / "logs" / f"pipeline-{phase}.log", level="INFO")
    args = build_args(argv, workdir, phase)
    paths = WorkPaths.from_workdir(workdir)
    gpu_lock = _gpu_lock_factory(gpu_lock_path)

    if phase == GPU_PHASE and spec.get("reset_from"):
        removed = reset_from(paths, str(spec["reset_from"]))
        emit("reset", stage=spec["reset_from"], removed=removed)

    handlers = {
        "prepare": stage_helpers.cmd_prepare,
        "transcribe": stage_helpers.cmd_transcribe,
        "correct": stage_helpers.cmd_correct,
        "align": stage_helpers.cmd_align,
        "split": stage_helpers.cmd_split,
        "translate": stage_helpers.cmd_translate,
        "mimo-proofread": stage_helpers.cmd_mimo_proofread,
        "proofread-realign": stage_helpers.cmd_proofread_realign,
        "text-proofread": stage_helpers.cmd_text_proofread,
        "normalize": stage_helpers.cmd_normalize,
        "export": stage_helpers.cmd_export,
    }
    keep = PIPELINE_GPU_STAGES if phase == GPU_PHASE else PIPELINE_API_STAGES
    wrapped = {}
    for name, handler in handlers.items():
        if phase == GPU_PHASE:
            wrapped[name] = _wrap(name, handler, nullcontext)  # the whole GPU phase holds the lock
        elif name == "proofread-realign":
            wrapped[name] = _wrap(name, handler, gpu_lock, has_unrealigned_proofread_changes)
        else:
            wrapped[name] = _wrap(name, handler, nullcontext)

    class PhaseRunner(PipelineRunner):
        def build_invocations(self, run_args):
            return [item for item in super().build_invocations(run_args) if item.name in keep]

    emit("phase", phase=phase, status="running")
    stage_helpers.ensure_preflight(args, paths, "run")
    runner_error = None
    context = gpu_lock() if phase == GPU_PHASE else nullcontext()
    with context:
        if phase == GPU_PHASE:
            emit("gpu_lock", status="acquired")
        try:
            code = PhaseRunner(paths, wrapped).run(args)
        except Exception as exc:  # noqa: BLE001 - reported as a failed phase
            code = 1
            runner_error = f"{type(exc).__name__}: {exc}"
            log("PHASE_ERROR", runner_error)

    if phase == GPU_PHASE:
        from qwen_asr.ci.recovery_flow import failed_alignment_count, recovery_settings, run_alignment_recovery

        if code and not failed_alignment_count(paths):
            emit("phase", phase=phase, status="failed", error=runner_error)
            return 1
        emit("stage", stage="align-recovery", status="running")
        summary = run_alignment_recovery(
            paths,
            settings=recovery_settings(args, DEFAULT_MODEL_CACHE_DIR),
            language_route=str(getattr(args, "language", None) or "auto"),
            gpu_lock=gpu_lock,
            log=log,
        )
        ok = not summary.get("failed_remaining") and paths.aligned_manifest.exists()
        emit("stage", stage="align-recovery", status="completed" if ok else "failed",
             elapsed_s=summary.get("elapsed_s"), summary=summary)
        emit("phase", phase=phase, status="completed" if ok else "failed")
        return 0 if ok else 1

    export_path = Path(argv[argv.index("--export-path") + 1]) if "--export-path" in argv else None
    ok = not code and (export_path is None or export_path.exists())
    emit("phase", phase=phase, status="completed" if ok else "failed", error=runner_error)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qwen_asr.ci phase")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--phase", choices=[GPU_PHASE, API_PHASE], required=True)
    parser.add_argument("--gpu-lock", default="")
    ns = parser.parse_args(argv)
    spec = json.loads(Path(ns.spec).read_text(encoding="utf-8"))
    return run_phase(spec, ns.phase, ns.gpu_lock or None)


if __name__ == "__main__":
    sys.exit(main())
