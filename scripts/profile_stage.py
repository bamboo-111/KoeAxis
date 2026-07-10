from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_asr.cli import build_parser
from qwen_asr.logging_utils import setup_logging
from qwen_asr.models import WorkPaths
from qwen_asr.progress import read_progress, write_progress


@dataclass(slots=True)
class Event:
    kind: str
    name: str
    elapsed_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Profiler:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def wrap(
        self,
        kind: str,
        name: str,
        func: Callable[..., Any],
        metadata_builder: Callable[[tuple[Any, ...], dict[str, Any], Any], dict[str, Any]] | None = None,
    ) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - started
            metadata = metadata_builder(args, kwargs, result) if metadata_builder else {}
            self.events.append(
                Event(
                    kind=kind,
                    name=name,
                    elapsed_s=round(elapsed, 6),
                    metadata=metadata,
                )
            )
            return result

        return wrapped

    def summary(self) -> dict[str, Any]:
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for event in self.events:
            key = f"{event.kind}:{event.name}"
            totals[key] += event.elapsed_s
            counts[key] += 1
        aggregate = []
        for key in sorted(totals):
            aggregate.append(
                {
                    "key": key,
                    "calls": counts[key],
                    "total_s": round(totals[key], 6),
                    "avg_s": round(totals[key] / counts[key], 6),
                }
            )
        return {
            "aggregate": aggregate,
            "events": [
                {
                    "kind": event.kind,
                    "name": event.name,
                    "elapsed_s": event.elapsed_s,
                    "metadata": event.metadata,
                }
                for event in self.events
            ],
        }


def _segment_metadata(args: tuple[Any, ...], _: dict[str, Any], result: Any) -> dict[str, Any]:
    segment = args[1]
    metadata = {"segment_id": getattr(segment, "segment_id", "")}
    if result is not None:
        metadata["status"] = getattr(result, "status", "")
    return metadata


def _path_metadata(args: tuple[Any, ...], kwargs: dict[str, Any], _: Any) -> dict[str, Any]:
    path = args[0] if args else kwargs.get("path")
    return {"path": str(path) if path is not None else ""}


def _write_text_metadata(args: tuple[Any, ...], _: dict[str, Any], transcripts: Any) -> dict[str, Any]:
    path = args[0]
    items = args[1]
    completed = 0
    if isinstance(items, list):
        completed = sum(1 for item in items if getattr(item, "status", "") == "completed")
    return {"path": str(path), "items": len(items), "completed": completed}


def _export_segment_metadata(args: tuple[Any, ...], _: dict[str, Any], result: Any) -> dict[str, Any]:
    segment = args[1]
    return {
        "segment_id": getattr(segment, "segment_id", ""),
        "duration": getattr(segment, "duration", None),
        "path": str(result) if result is not None else "",
    }


def _extract_audio_metadata(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> dict[str, Any]:
    media_path = args[0] if args else kwargs.get("media_path")
    output_path = args[1] if len(args) > 1 else kwargs.get("output_path")
    return {
        "media_path": str(media_path) if media_path is not None else "",
        "output_path": str(output_path) if output_path is not None else "",
        "result": str(result) if result is not None else "",
    }


def _speech_metadata(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    count = len(result) if isinstance(result, list) else 0
    return {"speech_regions": count}


def _build_parser_with_profile() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile one qwen_asr stage without changing outputs.")
    parser.add_argument("stage", choices=["prepare", "transcribe", "align"])
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("stage_args", nargs=argparse.REMAINDER)
    return parser


def _run_stage_with_progress(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    write_progress(work_paths, stage=args.command, status="running", current="", summary=f"{args.command} started")
    status = args.func(args, work_paths)
    existing = read_progress(work_paths) or {}
    write_progress(
        work_paths,
        stage=args.command,
        status="completed" if status == 0 else "failed",
        done=existing.get("done"),
        total=existing.get("total"),
        current=existing.get("current", ""),
        summary=existing.get("summary") or f"{args.command} {'completed' if status == 0 else 'failed'}",
    )
    return status


def _install_prepare_patches(stack: ExitStack, profiler: Profiler) -> None:
    import qwen_asr.audio as audio_mod
    import qwen_asr.commands.stages as stages_mod
    import qwen_asr.vad as vad_mod

    stack.enter_context(
        patch.object(
            stages_mod,
            "extract_audio",
            profiler.wrap("stage", "extract_audio", stages_mod.extract_audio, _extract_audio_metadata),
        )
    )
    stack.enter_context(
        patch.object(
            vad_mod.SileroVADAdapter,
            "detect",
            profiler.wrap("stage", "vad_detect", vad_mod.SileroVADAdapter.detect, _speech_metadata),
        )
    )
    stack.enter_context(
        patch.object(
            stages_mod,
            "export_segment_audio",
            profiler.wrap("segment", "export_segment_audio", audio_mod.export_segment_audio, _export_segment_metadata),
        )
    )
    stack.enter_context(
        patch.object(
            stages_mod,
            "write_json_atomic",
            profiler.wrap("io", "write_json_atomic", stages_mod.write_json_atomic, _path_metadata),
        )
    )


def _install_transcribe_patches(stack: ExitStack, profiler: Profiler) -> None:
    import qwen_asr.asr as asr_mod
    import qwen_asr.commands.stages as stages_mod

    stack.enter_context(
        patch.object(
            asr_mod.QwenASRTranscriber,
            "load",
            profiler.wrap("model", "load", asr_mod.QwenASRTranscriber.load),
        )
    )
    stack.enter_context(
        patch.object(
            asr_mod.QwenASRTranscriber,
            "run_segment",
            profiler.wrap("segment", "run_segment", asr_mod.QwenASRTranscriber.run_segment, _segment_metadata),
        )
    )
    stack.enter_context(
        patch.object(
            asr_mod,
            "_cleanup_torch",
            profiler.wrap("runtime", "cleanup_torch", asr_mod._cleanup_torch),
        )
    )
    stack.enter_context(
        patch.object(
            stages_mod,
            "write_json_atomic",
            profiler.wrap("io", "write_json_atomic", stages_mod.write_json_atomic, _path_metadata),
        )
    )
    stack.enter_context(
        patch.object(
            stages_mod,
            "write_transcript_text",
            profiler.wrap("io", "write_transcript_text", stages_mod.write_transcript_text, _write_text_metadata),
        )
    )


def _install_align_patches(stack: ExitStack, profiler: Profiler) -> None:
    import qwen_asr.align as align_mod
    import qwen_asr.commands.stages as stages_mod

    stack.enter_context(
        patch.object(
            align_mod.QwenForcedAligner,
            "load",
            profiler.wrap("model", "load", align_mod.QwenForcedAligner.load),
        )
    )
    stack.enter_context(
        patch.object(
            align_mod.QwenForcedAligner,
            "run_segment",
            profiler.wrap("segment", "run_segment", align_mod.QwenForcedAligner.run_segment, _segment_metadata),
        )
    )
    stack.enter_context(
        patch.object(
            align_mod,
            "_cleanup_torch",
            profiler.wrap("runtime", "cleanup_torch", align_mod._cleanup_torch),
        )
    )
    stack.enter_context(
        patch.object(
            stages_mod,
            "write_json_atomic",
            profiler.wrap("io", "write_json_atomic", stages_mod.write_json_atomic, _path_metadata),
        )
    )


def main() -> int:
    top_parser = _build_parser_with_profile()
    top_args = top_parser.parse_args()

    parser = build_parser()
    stage_tokens = [top_args.stage, *top_args.stage_args]
    args = parser.parse_args(stage_tokens)
    work_paths = WorkPaths.from_workdir(Path(args.workdir))
    setup_logging(log_file=work_paths.logs_dir / f"{args.command}.log", level=args.log_level)

    profiler = Profiler()
    installers = {
        "prepare": _install_prepare_patches,
        "transcribe": _install_transcribe_patches,
        "align": _install_align_patches,
    }
    started = time.perf_counter()
    with ExitStack() as stack:
        installers[top_args.stage](stack, profiler)
        exit_code = _run_stage_with_progress(args, work_paths)
    elapsed = time.perf_counter() - started

    report = {
        "stage": top_args.stage,
        "workdir": str(work_paths.workdir),
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 6),
        **profiler.summary(),
    }
    output_path = Path(top_args.output) if top_args.output else work_paths.workdir / f"{top_args.stage}.profile.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
