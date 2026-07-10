from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from qwen_asr.artifact_state import ArtifactState
from qwen_asr.models import WorkPaths
from qwen_asr.progress import read_progress, write_progress
from qwen_asr.stages import stage_names_for_run


StageHandler = Callable[[argparse.Namespace, WorkPaths], int]


@dataclass(frozen=True, slots=True)
class StageInvocation:
    name: str
    args: argparse.Namespace


class PipelineRunner:
    def __init__(self, work_paths: WorkPaths, handlers: dict[str, StageHandler]) -> None:
        self.work_paths = work_paths
        self.handlers = handlers
        self.state = ArtifactState(work_paths)

    def build_invocations(self, args: argparse.Namespace) -> list[StageInvocation]:
        names = stage_names_for_run(
            with_correct=bool(args.with_correct),
            with_align=bool(args.with_align),
            with_split=bool(args.with_split),
            with_translate=bool(args.with_translate),
            with_mimo_proofread=bool(getattr(args, "with_mimo_proofread", False)),
            with_normalize=bool(args.with_normalize),
        )
        invocations: list[StageInvocation] = []
        for name in names:
            stage_args = argparse.Namespace(**vars(args))
            if name == "correct":
                stage_args.batch_num = getattr(args, "correct_batch_num", 8)
            elif name == "align":
                stage_args.model = args.align_model
                stage_args.cleanup_interval = getattr(args, "align_cleanup_interval", 4)
            elif name == "normalize":
                stage_args.source = getattr(args, "normalize_source", "auto")
            invocations.append(StageInvocation(name=name, args=stage_args))
        return invocations

    def run(self, args: argparse.Namespace) -> int:
        for invocation in self.build_invocations(args):
            if invocation.name == "prepare" and not args.force and self.state.is_complete("prepare"):
                write_progress(
                    self.work_paths,
                    stage="prepare",
                    status="skipped",
                    summary="prepare artifacts already complete",
                )
                continue
            handler = self.handlers[invocation.name]
            write_progress(
                self.work_paths,
                stage=invocation.name,
                status="running",
                summary=f"{invocation.name} started",
            )
            status = handler(invocation.args, self.work_paths)
            existing = read_progress(self.work_paths) or {}
            write_progress(
                self.work_paths,
                stage=invocation.name,
                status="completed" if status == 0 else "failed",
                done=existing.get("done"),
                total=existing.get("total"),
                current=existing.get("current", ""),
                summary=existing.get("summary") or f"{invocation.name} {'completed' if status == 0 else 'failed'}",
            )
            if status != 0:
                return status
        return 0
