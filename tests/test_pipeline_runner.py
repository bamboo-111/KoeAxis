from __future__ import annotations

import argparse
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.pipeline_runner import PipelineRunner
from qwen_asr.stages import StageResult, StageStatus


def test_stage_order_with_correct(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []
    seen_args: dict[str, argparse.Namespace] = {}

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            seen_args[name] = args
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=True,
        with_align=True,
        with_split=True,
        with_translate=True,
        with_normalize=True,
        skip_preflight=False,
        force=True,
        correct_batch_num=8,
        align_model="align-model",
        normalize_source="translated",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "correct", "align", "split", "translate", "normalize", "export")}

    status = PipelineRunner(paths, handlers).run(args)

    assert status == 0
    assert calls == ["prepare", "transcribe", "correct", "align", "split", "translate", "normalize", "export"]
    assert seen_args["correct"].batch_num == 8
    assert seen_args["align"].cleanup_interval == 4
    assert seen_args["normalize"].source == "translated"


def test_stage_order_places_mimo_proofread_after_translate(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=True,
        with_split=True,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=True,
        skip_preflight=False,
        force=True,
        correct_batch_num=8,
        align_model="align-model",
        normalize_source="translated",
    )
    handlers = {
        name: handler(name)
        for name in ("prepare", "transcribe", "correct", "align", "split", "translate", "mimo-proofread", "normalize", "export")
    }

    assert PipelineRunner(paths, handlers).run(args) == 0
    assert calls == ["prepare", "transcribe", "align", "split", "translate", "mimo-proofread", "normalize", "export"]


def test_stage_result_defaults() -> None:
    result = StageResult(stage="prepare", status=StageStatus.SKIPPED, summary="already complete")

    assert result.return_code == 0
    assert result.status == StageStatus.SKIPPED
    assert result.summary == "already complete"
