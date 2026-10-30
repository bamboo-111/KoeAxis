from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_asr.batch_runner import BatchTask, _load_tasks, _write_batch_summary, run_batch_command
from qwen_asr.storage import read_json


def test_batch_runner_loads_multiple_media_files(tmp_path: Path) -> None:
    media_a = tmp_path / "a.mp3"
    media_a.write_bytes(b"a")
    args = argparse.Namespace(manifest=None, media_files=[str(media_a), str(media_a)])
    tasks = _load_tasks(args)
    assert len(tasks) == 2
    assert tasks[0].media == str(media_a.resolve())
    assert tasks[0].workdir != tasks[1].workdir


def test_batch_runner_deduplicates_manifest_workdirs(tmp_path: Path) -> None:
    media_a = tmp_path / "a.mp3"
    media_b = tmp_path / "b.mp3"
    media_a.write_bytes(b"a")
    media_b.write_bytes(b"b")
    manifest = tmp_path / "tasks.json"
    shared_workdir = tmp_path / "shared"
    manifest.write_text(
        json.dumps(
            [
                {"media": str(media_a), "workdir": str(shared_workdir)},
                {"media": str(media_b), "workdir": str(shared_workdir)},
            ]
        ),
        encoding="utf-8",
    )

    tasks = _load_tasks(argparse.Namespace(manifest=str(manifest), media_files=[]))

    assert len(tasks) == 2
    assert tasks[0].workdir == str(shared_workdir.resolve())
    assert tasks[1].workdir.endswith("shared-2")
    assert tasks[1].requested_workdir == str(shared_workdir.resolve())


def test_batch_runner_writes_summary_files(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    _write_batch_summary(
        summary_dir,
        [
            {"media": "a.mp3", "workdir": "w1", "status": "completed"},
            {"media": "b.mp3", "workdir": "w2", "status": "failed", "error": "boom"},
        ],
    )
    assert (summary_dir / "batch-summary.json").exists()
    assert (summary_dir / "batch-summary.txt").exists()
    payload = read_json(summary_dir / "batch-summary.json")
    assert payload["skipped"] == 0
    assert payload["tasks"][1]["failed_stage"] == ""
    assert "status\tstage\telapsed_s\tworkdir\tmedia\terror" in (summary_dir / "batch-summary.txt").read_text(encoding="utf-8")


def test_batch_runner_continues_after_prepare_failure(tmp_path: Path, monkeypatch) -> None:
    tasks = [
        BatchTask(task_id=0, media="bad.mp3", workdir=str(tmp_path / "bad")),
        BatchTask(task_id=1, media="good.mp3", workdir=str(tmp_path / "good")),
    ]
    monkeypatch.setattr("qwen_asr.batch_runner._load_tasks", lambda args: tasks)
    monkeypatch.setattr("qwen_asr.batch_runner.ensure_preflight", lambda *args, **kwargs: None)

    class FakeRunner:
        def __init__(self, work_paths, handlers):
            self.work_paths = work_paths

        def run(self, args):
            return 0

    monkeypatch.setattr("qwen_asr.batch_runner.PipelineRunner", FakeRunner)

    def prepare(args, work_paths):
        if args.media == "bad.mp3":
            raise RuntimeError("prepare boom")
        return 0

    status = run_batch_command(
        argparse.Namespace(workdir=str(tmp_path / "batch"), prepare_workers=1, fail_fast=False),
        {"prepare": prepare},
    )

    summary = read_json(tmp_path / "batch" / "summary" / "batch-summary.json")
    assert status == 1
    assert summary["failed"] == 1
    assert summary["succeeded"] == 1
    assert summary["tasks"][0]["failed_stage"] == "prepare"
    assert summary["tasks"][1]["status"] == "completed"


def test_batch_runner_fail_fast_writes_partial_summary(tmp_path: Path, monkeypatch) -> None:
    tasks = [
        BatchTask(task_id=0, media="bad.mp3", workdir=str(tmp_path / "bad")),
        BatchTask(task_id=1, media="good.mp3", workdir=str(tmp_path / "good")),
    ]
    monkeypatch.setattr("qwen_asr.batch_runner._load_tasks", lambda args: tasks)
    monkeypatch.setattr("qwen_asr.batch_runner.ensure_preflight", lambda *args, **kwargs: None)

    def prepare(args, work_paths):
        if args.media == "bad.mp3":
            raise RuntimeError("prepare boom")
        return 0

    status = run_batch_command(
        argparse.Namespace(workdir=str(tmp_path / "batch"), prepare_workers=1, fail_fast=True),
        {"prepare": prepare},
    )

    summary = read_json(tmp_path / "batch" / "summary" / "batch-summary.json")
    assert status == 1
    assert summary["failed"] == 1
    assert summary["tasks"][0]["failed_stage"] == "prepare"
    assert (tmp_path / "batch" / "logs" / "batch-run.log").exists()
