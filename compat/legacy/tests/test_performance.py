from __future__ import annotations

import argparse
import time
import types
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.performance import PipelinePerformanceRecorder, measure_operation, record_network_request
from qwen_asr.pipeline_runner import PipelineRunner
from qwen_asr.storage import append_jsonl, read_json, write_json_atomic


def test_stage_span_records_resources_and_json_io(tmp_path: Path) -> None:
    output = tmp_path / "performance.json"
    data = tmp_path / "manifest_segments.json"
    events = tmp_path / "aligned.events.jsonl"
    recorder = PipelinePerformanceRecorder(output, enabled=True, sample_interval_s=0.05)

    with recorder.stage("transcribe") as span:
        write_json_atomic(data, [{"segment_id": "s1"}])
        assert read_json(data)[0]["segment_id"] == "s1"
        append_jsonl(events, {"segment_id": "s1", "status": "completed"})
        time.sleep(0.06)
        span.set_result("completed", "ok")

    report = read_json(output)
    stage = report["stages"][0]
    assert stage["stage"] == "transcribe"
    assert stage["status"] == "completed"
    assert stage["wall_time_s"] >= 0.05
    assert stage["peak_host_ram_mb"] > 0
    assert stage["instrumented_io"]["manifest_write"]["count"] == 1
    assert stage["instrumented_io"]["manifest_read"]["count"] == 1
    assert stage["instrumented_io"]["event_append"]["count"] == 1
    breakdown = stage["timing_breakdown"]
    assert breakdown["stage_wall_s"] >= 0.05
    assert breakdown["instrumented_io_s"] > 0
    assert 0 <= breakdown["remaining_ratio"] <= 1


def test_pipeline_runner_records_skipped_and_completed_stages(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.parent.mkdir(parents=True, exist_ok=True)
    paths.audio_path.write_bytes(b"audio")
    write_json_atomic(paths.segments_manifest, [{"segment_id": "s1"}])

    def transcribe(args: argparse.Namespace, work_paths: WorkPaths) -> int:
        del args
        write_json_atomic(
            work_paths.transcript_manifest,
            [{"segment_id": "s1", "global_start_time": 0.0, "global_end_time": 1.0, "text": "ok", "status": "completed"}],
        )
        return 1

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=False,
        with_mimo_proofread=False,
        with_normalize=False,
        force=False,
        align_model="align-model",
        performance_profile=True,
        performance_output="",
        performance_sample_interval=0.05,
    )

    assert PipelineRunner(paths, {"transcribe": transcribe}).run(args) == 1
    report = read_json(tmp_path / "reports" / "pipeline_performance.json")
    assert [(item["stage"], item["status"]) for item in report["stages"]] == [
        ("prepare", "skipped"),
        ("transcribe", "failed"),
    ]


def test_pipeline_runner_preserves_handler_resume_skip(tmp_path: Path) -> None:
    from qwen_asr.progress import write_progress

    paths = WorkPaths.from_workdir(tmp_path)

    def prepare(args: argparse.Namespace, work_paths: WorkPaths) -> int:
        del args
        write_progress(work_paths, stage="prepare", status="skipped", done=1, total=1, summary="resume hit")
        return 0

    def transcribe(args: argparse.Namespace, work_paths: WorkPaths) -> int:
        del args, work_paths
        return 1

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=False,
        with_mimo_proofread=False,
        with_normalize=False,
        force=True,
        align_model="align-model",
        performance_profile=True,
        performance_output="",
        performance_sample_interval=0.05,
    )

    assert PipelineRunner(paths, {"prepare": prepare, "transcribe": transcribe}).run(args) == 1
    report = read_json(tmp_path / "reports" / "pipeline_performance.json")
    assert report["stages"][0]["status"] == "skipped"
    assert report["stages"][1]["status"] == "failed"


def test_stage_span_records_operations_and_threaded_network_requests(tmp_path: Path) -> None:
    import threading

    output = tmp_path / "performance.json"
    recorder = PipelinePerformanceRecorder(output, enabled=True, sample_interval_s=0.05)

    with recorder.stage("translate") as span:
        with measure_operation("model_load", component="test-model"):
            time.sleep(0.01)

        worker = threading.Thread(
            target=record_network_request,
            kwargs={
                "provider": "test",
                "model": "model",
                "duration_s": 0.2,
                "success": True,
                "attempts": 2,
                "timeout": False,
                "input_tokens": 10,
                "output_tokens": 5,
                "reasoning_tokens": 0,
                "thinking_disabled": True,
                "no_think_sent": True,
                "extra_body_sent": True,
            },
        )
        worker.start()
        worker.join()
        span.set_result("completed", "ok")

    stage = read_json(output)["stages"][0]
    assert stage["operations"][0]["kind"] == "model_load"
    request = stage["network_requests"][0]
    assert request["retry_count"] == 1
    assert request["reasoning_tokens"] == 0
    assert request["no_think_sent"] is True
    assert stage["network_summary"]["latency_s"]["p95"] == 0.2
    assert stage["network_summary"]["token_totals"]["input_tokens"] == 10
    assert request["cached_tokens"] is None
    assert stage["network_summary"]["token_unavailable_count"]["cached_tokens"] == 1
    assert stage["timing_breakdown"]["network_request_latency_sum_s"] == 0.2
    assert stage["timing_breakdown"]["network_parallelism_ratio"] > 0


def test_optimizer_llm_request_is_recorded_without_credentials(monkeypatch, tmp_path: Path) -> None:
    from optimizer.llm_client import call_llm

    usage = types.SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=7,
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=0),
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=8),
    )
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"ok": true}'))],
        usage=usage,
    )
    completions = types.SimpleNamespace(create=lambda **kwargs: response)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    monkeypatch.setattr("optimizer.llm_client._create_client", lambda *args, **kwargs: client)
    output = tmp_path / "performance.json"
    recorder = PipelinePerformanceRecorder(output, enabled=True, sample_interval_s=0.05)

    with recorder.stage("translate") as span:
        result = call_llm(
            messages=[{"role": "system", "content": "reply"}],
            model="test-model",
            temperature=0,
            base_url="https://example.invalid/v1",
            api_key="must-not-appear",
            disable_thinking=True,
            llm_extra_body={"thinking": {"type": "disabled"}},
        )
        assert result is response
        span.set_result("completed", "ok")

    raw = output.read_text(encoding="utf-8")
    assert "must-not-appear" not in raw
    request = read_json(output)["stages"][0]["network_requests"][0]
    assert request["provider"] == "example.invalid"
    assert request["input_tokens"] == 12
    assert request["reasoning_tokens"] == 0
    assert request["no_think_sent"] is True
    assert request["cached_tokens"] == 8
    assert read_json(output)["stages"][0]["network_summary"]["token_totals"]["cached_tokens"] == 8


def test_workload_snapshot_counts_manifests(tmp_path: Path) -> None:
    from qwen_asr.performance import collect_workload_snapshot

    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.segments_manifest,
        [
            {"segment_id": "s1", "duration": 1.25},
            {"segment_id": "s2", "global_start_time": 2.0, "global_end_time": 3.5},
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {"segment_id": "s1", "status": "completed", "tokens": [{"text": "a", "start_time": 0.0, "end_time": 1.0}]},
            {"segment_id": "s2", "status": "failed", "tokens": []},
        ],
    )
    write_json_atomic(paths.split_manifest, {"1": {}, "2": {}, "3": {}})

    snapshot = collect_workload_snapshot(paths)

    assert snapshot["audio_seconds"] == 2.75
    assert snapshot["segment_count"] == 2
    assert snapshot["aligned_count"] == 2
    assert snapshot["alignment_status"] == {"completed_exact": 1, "failed": 1}
    assert snapshot["split_cue_count"] == 3
