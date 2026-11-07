from __future__ import annotations

import json
import logging

from qwen_asr.asr import BatchMemorySnapshot, QwenASRTranscriber
from qwen_asr.models import AudioSegment


def test_run_batch_logs_memory_probe_when_enabled(monkeypatch, caplog) -> None:
    segment = AudioSegment(
        segment_id="segment_000001",
        audio_path="segment.wav",
        source_audio_path="source.wav",
        global_start_time=0.0,
        global_end_time=1.0,
        duration=1.0,
        status="prepared",
    )

    class FakeModel:
        def transcribe(self, *, audio, language):
            del language
            assert audio == ["segment.wav", "segment.wav"]
            return [{"text": "a"}, {"text": "b"}]

    snapshots = [
        BatchMemorySnapshot(
            host_private_mb=100.0,
            host_rss_mb=90.0,
            system_available_mb=16000.0,
            cuda_allocated_mb=500.0,
            cuda_reserved_mb=640.0,
        ),
        BatchMemorySnapshot(
            host_private_mb=120.0,
            host_rss_mb=110.0,
            system_available_mb=15900.0,
            cuda_allocated_mb=520.0,
            cuda_reserved_mb=700.0,
        ),
        BatchMemorySnapshot(
            host_private_mb=105.0,
            host_rss_mb=92.0,
            system_available_mb=15950.0,
            cuda_allocated_mb=0.0,
            cuda_reserved_mb=256.0,
        ),
    ]

    monkeypatch.setattr("qwen_asr.asr.capture_batch_memory_snapshot", lambda: snapshots.pop(0))
    monkeypatch.setattr("qwen_asr.asr._cleanup_torch", lambda full=False: None)

    transcriber = QwenASRTranscriber(model_name="model", batch_size=2, profile_batches=True)
    transcriber._model = FakeModel()

    with caplog.at_level(logging.INFO):
        results = transcriber.run_batch([segment, segment])

    assert [item.text for item in results] == ["a", "b"]
    payloads = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "qwen_asr.asr" and '"event": "asr_batch_memory_probe"' in record.message
    ]
    assert [payload["phase"] for payload in payloads] == [
        "before_inference",
        "after_inference",
        "after_cleanup",
    ]
    assert all(payload["batch_size"] == 2 for payload in payloads)
    consumed = transcriber.consume_last_batch_memory_probes()
    assert [payload["phase"] for payload in consumed] == [
        "before_inference",
        "after_inference",
        "after_cleanup",
    ]
