from __future__ import annotations

from pathlib import Path

from qwen_asr.commands.stages import load_aligned_segments
from qwen_asr.storage import read_json, serialize_manifest, write_json_atomic


def test_coarse_recovery_evidence_survives_reload_and_rewrite(tmp_path: Path) -> None:
    manifest = tmp_path / "aligned_segments.json"
    recovery = {"method": "vad-local", "actor": "radio-batch-auto", "original_error": "local density"}
    write_json_atomic(
        manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "a.wav",
                "global_start_time": 1.0,
                "global_end_time": 4.0,
                "text": "テスト",
                "tokens": [],
                "status": "completed",
                "alignment_backend": "vad-local",
                "alignment_unit": "segment",
                "alignment_state": "completed_coarse",
                "recovery": recovery,
            }
        ],
    )

    segments = load_aligned_segments(manifest)
    write_json_atomic(manifest, serialize_manifest(segments))

    assert segments[0].alignment_state == "completed_coarse"
    assert read_json(manifest)[0]["recovery"] == recovery
