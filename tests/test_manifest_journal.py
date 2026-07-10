from __future__ import annotations

from pathlib import Path

from qwen_asr.commands.stages import _recover_manifest_from_checkpoint_and_events
from qwen_asr.storage import append_jsonl, write_checkpoint_json


def test_manifest_recovery_from_checkpoint_and_events(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    events = tmp_path / "events.jsonl"
    write_checkpoint_json(
        checkpoint,
        [
            {"segment_id": "segment_000001", "text": "hello", "status": "completed"},
        ],
    )
    append_jsonl(
        events,
        {
            "type": "transcript",
            "segment_id": "segment_000002",
            "payload": {"segment_id": "segment_000002", "text": "world", "status": "completed"},
        },
    )
    manifest = _recover_manifest_from_checkpoint_and_events(checkpoint, events)
    assert len(manifest) == 2
    assert {item["segment_id"] for item in manifest} == {"segment_000001", "segment_000002"}
