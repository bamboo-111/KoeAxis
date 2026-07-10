from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qwen_asr.storage import read_json, write_json_atomic


@dataclass(slots=True)
class SamplePayload:
    name: str
    path: Path


def test_write_json_atomic_serializes_paths_and_dataclasses(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"

    write_json_atomic(target, {"item": SamplePayload(name="audio", path=tmp_path / "audio.wav")})

    payload = read_json(target)
    assert payload == {"item": {"name": "audio", "path": str(tmp_path / "audio.wav")}}
    assert not list(target.parent.glob("*.tmp"))


def test_read_json_returns_default_for_missing_file(tmp_path: Path) -> None:
    assert read_json(tmp_path / "missing.json", default={"missing": True}) == {"missing": True}


def test_read_json_accepts_utf8_bom_without_orjson_assumption(tmp_path: Path) -> None:
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf{\"ok\": true}")

    assert read_json(path) == {"ok": True}
