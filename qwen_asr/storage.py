from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

from qwen_asr.models import dataclass_to_dict

LOGGER = logging.getLogger(__name__)

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover
    orjson = None


T = TypeVar("T")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_atomic(path: Path, payload: Any) -> None:
    started = time.perf_counter()
    ensure_directory(path.parent)
    serialized = _serialize(payload)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            if orjson is not None:
                handle.write(orjson.dumps(serialized, option=orjson.OPT_INDENT_2))
            else:
                handle.write(json.dumps(serialized, ensure_ascii=False, indent=2).encode("utf-8"))
        try:
            temp_path.replace(path)
        except PermissionError:
            LOGGER.warning("Atomic replace failed for %s, falling back to direct write.", path)
            if orjson is not None:
                path.write_bytes(orjson.dumps(serialized, option=orjson.OPT_INDENT_2))
            else:
                path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        except PermissionError:
            LOGGER.warning("Could not remove temporary file: %s", temp_path)
    try:
        byte_count = path.stat().st_size
    except OSError:
        byte_count = 0
    _record_performance_io("write", path, byte_count, time.perf_counter() - started)


def read_json(path: Path, default: Any = None) -> Any:
    started = time.perf_counter()
    if not path.exists():
        return default
    text = path.read_bytes()
    if text.startswith(b"\xef\xbb\xbf"):
        text = text[3:]
    if orjson is not None:
        result = orjson.loads(text)
    else:
        result = json.loads(text.decode("utf-8-sig"))
    _record_performance_io("read", path, len(text), time.perf_counter() - started)
    return result


def append_jsonl(path: Path, payload: Any) -> None:
    started = time.perf_counter()
    ensure_directory(path.parent)
    serialized = _serialize(payload)
    if orjson is not None:
        line = orjson.dumps(serialized) + b"\n"
        with path.open("ab") as handle:
            handle.write(line)
        _record_performance_io("append", path, len(line), time.perf_counter() - started)
        return
    line = json.dumps(serialized, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
    _record_performance_io("append", path, len(line.encode("utf-8")), time.perf_counter() - started)


def load_jsonl(path: Path) -> list[Any]:
    started = time.perf_counter()
    if not path.exists():
        return []
    rows: list[Any] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    try:
        byte_count = path.stat().st_size
    except OSError:
        byte_count = 0
    _record_performance_io("read", path, byte_count, time.perf_counter() - started)
    return rows


def write_checkpoint_json(path: Path, payload: Any) -> None:
    write_json_atomic(path, payload)


def append_logical_warning(message: str) -> None:
    LOGGER.warning(message)


def serialize_manifest(items: Iterable[Any]) -> list[Any]:
    return [_serialize(item) for item in items]


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return dataclass_to_dict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _record_performance_io(operation: str, path: Path, byte_count: int, duration_s: float) -> None:
    from qwen_asr.performance import record_io

    record_io(operation, path, byte_count, duration_s)
