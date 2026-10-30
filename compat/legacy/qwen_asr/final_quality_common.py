from __future__ import annotations

from typing import Any


def normalize_status(value: Any) -> str:
    text = str(value or "").upper()
    return text if text in {"PASS", "WARN", "FAIL"} else "WARN"


def passed(name: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS", "message": message, **extra}


def warn(name: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": "WARN", "message": message, **extra}


def fail(name: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": "FAIL", "message": message, **extra}


def skip(name: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS", "skipped": True, "message": message, **extra}


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None
