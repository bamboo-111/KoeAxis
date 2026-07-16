from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

OFFLINE_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def sanitize_raw_output(raw_output: Any) -> Any:
    if raw_output is None or isinstance(raw_output, (str, int, float, bool)):
        return raw_output
    if isinstance(raw_output, dict):
        return {str(key): sanitize_raw_output(value) for key, value in raw_output.items()}
    if isinstance(raw_output, (list, tuple)):
        return [sanitize_raw_output(item) for item in raw_output]
    if hasattr(raw_output, "__dict__"):
        return {str(key): sanitize_raw_output(value) for key, value in vars(raw_output).items()}
    return str(raw_output)


def cleanup_torch(full: bool = False) -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    gc.collect()
    if not torch.cuda.is_available():
        return
    if full:
        torch.cuda.synchronize()
    torch.cuda.empty_cache()


def normalize_device_map(device: str) -> str:
    return "cuda:0" if device == "cuda" else device


@contextmanager
def offline_model_loading(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    previous = {key: os.environ.get(key) for key in OFFLINE_ENV_KEYS}
    try:
        for key in OFFLINE_ENV_KEYS:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
