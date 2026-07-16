from __future__ import annotations

import sys
from types import SimpleNamespace

from qwen_asr import model_runtime


def test_sanitize_raw_output_converts_nested_objects() -> None:
    payload = SimpleNamespace(value={1: ("text", SimpleNamespace(count=2))})

    assert model_runtime.sanitize_raw_output(payload) == {"value": {"1": ["text", {"count": 2}]}}


def test_normalize_device_map_expands_default_cuda_device() -> None:
    assert model_runtime.normalize_device_map("cuda") == "cuda:0"
    assert model_runtime.normalize_device_map("cuda:1") == "cuda:1"
    assert model_runtime.normalize_device_map("cpu") == "cpu"


def test_offline_model_loading_restores_environment(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)

    with model_runtime.offline_model_loading(True):
        assert {key: model_runtime.os.environ[key] for key in model_runtime.OFFLINE_ENV_KEYS} == {
            key: "1" for key in model_runtime.OFFLINE_ENV_KEYS
        }

    assert model_runtime.os.environ["HF_HUB_OFFLINE"] == "previous"
    assert "TRANSFORMERS_OFFLINE" not in model_runtime.os.environ
    assert "HF_DATASETS_OFFLINE" not in model_runtime.os.environ


def test_offline_model_loading_disabled_leaves_environment_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "current")

    with model_runtime.offline_model_loading(False):
        assert model_runtime.os.environ["HF_HUB_OFFLINE"] == "current"

    assert model_runtime.os.environ["HF_HUB_OFFLINE"] == "current"


def test_cleanup_torch_releases_cuda_cache(monkeypatch) -> None:
    calls: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("synchronize"),
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(model_runtime.gc, "collect", lambda: calls.append("gc"))

    model_runtime.cleanup_torch(full=True)

    assert calls == ["gc", "synchronize", "empty_cache"]
