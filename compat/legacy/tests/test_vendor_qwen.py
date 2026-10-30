from __future__ import annotations

import importlib.metadata

import pytest

from qwen_asr.vendor_qwen import import_qwen_asr_distribution


def test_import_qwen_asr_distribution_avoids_local_package_shadowing() -> None:
    try:
        importlib.metadata.distribution("qwen-asr")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("qwen-asr distribution is not installed")

    module = import_qwen_asr_distribution()

    assert hasattr(module, "Qwen3ASRModel")
    assert hasattr(module, "Qwen3ForcedAligner")
    assert "site-packages" in str(module.__file__)
