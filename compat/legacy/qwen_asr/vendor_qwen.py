from __future__ import annotations

import importlib
import importlib.metadata
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator


@contextmanager
def _prefer_distribution_package(distribution_name: str, package_name: str) -> Iterator[None]:
    dist = importlib.metadata.distribution(distribution_name)
    site_root = str(Path(dist.locate_file("")).resolve())
    project_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == package_name or name.startswith(f"{package_name}.")
    }
    original_path = list(sys.path)
    try:
        for name in project_modules:
            sys.modules.pop(name, None)
        sys.path = [site_root, *[entry for entry in sys.path if str(Path(entry or ".").resolve()) != site_root]]
        yield
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        sys.modules.update(project_modules)
        sys.path = original_path


def import_qwen_asr_distribution() -> ModuleType:
    try:
        with _prefer_distribution_package("qwen-asr", "qwen_asr"):
            return importlib.import_module("qwen_asr")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("qwen-asr distribution is not installed") from exc


def get_qwen3_asr_model_class() -> type:
    return import_qwen_asr_distribution().Qwen3ASRModel


def get_qwen3_forced_aligner_class() -> type:
    return import_qwen_asr_distribution().Qwen3ForcedAligner
