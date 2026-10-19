"""Stage catalogue, presets and run-argument building.

Kept free of heavy imports: the server imports this module to validate presets and options.

A preset is a JSON file (``<presets_dir>/<id>.json`` or ``<presets_dir>/<id>/preset.json``)::

    {
      "id": "togeradi",
      "title": "刺刺广播（トゲラジ）",
      "description": "...",
      "language": "Japanese",            # ASR language; null/"" = auto detect
      "target_language": "简体中文",
      "run_args": ["--with-correct", ...],   # extra arguments for `main.py run` (open: any CLI flag)
      "files": [{"flag": "--glossary-xlsx", "path": "翻译对照.xlsx"}],  # shipped to the worker
      "env": {"QWEN_ASR_MFA_BACKGROUND_PREFETCH": "1"},              # worker-side environment
      "options": {"asr_model": "", "llm_model": "gemini-3.8-flash"} # defaults users may override
    }

File paths are relative to the preset file (or absolute). The server serves them to workers, so a
worker never needs the server's filesystem layout.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GPU_PHASE = "gpu"
API_PHASE = "api"
PHASES = (GPU_PHASE, API_PHASE)

# Display order of the stages a run reports. `align-recovery` is the CI-level recovery step
# (Qwen retry -> local MFA -> audited coarse timing) that runs after `align` in the GPU phase.
STAGES: tuple[tuple[str, str], ...] = (
    ("prepare", GPU_PHASE),
    ("transcribe", GPU_PHASE),
    ("correct", GPU_PHASE),
    ("align", GPU_PHASE),
    ("align-recovery", GPU_PHASE),
    ("split", API_PHASE),
    ("translate", API_PHASE),
    ("mimo-proofread", API_PHASE),
    ("proofread-realign", API_PHASE),
    ("text-proofread", API_PHASE),
    ("quality-gate", API_PHASE),
    ("normalize", API_PHASE),
    ("export", API_PHASE),
)
STAGE_NAMES = tuple(name for name, _ in STAGES)
STAGE_PHASE = dict(STAGES)
PIPELINE_GPU_STAGES = ("prepare", "transcribe", "correct", "align")
PIPELINE_API_STAGES = ("split", "translate", "mimo-proofread", "proofread-realign", "text-proofread", "quality-gate",
                       "normalize", "export")
# Stages that may be used as a retry starting point (their outputs and everything downstream are discarded).
RETRY_STAGES = ("prepare", "transcribe", "correct", "align", "split", "translate", "mimo-proofread",
                "proofread-realign", "text-proofread", "normalize", "export")

# Options a user may set per run; each maps to one CLI flag of `main.py run`.
OPTION_FLAGS: dict[str, str] = {
    "asr_model": "--model",
    "align_model": "--align-model",
    "llm_model": "--llm-model",
    "target_language": "--target-language",
}
PRESET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class PresetFile:
    flag: str
    path: Path

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    description: str = ""
    language: str | None = None
    target_language: str = "简体中文"
    run_args: tuple[str, ...] = ()
    files: tuple[PresetFile, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    options: dict[str, str] = field(default_factory=dict)
    source: Path | None = None

    def public(self) -> dict[str, Any]:
        """What clients and workers see (no server paths)."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "target_language": self.target_language,
            "run_args": list(self.run_args),
            "files": [{"index": index, "flag": item.flag, "name": item.name} for index, item in enumerate(self.files)],
            "env": dict(self.env),
            "options": dict(self.options),
        }


def load_preset(path: Path) -> Preset:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PresetError(f"cannot read preset {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PresetError(f"preset {path} must be a JSON object")
    preset_id = str(raw.get("id") or (path.parent.name if path.name == "preset.json" else path.stem))
    if not PRESET_ID_RE.match(preset_id):
        raise PresetError(f"invalid preset id {preset_id!r}")
    run_args = raw.get("run_args") or []
    if not isinstance(run_args, list) or not all(isinstance(item, str) for item in run_args):
        raise PresetError(f"preset {preset_id}: run_args must be a list of strings")
    files = []
    for item in raw.get("files") or []:
        if not isinstance(item, dict) or not str(item.get("flag", "")).startswith("--") or not item.get("path"):
            raise PresetError(f"preset {preset_id}: each file needs a --flag and a path")
        file_path = Path(str(item["path"]))
        if not file_path.is_absolute():
            file_path = (path.parent / file_path).resolve()
        files.append(PresetFile(flag=str(item["flag"]), path=file_path))
    options = {str(key): str(value) for key, value in (raw.get("options") or {}).items() if key in OPTION_FLAGS}
    language = raw.get("language")
    return Preset(
        id=preset_id,
        title=str(raw.get("title") or preset_id),
        description=str(raw.get("description") or ""),
        language=str(language) if language else None,
        target_language=str(raw.get("target_language") or "简体中文"),
        run_args=tuple(run_args),
        files=tuple(files),
        env={str(key): str(value) for key, value in (raw.get("env") or {}).items()},
        options=options,
        source=path,
    )


def discover_presets(directories: list[Path]) -> dict[str, Preset]:
    """Later directories override earlier ones with the same id."""
    presets: dict[str, Preset] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        candidates = sorted(directory.glob("*.json")) + sorted(directory.glob("*/preset.json"))
        for candidate in candidates:
            preset = load_preset(candidate)
            presets[preset.id] = preset
    return presets


def validate_options(options: dict[str, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (options or {}).items():
        if key not in OPTION_FLAGS and key != "language":
            raise PresetError(f"unknown option {key!r}; allowed: {sorted([*OPTION_FLAGS, 'language'])}")
        if value is None:
            continue
        text = str(value).strip()
        if len(text) > 300 or "\n" in text:
            raise PresetError(f"option {key!r} is too long or multi-line")
        result[key] = text
    return result


def build_run_argv(
    preset: dict[str, Any],
    *,
    workdir: Path,
    media: Path,
    export_path: Path,
    file_paths: list[tuple[str, Path]],
    options: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    worker_args: list[str] | None = None,
) -> list[str]:
    """Arguments for `main.py run` (without the leading `run`).

    Order matters for argparse "last wins": preset < worker defaults < run options < run extra args.
    """
    options = dict(options or {})
    preset_options = dict(preset.get("options") or {})
    language = options.pop("language", None)
    if language is None:
        language = preset.get("language") or ""
    argv = ["--workdir", str(workdir), "--media", str(media)]
    if language and language.lower() != "auto":
        argv += ["--language", language]
    argv += ["--target-language", str(preset.get("target_language") or "简体中文")]
    argv += list(preset.get("run_args") or [])
    for flag, path in file_paths:
        argv += [flag, str(path)]
    argv += list(worker_args or [])
    merged = {**{k: v for k, v in preset_options.items() if v}, **{k: v for k, v in options.items() if v}}
    for key, value in merged.items():
        argv += [OPTION_FLAGS[key], value]
    argv += list(extra_args or [])
    argv += ["--export-mode", "custom", "--export-path", str(export_path)]
    return argv


def safe_filename(name: str, default: str = "file") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(name)).strip(" .")
    return cleaned[:180] or default
