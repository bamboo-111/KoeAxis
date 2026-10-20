"""Program background prompts shared by the text LLM stages (correct / translate / text-proofread).

A program context is plain text (Markdown is fine) kept in its own file, e.g.
``samples/program-context/togeradi.md``: what the show is, who the hosts are, recurring corners and
names that ASR tends to mishear. Several files may be given (``--program-context`` repeated); they are
joined in order, followed by the episode title (``--episode-title``: ``auto`` = media file name, added only
when context files are given; ``off``; or literal text), and inserted into each stage's main prompt as one
``<program_context>`` block. Without files and title nothing is added, so prompts stay byte-identical.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json

LOGGER = logging.getLogger(__name__)

# Long enough for a show description plus a member / corner list; longer files are cut with a warning.
MAX_CONTEXT_CHARS = 8000
EPISODE_TITLE_AUTO = "auto"
EPISODE_TITLE_OFF = {"", "off", "none", "no"}


def read_context_files(paths: Iterable[str | Path] | None) -> list[str]:
    texts: list[str] = []
    for raw in paths or []:
        if not str(raw).strip():
            continue
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"program context file not found: {path}")
        text = path.read_text(encoding="utf-8-sig").strip()
        if text:
            texts.append(text)
    return texts


def episode_title_from_workdir(work_paths: WorkPaths) -> str:
    """Media file name of the episode (project.json ``source_name``), which often carries the topic."""
    payload = read_json(work_paths.project_metadata, default={})
    if not isinstance(payload, dict):
        return ""
    name = str(payload.get("source_name") or "").strip()
    if not name and payload.get("original_media_path"):
        name = Path(str(payload["original_media_path"])).stem
    return name


def resolve_episode_title(option: str | None, work_paths: WorkPaths | None) -> str:
    value = EPISODE_TITLE_AUTO if option is None else str(option).strip()
    if value.lower() in EPISODE_TITLE_OFF:
        return ""
    if value.lower() == EPISODE_TITLE_AUTO:
        return episode_title_from_workdir(work_paths) if work_paths is not None else ""
    return value


def build_program_context(files: Iterable[str | Path] | None = None, episode_title: str = "") -> str:
    parts = read_context_files(files)
    if episode_title:
        parts.append(f"本期标题（媒体文件名，可作话题参考）：{episode_title}")
    text = "\n\n".join(parts).strip()
    if len(text) > MAX_CONTEXT_CHARS:
        LOGGER.warning("program context is %d chars; only the first %d are used", len(text), MAX_CONTEXT_CHARS)
        text = text[:MAX_CONTEXT_CHARS]
    return text


def program_context_from_args(args: Any, work_paths: WorkPaths | None) -> str:
    """``--program-context`` files (repeatable) plus ``--episode-title`` (default: auto from the media name)."""
    files = getattr(args, "program_context", None) or []
    if isinstance(files, (str, Path)):
        files = [files]
    option = getattr(args, "episode_title", EPISODE_TITLE_AUTO)
    if not files and str(option or "").strip().lower() == EPISODE_TITLE_AUTO:
        return ""  # the automatic title belongs to a configured program context; default prompts stay unchanged
    return build_program_context(files, resolve_episode_title(option, work_paths))


def program_context_block(context: str) -> str:
    """The block inserted into a main prompt; empty when there is no context."""
    context = (context or "").strip()
    if not context:
        return ""
    return (
        "<program_context>\n"
        "Background of the program (for understanding names, topics and likely ASR mishearings; "
        "do not copy it into the output):\n"
        f"{context}\n"
        "</program_context>\n"
    )
