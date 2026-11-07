from __future__ import annotations

from qwen_asr.web import server as _server
from qwen_asr.web.commands import (
    HOST,
    PORT,
    ROOT,
    SUPPORTED_ASR_LANGUAGES,
    WORKSPACES_DIR,
    build_command,
    normalize_asr_language,
)
from qwen_asr.web.server import (
    JOB_LOCK,
    Handler,
    get_active_job,
    main,
    start_job,
    stop_job,
)
from qwen_asr.web.static_html import INDEX_HTML
from qwen_asr.web.status import build_progress, get_status

__all__ = [
    "HOST",
    "INDEX_HTML",
    "JOB_LOCK",
    "PORT",
    "ROOT",
    "SUPPORTED_ASR_LANGUAGES",
    "WORKSPACES_DIR",
    "Handler",
    "build_command",
    "build_progress",
    "get_active_job",
    "get_status",
    "main",
    "normalize_asr_language",
    "start_job",
    "stop_job",
]


def __getattr__(name: str):
    if name == "ACTIVE_JOB":
        return _server.ACTIVE_JOB
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
