from __future__ import annotations

import html
from pathlib import Path

from qwen_asr.defaults import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_LLM_MODEL,
    DEFAULT_MIMO_CONCURRENCY,
)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_TEMPLATE_PATH = TEMPLATE_DIR / "index.html"
WORKBENCH_TEMPLATE_PATH = TEMPLATE_DIR / "workbench.html"


def index_template_defaults() -> dict[str, str]:
    """Values substituted into ``{{NAME}}`` placeholders so the page shares the CLI defaults."""
    return {
        "DEFAULT_LLM_MODEL": DEFAULT_LLM_MODEL,
        "DEFAULT_LLM_BASE_URL": DEFAULT_LLM_BASE_URL,
        "DEFAULT_LLM_CONCURRENCY": str(DEFAULT_LLM_CONCURRENCY),
        "DEFAULT_MIMO_CONCURRENCY": str(DEFAULT_MIMO_CONCURRENCY),
    }


def load_index_html() -> str:
    text = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    for name, value in index_template_defaults().items():
        text = text.replace("{{" + name + "}}", html.escape(value, quote=True))
    return text


INDEX_HTML = load_index_html()
WORKBENCH_HTML = WORKBENCH_TEMPLATE_PATH.read_text(encoding="utf-8")


def load_static_asset(name: str) -> tuple[bytes, str]:
    allowed = {
        "workbench.css": "text/css; charset=utf-8",
        "workbench.js": "text/javascript; charset=utf-8",
    }
    content_type = allowed.get(name)
    if content_type is None:
        raise FileNotFoundError(name)
    return (STATIC_DIR / name).read_bytes(), content_type
