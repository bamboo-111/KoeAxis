from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_TEMPLATE_PATH = TEMPLATE_DIR / "index.html"
WORKBENCH_TEMPLATE_PATH = TEMPLATE_DIR / "workbench.html"


def load_index_html() -> str:
    return INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")


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
