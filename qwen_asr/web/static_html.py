from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
INDEX_TEMPLATE_PATH = TEMPLATE_DIR / "index.html"


def load_index_html() -> str:
    return INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")


INDEX_HTML = load_index_html()
