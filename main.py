from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "compat" / "legacy"))

from qwen_asr.cli import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
