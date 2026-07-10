from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimizer.llm_client import DEFAULT_TIMEOUT


@dataclass(frozen=True, slots=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    disable_thinking: bool = True
    llm_extra_body: dict[str, Any] | None = None
    timeout: float = DEFAULT_TIMEOUT
