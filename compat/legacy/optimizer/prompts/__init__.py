"""Prompt template loading helpers."""

import functools
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).parent


@functools.lru_cache(maxsize=32)
def _load_prompt_file(prompt_path: str) -> str:
    """Load a raw prompt template from a .md file."""
    file_path = PROMPTS_DIR / f"{prompt_path}.md"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}.md\n"
            f"Expected location: {file_path}"
        )

    return file_path.read_text(encoding="utf-8")


def get_prompt(prompt_path: str, **kwargs: str | int) -> str:
    """Load a prompt and substitute ``${variable}`` placeholders."""
    raw_prompt = _load_prompt_file(prompt_path)

    if not kwargs:
        return raw_prompt

    template = Template(raw_prompt)
    return template.safe_substitute(**{k: str(v) for k, v in kwargs.items()})


def list_prompts() -> list[str]:
    """List all available prompt paths."""
    prompts: list[str] = []
    for md_file in PROMPTS_DIR.rglob("*.md"):
        if md_file.name == "README.md":
            continue
        rel_path = md_file.relative_to(PROMPTS_DIR)
        prompt_path = str(rel_path.with_suffix("")).replace("\\", "/")
        prompts.append(prompt_path)
    return sorted(prompts)


def reload_cache() -> None:
    """Clear the prompt file cache."""
    _load_prompt_file.cache_clear()


__all__ = ["get_prompt", "list_prompts", "reload_cache"]
