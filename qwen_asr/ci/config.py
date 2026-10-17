"""Server settings. Every field can be set by environment variable ``KOEAXIS_CI_<FIELD>`` (upper case).

Deployment-specific choices (database, storage path, proxy/SSO headers, webhooks, cookie flags) live
here so that operators can adjust them without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "KOEAXIS_CI_"
DEFAULT_INPUT_FLAGS = (
    "--glossary-xlsx",
    "--asr-context-file",
    "--program-memory-file",
)


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Storage --------------------------------------------------------------------------------
    data_dir: Path = Path("ci-data")
    db_url: str = ""                      # default: sqlite:///<data_dir>/ci.sqlite3 (any SQLAlchemy URL)
    blob_dir: Path | None = None          # default: <data_dir>/blobs
    log_dir: Path | None = None           # default: <data_dir>/logs
    presets_dirs: list[Path] = field(default_factory=lambda: [Path("ci-presets")])
    max_upload_mb: int = 4096
    # Auth -----------------------------------------------------------------------------------
    session_ttl_hours: float = 24 * 14
    cookie_name: str = "koeaxis_session"
    cookie_secure: bool = False           # set true behind HTTPS
    trusted_user_header: str = ""         # e.g. X-Forwarded-User (SSO via reverse proxy); empty = off
    trusted_proxy_ips: list[str] = field(default_factory=lambda: ["127.0.0.1", "::1"])  # "*" = any
    trusted_header_auto_create: bool = False
    member_extra_args: bool = False       # members may pass raw CLI args (admins always may)
    input_flags: list[str] = field(default_factory=lambda: list(DEFAULT_INPUT_FLAGS))
    # Workers / queue --------------------------------------------------------------------------
    worker_stale_seconds: int = 600       # running run without heartbeat for this long -> failed
    worker_affinity_seconds: int = 300    # retries prefer the worker holding the workspace this long
    # Integrations ---------------------------------------------------------------------------
    webhook_urls: list[str] = field(default_factory=list)
    webhook_secret: str = ""              # HMAC-SHA256 signature in X-KoeAxis-Signature
    root_path: str = ""                   # when served under a sub-path by a reverse proxy
    cors_origins: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.blob_dir = Path(self.blob_dir) if self.blob_dir else self.data_dir / "blobs"
        self.log_dir = Path(self.log_dir) if self.log_dir else self.data_dir / "logs"
        self.presets_dirs = [Path(item) for item in self.presets_dirs]
        if not self.db_url:
            self.db_url = f"sqlite:///{(self.data_dir / 'ci.sqlite3').as_posix()}"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **overrides: Any) -> Settings:
        environ = dict(os.environ if environ is None else environ)
        values: dict[str, Any] = {}
        for item in fields(cls):
            raw = environ.get(ENV_PREFIX + item.name.upper())
            if raw is None:
                continue
            kind = str(item.type)
            if "list[Path]" in kind:
                values[item.name] = [Path(part) for part in _split(raw.replace(os.pathsep, ","))]
            elif "list" in kind:
                values[item.name] = _split(raw)
            elif "bool" in kind:
                values[item.name] = _bool(raw)
            elif "int" in kind:
                values[item.name] = int(raw)
            elif "float" in kind:
                values[item.name] = float(raw)
            else:
                values[item.name] = raw
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.blob_dir, self.log_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
