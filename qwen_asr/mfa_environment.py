from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


def detect_mfa_environment(
    *,
    run_version_check: bool = True,
    project_mfa_command: Callable[[], tuple[list[str], str, str]] | None = None,
    project_mfa_root: Callable[[], Path | None] | None = None,
    path_mfa_lookup: Callable[[str], str | None] | None = None,
    package_version_lookup: Callable[[str], str] | None = None,
    run_command: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    project_mfa_command = project_mfa_command or build_project_mfa_command
    project_mfa_root = project_mfa_root or find_project_mfa_root
    path_mfa_lookup = path_mfa_lookup or shutil.which
    package_version_lookup = package_version_lookup or _package_version
    run_command = run_command or subprocess.run

    command, executable, invocation = project_mfa_command()
    if not command:
        path_mfa = path_mfa_lookup("mfa") or ""
        if path_mfa:
            command = [path_mfa]
            executable = path_mfa
            invocation = "path"
    root_dir = project_mfa_root()
    package_version = package_version_lookup("montreal-forced-aligner")
    version_output = ""
    version_error = ""
    if command and run_version_check:
        try:
            env = os.environ.copy()
            if root_dir:
                env["MFA_ROOT_DIR"] = str(root_dir)
            completed = run_command(
                [*command, "version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
                env=env,
            )
            version_output = (completed.stdout or completed.stderr or "").strip()
            if completed.returncode != 0:
                version_error = f"mfa version exited with {completed.returncode}"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            version_error = str(exc)
    return {
        "available": bool(command),
        "executable": executable or "",
        "invocation": invocation,
        "command": command,
        "root_dir": str(root_dir) if root_dir else "",
        "package_version": package_version,
        "version_output": version_output,
        "version_error": version_error,
    }


def build_project_mfa_command(project_root: Path | None = None) -> tuple[list[str], str, str]:
    project_root = project_root or Path(__file__).resolve().parents[1]
    micromamba = project_root / "tools" / "micromamba" / "extract" / "Library" / "bin" / "micromamba.exe"
    mfa_env = project_root / "tools" / "mfa-env"
    if micromamba.exists() and mfa_env.exists():
        return [str(micromamba), "run", "-p", str(mfa_env), "mfa"], str(micromamba), "micromamba-run"
    direct = find_project_mfa_executable(project_root)
    if direct:
        return [direct], direct, "direct"
    return [], "", ""


def find_project_mfa_executable(project_root: Path | None = None) -> str:
    project_root = project_root or Path(__file__).resolve().parents[1]
    candidate = project_root / "tools" / "mfa-env" / "Scripts" / "mfa.exe"
    return str(candidate) if candidate.exists() else ""


def find_project_mfa_root(project_root: Path | None = None) -> Path | None:
    project_root = project_root or Path(__file__).resolve().parents[1]
    candidate = project_root / "tools" / "mfa-root"
    return candidate if candidate.exists() else None


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""
