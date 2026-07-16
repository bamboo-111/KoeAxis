from __future__ import annotations

from pathlib import Path

from qwen_asr.mfa_environment import (
    build_project_mfa_command,
    detect_mfa_environment,
    find_project_mfa_executable,
    find_project_mfa_root,
)
from qwen_asr.mfa_experiment import (
    _project_mfa_command,
    _project_mfa_executable,
    _project_mfa_root,
)


def test_build_project_mfa_command_prefers_micromamba(tmp_path: Path) -> None:
    micromamba = tmp_path / "tools" / "micromamba" / "extract" / "Library" / "bin" / "micromamba.exe"
    mfa_env = tmp_path / "tools" / "mfa-env"
    micromamba.parent.mkdir(parents=True)
    micromamba.write_text("", encoding="utf-8")
    mfa_env.mkdir(parents=True)

    command, executable, invocation = build_project_mfa_command(tmp_path)

    assert command == [str(micromamba), "run", "-p", str(mfa_env), "mfa"]
    assert executable == str(micromamba)
    assert invocation == "micromamba-run"


def test_build_project_mfa_command_falls_back_to_direct_executable(tmp_path: Path) -> None:
    direct = tmp_path / "tools" / "mfa-env" / "Scripts" / "mfa.exe"
    direct.parent.mkdir(parents=True)
    direct.write_text("", encoding="utf-8")

    command, executable, invocation = build_project_mfa_command(tmp_path)

    assert command == [str(direct)]
    assert executable == str(direct)
    assert invocation == "direct"
    assert find_project_mfa_executable(tmp_path) == str(direct)


def test_find_project_mfa_root_reports_existing_root(tmp_path: Path) -> None:
    root = tmp_path / "tools" / "mfa-root"
    root.mkdir(parents=True)

    assert find_project_mfa_root(tmp_path) == root


def test_detect_mfa_environment_uses_path_fallback_and_package_version() -> None:
    report = detect_mfa_environment(
        run_version_check=False,
        project_mfa_command=lambda: ([], "", ""),
        project_mfa_root=lambda: None,
        path_mfa_lookup=lambda _name: "path-mfa",
        package_version_lookup=lambda _name: "3.0.0",
    )

    assert report["available"] is True
    assert report["command"] == ["path-mfa"]
    assert report["executable"] == "path-mfa"
    assert report["invocation"] == "path"
    assert report["package_version"] == "3.0.0"


def test_detect_mfa_environment_passes_root_to_version_check(tmp_path: Path) -> None:
    root = tmp_path / "mfa-root"
    calls: list[dict[str, object]] = []

    class Completed:
        returncode = 0
        stdout = "mfa 3.0"
        stderr = ""

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        calls.append({"command": command, "env": kwargs["env"]})
        return Completed()

    report = detect_mfa_environment(
        project_mfa_command=lambda: (["mfa-bin"], "mfa-bin", "direct"),
        project_mfa_root=lambda: root,
        package_version_lookup=lambda _name: "",
        run_command=fake_run,
    )

    assert report["version_output"] == "mfa 3.0"
    assert report["root_dir"] == str(root)
    assert calls[0]["command"] == ["mfa-bin", "version"]
    assert calls[0]["env"]["MFA_ROOT_DIR"] == str(root)


def test_mfa_experiment_keeps_environment_compatibility_aliases() -> None:
    assert callable(_project_mfa_command)
    assert callable(_project_mfa_executable)
    assert callable(_project_mfa_root)
