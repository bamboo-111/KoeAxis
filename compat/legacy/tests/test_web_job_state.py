from __future__ import annotations

import json
from pathlib import Path

from qwen_asr.web import job_state


def test_public_job_redacts_secret_flag_values() -> None:
    job = {
        "status": "running",
        "_process": object(),
        "command": ["python", "-m", "qwen_asr", "--mimo-api-key", "secret-value", "run"],
    }

    payload = job_state.public_job(job)

    assert "_process" not in payload
    assert payload["command"][-2] == "***"
    assert "secret-value" not in json.dumps(payload)


def test_persist_job_writes_global_and_workspace_state(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "workspaces"
    workdir = root / "sample"
    workdir.mkdir(parents=True)
    monkeypatch.setattr(job_state, "WORKSPACES_DIR", root)
    monkeypatch.setattr(job_state, "GLOBAL_JOB_STATE_PATH", root / ".web-state" / "job.json")

    saved = job_state.persist_job(
        {"id": "1", "status": "running", "workdir": str(workdir), "pid": 123, "command": ["python"]}
    )

    global_payload = json.loads((root / ".web-state" / "job.json").read_text(encoding="utf-8"))
    workspace_payload = json.loads((workdir / "reports" / "web_job.json").read_text(encoding="utf-8"))
    assert global_payload["id"] == saved["id"] == "1"
    assert workspace_payload["status"] == "running"


def test_load_job_marks_missing_process_interrupted(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "workspaces"
    state_path = root / ".web-state" / "job.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"id": "1", "status": "running", "workdir": "", "pid": 999999}),
        encoding="utf-8",
    )
    monkeypatch.setattr(job_state, "WORKSPACES_DIR", root)
    monkeypatch.setattr(job_state, "GLOBAL_JOB_STATE_PATH", state_path)
    monkeypatch.setattr(job_state, "pid_is_running", lambda pid: False)

    payload = job_state.load_job()

    assert payload is not None
    assert payload["status"] == "interrupted"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "interrupted"


def test_corrupt_saved_job_is_treated_as_unavailable(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "job.json"
    state_path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(job_state, "GLOBAL_JOB_STATE_PATH", state_path)

    assert job_state.load_job() is None
