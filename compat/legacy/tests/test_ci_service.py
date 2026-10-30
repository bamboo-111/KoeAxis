"""Tests for the CI service (qwen_asr/ci): presets, storage, auth, HTTP API and the worker protocol.

The worker tests use a fake phase command (a tiny Python script) instead of the real ML pipeline.
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qwen_asr.ci import pipeline
from qwen_asr.ci.api import create_app
from qwen_asr.ci.auth import hash_password, hash_token, new_token, verify_password
from qwen_asr.ci.blobs import LocalBlobStore, LogStore, _complete_utf8_length
from qwen_asr.ci.config import Settings
from qwen_asr.ci.phase import EVENT_PREFIX
from qwen_asr.ci.store import Store, utcnow
from qwen_asr.ci.worker import RunState, ServerClient, Worker, WorkerConfig, parse_event, resolve_model_reference

ROOT = Path(__file__).resolve().parents[1]
CSRF = {"X-Requested-With": "koeaxis"}


# ------------------------------------------------------------------------------------------- fixtures
@pytest.fixture
def preset_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "presets"
    directory.mkdir()
    (directory / "glossary.txt").write_bytes("トゲトゲ=刺刺\n".encode("utf-8"))
    (directory / "demo.json").write_text(json.dumps({
        "id": "demo", "title": "Demo", "language": "Japanese", "run_args": ["--with-split", "--format", "srt"],
        "files": [{"flag": "--asr-context-file", "path": "glossary.txt"}],
        "env": {"DEMO_ENV": "1"}, "options": {"llm_model": "mimo-v2.6-pro"},
    }), encoding="utf-8")
    return directory


@pytest.fixture
def server(tmp_path: Path, preset_dir: Path):
    settings = Settings(data_dir=tmp_path / "data", presets_dirs=[preset_dir], worker_stale_seconds=60)
    app = create_app(settings, start_background=False)
    store: Store = app.state.store
    store.create_user("admin", hash_password("admin-pass-1"), "admin")
    store.create_user("alice", hash_password("alice-pass-1"), "member")
    worker_token = new_token("worker")
    store.create_token("gpu-01", hash_token(worker_token), "worker")
    with TestClient(app) as client:
        yield app, client, worker_token
    store.engine.dispose()


def login(client: TestClient, username: str, password: str) -> None:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text


def upload(client: TestClient, name: str, data: bytes) -> dict:
    response = client.post("/api/v1/blobs", files={"file": (name, BytesIO(data), "audio/mpeg")}, headers=CSRF)
    assert response.status_code == 200, response.text
    return response.json()


def worker_headers(token: str, worker_id: str = "gpu-01") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-KoeAxis-Worker": worker_id}


# ------------------------------------------------------------------------------------------- units
def test_presets_and_run_argv_order(preset_dir: Path) -> None:
    presets = pipeline.discover_presets([preset_dir])
    assert set(presets) == {"demo"}
    public = presets["demo"].public()
    assert public["files"] == [{"index": 0, "flag": "--asr-context-file", "name": "glossary.txt"}]
    argv = pipeline.build_run_argv(
        public, workdir=Path("w"), media=Path("m.mp3"), export_path=Path("out.srt"),
        file_paths=[("--asr-context-file", Path("g.txt"))], options={"llm_model": "custom", "language": "auto"},
        extra_args=["--format", "both"], worker_args=["--device", "cuda:0"],
    )
    assert argv[:4] == ["--workdir", "w", "--media", "m.mp3"]
    assert "--language" not in argv                     # "auto" = autodetect
    assert argv.index("--device") < argv.index("--llm-model") < argv.index("both") < argv.index("--export-path")
    assert argv[argv.index("--llm-model") + 1] == "custom"   # run option overrides preset default
    assert argv[-4:] == ["--export-mode", "custom", "--export-path", "out.srt"]
    with pytest.raises(pipeline.PresetError):
        pipeline.validate_options({"evil": "1"})


def test_repository_presets_load() -> None:
    presets = pipeline.discover_presets([ROOT / "ci-presets"])
    assert {"togeradi", "japanese-generic"} <= set(presets)
    for preset in presets.values():
        for item in preset.files:
            assert item.path.is_file(), item.path


def test_password_and_token_helpers() -> None:
    encoded = hash_password("correct horse")
    assert verify_password("correct horse", encoded)
    assert not verify_password("wrong horse!", encoded)
    assert not verify_password("x", "garbage")
    with pytest.raises(ValueError):
        hash_password("short")
    token = new_token("worker")
    assert token.startswith("kxw_") and hash_token(token) != token


def test_blob_store_dedup_and_log_store_utf8(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    sha1, size1 = blobs.put(BytesIO(b"hello"))
    sha2, _ = blobs.put(BytesIO(b"hello"))
    assert sha1 == sha2 and size1 == 5 and blobs.exists(sha1)
    logs = LogStore(tmp_path / "logs")
    logs.append("run1", "日本語ログ\n")
    text, offset = logs.read("run1", 0, 4)          # 4 bytes cut inside the second character
    assert text == "日" and offset == 3
    rest, end = logs.read("run1", offset)
    assert text + rest == "日本語ログ\n" and end == logs.size("run1")
    assert _complete_utf8_length("ab".encode()) == 2


def test_parse_event_and_run_state() -> None:
    line = EVENT_PREFIX + json.dumps({"event": "stage", "stage": "align", "status": "completed", "elapsed_s": 3.5})
    event = parse_event(line + "\n")
    assert event and event["stage"] == "align"
    assert parse_event("plain log line") is None
    state = RunState()
    state.on_event(event)
    state.merge_performance({"stages": [{"stage": "translate", "status": "completed", "wall_time_s": 62.87,
                                         "network_summary": {"request_count": 42, "failure_count": 0, "token_totals": {
                                             "input_tokens": 1000, "cached_tokens": 900, "output_tokens": 50}}}]})
    state.set_checks({"status": "WARN", "checks": [{"name": "content_quality", "status": "WARN", "message": "m"}]})
    state.finalize()
    snap = state.snapshot()
    stages = {item["name"]: item for item in snap["stages"]}
    assert stages["align"]["status"] == "completed" and stages["prepare"]["status"] == "skipped"
    assert snap["metrics"]["cache_hit_rate"] == 0.9 and snap["metrics"]["quality_status"] == "WARN"
    assert snap["checks"][0] == {"name": "final_quality", "status": "WARN", "message": "聚合质量门"}


def test_resolve_model_reference(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--org--model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    assert resolve_model_reference("org/model", tmp_path) == str(snapshot)
    assert resolve_model_reference("org/other", tmp_path) == "org/other"
    assert resolve_model_reference("org/model", None) == "org/model"


def test_settings_from_env(tmp_path: Path) -> None:
    settings = Settings.from_env({
        "KOEAXIS_CI_DATA_DIR": str(tmp_path), "KOEAXIS_CI_COOKIE_SECURE": "true",
        "KOEAXIS_CI_WEBHOOK_URLS": "http://a/hook, http://b/hook", "KOEAXIS_CI_WORKER_STALE_SECONDS": "30",
    })
    assert settings.cookie_secure is True and settings.worker_stale_seconds == 30
    assert settings.webhook_urls == ["http://a/hook", "http://b/hook"]
    assert settings.db_url.endswith("ci.sqlite3") and settings.blob_dir == tmp_path / "blobs"


# ------------------------------------------------------------------------------------------- API
def test_auth_required_and_csrf(server) -> None:
    _app, client, _token = server
    assert client.get("/api/v1/runs").status_code == 401
    assert client.post("/api/v1/auth/login", json={"username": "admin", "password": "nope-nope"}).status_code == 401
    login(client, "admin", "admin-pass-1")
    assert client.get("/api/v1/me").json()["user"]["role"] == "admin"
    # cookie-authenticated mutation without the CSRF header is refused
    response = client.post("/api/v1/blobs", files={"file": ("a.mp3", BytesIO(b"x"), "audio/mpeg")})
    assert response.status_code == 403
    assert client.get("/api/v1/openapi.json").json()["info"]["title"] == "KoeAxis CI"
    assert client.get("/").status_code == 200


def test_full_run_lifecycle_over_worker_protocol(server) -> None:
    app, client, worker_token = server
    login(client, "alice", "alice-pass-1")
    media = upload(client, "ep29.mp3", b"fake-audio")
    # members cannot pass raw CLI args by default
    denied = client.post("/api/v1/runs", json={"preset_id": "demo", "media_blob_id": media["id"],
                                               "extra_args": ["--force"]}, headers=CSRF)
    assert denied.status_code == 403
    run = client.post("/api/v1/runs", json={"preset_id": "demo", "media_blob_id": media["id"],
                                            "options": {"asr_model": "org/model"}, "labels": {"episode": "29"}},
                      headers=CSRF).json()
    assert run["status"] == "queued" and run["queue_position"] == 1 and run["title"] == "ep29"
    assert [item["source"] for item in run["inputs"]] == ["preset"]

    wh = worker_headers(worker_token)
    assert client.post("/api/v1/worker/claim", json={}, headers={"Authorization": "Bearer bad", **CSRF}).status_code == 401
    claimed = client.post("/api/v1/worker/claim", json={"name": "GPU 1", "info": {"gpu": "4090"}}, headers=wh)
    assert claimed.status_code == 200
    spec = claimed.json()["run"]
    assert spec["id"] == run["id"] and spec["env"] == {"DEMO_ENV": "1"} and spec["options"]["asr_model"] == "org/model"
    assert client.post("/api/v1/worker/claim", json={}, headers=wh).status_code == 204
    assert client.get(spec["media"]["url"], headers=wh).content == b"fake-audio"
    assert client.get(spec["inputs"][0]["url"], headers=wh).content.decode("utf-8") == "トゲトゲ=刺刺\n"
    # another worker cannot touch the run
    assert client.get(spec["media"]["url"], headers=worker_headers(worker_token, "other")).status_code == 404

    stages = [{"name": "prepare", "phase": "gpu", "status": "completed", "elapsed_s": 1.0}]
    beat = client.post(f"/api/v1/worker/runs/{run['id']}/heartbeat", json={"stages": stages, "summary": "prepare"},
                       headers=wh).json()
    assert beat == {"cancel_requested": False, "status": "running"}
    client.post(f"/api/v1/worker/runs/{run['id']}/log", content="第一行\n".encode(), headers=wh)
    srt = "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n你好\n\n"
    client.post(f"/api/v1/worker/runs/{run['id']}/artifacts", headers=wh, data={"name": "ep29.srt", "kind": "subtitle"},
                files={"file": ("ep29.srt", BytesIO(srt.encode()), "text/plain")})
    done = client.post(f"/api/v1/worker/runs/{run['id']}/complete", headers=wh, json={
        "status": "succeeded", "metrics": {"subtitle_count": 1}, "checks": [{"name": "final_quality", "status": "PASS"}]})
    assert done.json()["status"] == "succeeded"

    detail = client.get(f"/api/v1/runs/{run['id']}").json()
    assert detail["artifacts"][0]["name"] == "ep29.srt" and detail["stages"] == stages
    assert client.get(f"/api/v1/runs/{run['id']}/artifacts/ep29.srt").content.decode() == srt
    log = client.get(f"/api/v1/runs/{run['id']}/log").json()
    assert "第一行" in log["text"] and "finished: succeeded" in log["text"] and log["complete"]
    tail = client.get(f"/api/v1/runs/{run['id']}/log", params={"offset": log["next_offset"]}).json()
    assert tail["text"] == ""
    workers = client.get("/api/v1/workers").json()
    assert workers[0]["id"] == "gpu-01" and workers[0]["info"]["gpu"] == "4090"

    # retry from translate reuses the workspace and prefers the same worker
    retry = client.post(f"/api/v1/runs/{run['id']}/retry", json={"reset_from": "translate",
                                                                  "options": {"llm_model": "mimo-v2.6-flash"}},
                        headers=CSRF).json()
    assert retry["workspace_key"] == run["workspace_key"] and retry["reset_from"] == "translate"
    assert retry["worker_hint"] == "gpu-01" and retry["options"]["llm_model"] == "mimo-v2.6-flash"
    assert client.post(f"/api/v1/runs/{run['id']}/retry", json={"reset_from": "bogus"}, headers=CSRF).status_code == 400
    assert client.post("/api/v1/worker/claim", json={}, headers=worker_headers(worker_token, "gpu-02")).status_code == 204
    second = client.post("/api/v1/worker/claim", json={}, headers=wh).json()["run"]
    assert second["id"] == retry["id"] and second["reset_from"] == "translate"
    client.post(f"/api/v1/worker/runs/{retry['id']}/artifacts", headers=wh, data={"name": "ep29.srt", "kind": "subtitle"},
                files={"file": ("ep29.srt", BytesIO(srt.replace("你好", "您好").encode()), "text/plain")})
    client.post(f"/api/v1/worker/runs/{retry['id']}/complete", headers=wh, json={"status": "succeeded"})

    compare = client.get("/api/v1/compare", params={"a": run["id"], "b": retry["id"]}).json()
    assert compare["srt"]["a_count"] == 1 and compare["srt"]["similarity"] == 0.0
    assert any(line.startswith("+") and "您好" in line for line in compare["srt"]["diff"])

    # only the creator or an admin may delete; deleting removes artifacts
    assert client.delete(f"/api/v1/runs/{run['id']}", headers=CSRF).status_code == 200
    assert client.get(f"/api/v1/runs/{run['id']}").status_code == 404
    assert app.state.store.list_runs()[1] == 1


def test_cancel_queued_and_running(server) -> None:
    _app, client, worker_token = server
    login(client, "admin", "admin-pass-1")
    media = upload(client, "a.mp3", b"a")
    queued = client.post("/api/v1/runs", json={"preset_id": "demo", "media_blob_id": media["id"]}, headers=CSRF).json()
    cancelled = client.post(f"/api/v1/runs/{queued['id']}/cancel", headers=CSRF).json()
    assert cancelled["status"] == "cancelled"
    running = client.post("/api/v1/runs", json={"preset_id": "demo", "media_blob_id": media["id"],
                                                "extra_args": ["--force"]}, headers=CSRF).json()   # admin may
    wh = worker_headers(worker_token)
    assert client.post("/api/v1/worker/claim", json={}, headers=wh).json()["run"]["id"] == running["id"]
    client.post(f"/api/v1/runs/{running['id']}/cancel", headers=CSRF)
    assert client.post(f"/api/v1/worker/runs/{running['id']}/heartbeat", json={}, headers=wh).json()["cancel_requested"]
    final = client.post(f"/api/v1/worker/runs/{running['id']}/complete", json={"status": "failed"}, headers=wh).json()
    assert final["status"] == "cancelled"


def test_stale_runs_are_reaped(server) -> None:
    app, client, worker_token = server
    login(client, "admin", "admin-pass-1")
    media = upload(client, "a.mp3", b"a")
    run = client.post("/api/v1/runs", json={"preset_id": "demo", "media_blob_id": media["id"]}, headers=CSRF).json()
    client.post("/api/v1/worker/claim", json={}, headers=worker_headers(worker_token))
    app.state.store.update_run(run["id"], heartbeat_at=utcnow() - timedelta(seconds=3600))
    assert client.post("/api/v1/admin/reap", headers=CSRF).json() == {"reaped": [run["id"]]}
    detail = client.get(f"/api/v1/runs/{run['id']}").json()
    assert detail["status"] == "failed" and detail["error"] == "worker-lost"


def test_admin_endpoints_and_member_restrictions(server) -> None:
    _app, client, _token = server
    login(client, "alice", "alice-pass-1")
    assert client.get("/api/v1/admin/users").status_code == 403
    created = client.post("/api/v1/me/tokens", json={"name": "script"}, headers=CSRF).json()
    api_client = TestClient(client.app)
    assert api_client.get("/api/v1/me", headers={"Authorization": f"Bearer {created['token']}"}).json()["via"] == "token"
    login(client, "admin", "admin-pass-1")
    user = client.post("/api/v1/admin/users", json={"username": "bob", "password": "bob-pass-12"}, headers=CSRF).json()
    assert user["role"] == "member" and "password_hash" not in user
    client.patch(f"/api/v1/admin/users/{user['id']}", json={"disabled": True}, headers=CSRF)
    assert client.post("/api/v1/auth/login", json={"username": "bob", "password": "bob-pass-12"}).status_code == 401
    token = client.post("/api/v1/admin/tokens", json={"name": "gpu-02", "kind": "worker"}, headers=CSRF).json()
    assert token["token"].startswith("kxw_")
    actions = [entry["action"] for entry in client.get("/api/v1/admin/audit").json()]
    assert "user.create" in actions and "token.create" in actions


def test_trusted_proxy_header(tmp_path: Path, preset_dir: Path) -> None:
    settings = Settings(data_dir=tmp_path / "d", presets_dirs=[preset_dir], trusted_user_header="X-Forwarded-User",
                        trusted_proxy_ips=["testclient"], trusted_header_auto_create=True)
    app = create_app(settings, start_background=False)
    with TestClient(app) as client:
        me = client.get("/api/v1/me", headers={"X-Forwarded-User": "carol"}).json()
        assert me["user"]["username"] == "carol" and me["via"] == "header"
    app.state.store.engine.dispose()


def test_ip_trusted_supports_exact_cidr_and_wildcard() -> None:
    from qwen_asr.ci.api import ip_trusted

    assert ip_trusted("127.0.0.1", ["127.0.0.1"])
    assert ip_trusted("172.18.0.5", ["172.16.0.0/12"])
    assert ip_trusted("fd00::5", ["fd00::/8"])
    assert not ip_trusted("10.0.0.1", ["172.16.0.0/12", "bad/entry"])
    assert not ip_trusted("", ["127.0.0.1"]) and ip_trusted("anything", ["*"])


# ------------------------------------------------------------------------------------------- worker
FAKE_PHASE = textwrap.dedent('''
    import json, sys, time
    from pathlib import Path
    args = sys.argv[1:]
    spec = json.loads(Path(args[args.index("--spec") + 1]).read_text(encoding="utf-8"))
    phase = args[args.index("--phase") + 1]
    argv = spec["argv"]
    workdir = Path(argv[argv.index("--workdir") + 1])
    export = Path(argv[argv.index("--export-path") + 1])
    def emit(**payload):
        print("@@KOEAXIS " + json.dumps(payload, ensure_ascii=False), flush=True)
    (workdir / "reports").mkdir(parents=True, exist_ok=True)
    print("argv:", json.dumps(argv, ensure_ascii=False), flush=True)
    print("reset_from:", spec.get("reset_from"), flush=True)
    if phase == "gpu" and spec.get("reset_from"):
        emit(event="reset", stage=spec["reset_from"], removed=["translated.json"])
    if phase == "gpu":
        if "--hang" in argv:
            emit(event="stage", stage="transcribe", status="running")
            time.sleep(60)
        for stage in ("prepare", "transcribe", "align"):
            emit(event="stage", stage=stage, status="running")
            emit(event="stage", stage=stage, status="completed", elapsed_s=0.1)
        sys.exit(0)
    for stage in ("split", "translate", "export"):
        emit(event="stage", stage=stage, status="running")
        emit(event="stage", stage=stage, status="completed", elapsed_s=0.2)
    (workdir / "reports" / "pipeline_performance.api.json").write_text(json.dumps({"stages": [
        {"stage": "translate", "status": "completed", "wall_time_s": 1.5, "network_summary": {
            "request_count": 3, "failure_count": 0,
            "token_totals": {"input_tokens": 100, "cached_tokens": 80, "output_tokens": 10}}}]}), encoding="utf-8")
    (workdir / "final_quality.json").write_text(json.dumps({"status": "PASS", "checks": [
        {"name": "content_quality", "status": "PASS", "message": "ok"}]}), encoding="utf-8")
    (workdir / "manifests").mkdir(exist_ok=True)
    (workdir / "manifests" / "segments.json").write_text("{}", encoding="utf-8")
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_text("1\\n00:00:01,000 --> 00:00:02,000\\nテスト\\n测试\\n\\n", encoding="utf-8")
''')


def _make_worker(app, client: TestClient, token: str, tmp_path: Path) -> Worker:
    script = tmp_path / "fake_phase.py"
    script.write_text(FAKE_PHASE, encoding="utf-8")
    config = WorkerConfig(server="http://testserver", token=token, worker_id="gpu-01", data_dir=tmp_path / "worker",
                          phase_command=[sys.executable, str(script)], default_run_args=["--device", "cuda:0"],
                          heartbeat_interval_s=0.2, log_flush_interval_s=0.1, request_timeout_s=10,
                          model_cache_dir=tmp_path / "models")
    return Worker(config, ServerClient(config, client=client))


def test_worker_executes_run_end_to_end(server, tmp_path: Path) -> None:
    app, client, worker_token = server
    (tmp_path / "models" / "models--org--model" / "snapshots" / "rev1").mkdir(parents=True)
    login(client, "alice", "alice-pass-1")
    media = upload(client, "第29期.mp3", b"fake-audio")
    run = client.post("/api/v1/runs", json={"preset_id": "demo", "media_blob_id": media["id"],
                                            "options": {"asr_model": "org/model"}}, headers=CSRF).json()
    worker = _make_worker(app, client, worker_token, tmp_path)
    executor = worker.poll_once()
    assert executor is not None
    executor.join(timeout=60)
    assert not executor.is_alive()

    detail = client.get(f"/api/v1/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail
    stages = {item["name"]: item for item in detail["stages"]}
    assert stages["align"]["status"] == "completed" and stages["normalize"]["status"] == "skipped"
    assert detail["metrics"]["llm_requests"] == 3 and detail["metrics"]["subtitle_count"] == 1
    assert detail["checks"][0]["status"] == "PASS"
    names = {item["name"]: item["kind"] for item in detail["artifacts"]}
    assert names == {"第29期.srt": "subtitle", "reports.zip": "report", "manifests.zip": "data"}
    log = client.get(f"/api/v1/runs/{run['id']}/log").json()["text"]
    argv = json.loads(next(line for line in log.splitlines() if line.startswith("argv:"))[5:])
    assert argv[argv.index("--model") + 1].endswith("rev1")            # HF id resolved to the local snapshot
    assert argv[argv.index("--device") + 1] == "cuda:0"                # worker-level default
    assert "--asr-context-file" in argv
    assert "DEMO_ENV" not in log and "@@KOEAXIS" not in log            # events are parsed, not logged

    # a retry reuses the workspace (media is not downloaded again) and passes reset_from to the phase
    retry = client.post(f"/api/v1/runs/{run['id']}/retry", json={"reset_from": "translate"}, headers=CSRF).json()
    executor = worker.poll_once()
    executor.join(timeout=60)
    retry_log = client.get(f"/api/v1/runs/{retry['id']}/log").json()["text"]
    assert "reset_from: translate" in retry_log and "downloading media" not in retry_log
    assert "[ci] reset_from=translate: removed 1 item(s): translated.json" in retry_log
    assert client.get(f"/api/v1/runs/{retry['id']}").json()["status"] == "succeeded"


def test_worker_cancels_running_phase(server, tmp_path: Path) -> None:
    app, client, worker_token = server
    login(client, "admin", "admin-pass-1")
    media = upload(client, "x.mp3", b"x")
    run = client.post("/api/v1/runs", json={"preset_id": "demo", "media_blob_id": media["id"],
                                            "extra_args": ["--hang"]}, headers=CSRF).json()
    worker = _make_worker(app, client, worker_token, tmp_path)
    executor = worker.poll_once()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and executor.process is None:
        time.sleep(0.05)
    time.sleep(0.5)
    assert not worker.can_claim()                    # its GPU phase blocks further claims
    client.post(f"/api/v1/runs/{run['id']}/cancel", headers=CSRF)
    executor.join(timeout=60)
    assert not executor.is_alive()
    detail = client.get(f"/api/v1/runs/{run['id']}").json()
    assert detail["status"] == "cancelled"
    assert {item["name"]: item for item in detail["stages"]}["transcribe"]["status"] == "cancelled"


def test_phase_reset_from_discards_stage_and_downstream(tmp_path: Path) -> None:
    from qwen_asr.ci.phase import reset_from
    from qwen_asr.models import WorkPaths

    paths = WorkPaths.from_workdir(tmp_path / "work")
    keep = [paths.segments_manifest, paths.aligned_manifest, paths.split_manifest]
    drop = [paths.translated_manifest, paths.normalized_manifest]
    for path in keep + drop:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
    removed = reset_from(paths, "translate")
    assert all(path.exists() for path in keep)
    assert not any(path.exists() for path in drop)
    assert str(paths.translated_manifest) in removed
