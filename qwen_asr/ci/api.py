"""HTTP API of the CI control server (FastAPI). OpenAPI schema: ``/api/v1/openapi.json``, docs: ``/api/docs``.

Two kinds of callers:

* users  - browser session cookie, ``Authorization: Bearer kxu_...`` user token, or a trusted reverse-proxy
           header (SSO). Cookie-authenticated mutating requests must send ``X-Requested-With: koeaxis``.
* workers - ``Authorization: Bearer kxw_...`` plus ``X-KoeAxis-Worker: <worker id>``; they only see
           runs assigned to them.

The server never imports the ML stack; it stores files, state and logs and hands work to workers.
"""
from __future__ import annotations

import difflib
import hashlib
import hmac
import json
import logging
import re
import secrets
import threading
import time
import urllib.request
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qwen_asr.ci import pipeline
from qwen_asr.ci.auth import Principal, hash_password, hash_token, new_token, verify_password
from qwen_asr.ci.blobs import BlobStore, BlobTooLarge, LocalBlobStore, LogStore
from qwen_asr.ci.config import Settings
from qwen_asr.ci.store import FINAL_STATUSES, Store, utcnow

API_VERSION = "1"
STATIC_DIR = Path(__file__).with_name("static")
CSRF_HEADER = "x-requested-with"
WORKER_HEADER = "x-koeaxis-worker"
LOG_APPEND_LIMIT = 4 * 1024 * 1024
logger = logging.getLogger("koeaxis.ci")


# ------------------------------------------------------------------------------------------- schemas
class LoginBody(BaseModel):
    username: str
    password: str


class PasswordBody(BaseModel):
    old_password: str = ""
    new_password: str


class InputRef(BaseModel):
    flag: str
    blob_id: str


class RunCreate(BaseModel):
    preset_id: str
    media_blob_id: str
    title: str = ""
    options: dict[str, Any] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    inputs: list[InputRef] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    priority: int = 0


class RunRetry(BaseModel):
    reset_from: str | None = None
    options: dict[str, Any] | None = None
    extra_args: list[str] | None = None
    fresh_workspace: bool = False
    refresh_preset: bool = False
    title: str | None = None


class RunPatch(BaseModel):
    title: str | None = None
    labels: dict[str, str] | None = None
    priority: int | None = None


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "member"
    display_name: str = ""


class UserPatch(BaseModel):
    display_name: str | None = None
    role: str | None = None
    disabled: bool | None = None
    password: str | None = None


class TokenCreate(BaseModel):
    name: str
    kind: str = "user"
    username: str | None = None


class ClaimBody(BaseModel):
    name: str | None = None                 # None/omitted = keep the stored value
    info: dict[str, Any] | None = None
    preset_ids: list[str] | None = None


class HeartbeatBody(BaseModel):
    stages: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None
    checks: list[dict[str, Any]] | None = None
    summary: str | None = None


class CompleteBody(HeartbeatBody):
    status: str
    error: str = ""


USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


# ------------------------------------------------------------------------------------------- helpers
def ip_trusted(ip: str, entries: list[str]) -> bool:
    """True when `ip` matches any entry: "*", an exact address or a CIDR network (e.g. 172.16.0.0/12)."""
    import ipaddress

    if "*" in entries:
        return True
    if not ip:
        return False
    if ip in entries:
        return True
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in entries:
        if "/" not in entry:
            continue
        try:
            if address in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


class LoginThrottle:
    def __init__(self, limit: int = 8, window_s: float = 600) -> None:
        self.limit, self.window_s = limit, window_s
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, key: str) -> deque[float]:
        bucket = self._failures[key]
        while bucket and bucket[0] < time.monotonic() - self.window_s:
            bucket.popleft()
        return bucket

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._trim(key)) >= self.limit

    def fail(self, key: str) -> None:
        with self._lock:
            self._trim(key).append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in user.items() if key != "password_hash"}


def _srt_texts(raw: bytes) -> list[str]:
    text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) >= 2 and "-->" in lines[1]:
            cues.append(f"{lines[1].strip()}  " + " / ".join(line.strip() for line in lines[2:]))
        elif lines:
            cues.append(" / ".join(lines))
    return cues


class Webhooks:
    """POST ``{"event": ..., "run": {...}}`` to every configured URL (fire and forget, signed when a secret is set)."""

    def __init__(self, urls: list[str], secret: str) -> None:
        self.urls, self.secret = list(urls), secret

    def send(self, event: str, payload: dict[str, Any]) -> None:
        if not self.urls:
            return
        body = json.dumps({"event": event, **payload}, ensure_ascii=False, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": "KoeAxis-CI", "X-KoeAxis-Event": event}
        if self.secret:
            headers["X-KoeAxis-Signature"] = "sha256=" + hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()
        for url in self.urls:
            threading.Thread(target=self._post, args=(url, body, headers), daemon=True).start()

    @staticmethod
    def _post(url: str, body: bytes, headers: dict[str, str]) -> None:
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=10):  # noqa: S310 - operator-configured URL
                pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("webhook %s failed: %s", url, exc)


# ------------------------------------------------------------------------------------------- app
def create_app(settings: Settings | None = None, *, store: Store | None = None,
               blob_store: BlobStore | None = None, start_background: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    store = store or Store(settings.db_url)
    blob_store = blob_store or LocalBlobStore(settings.blob_dir)
    logs = LogStore(settings.log_dir)
    webhooks = Webhooks(settings.webhook_urls, settings.webhook_secret)
    throttle = LoginThrottle()
    stop_event = threading.Event()

    def reap_stale() -> list[str]:
        """Mark running runs whose worker stopped sending heartbeats as failed."""
        reaped = []
        for run in store.stale_runs(timedelta(seconds=settings.worker_stale_seconds)):
            store.update_run(run["id"], status="failed", finished_at=utcnow(), error="worker-lost",
                             summary=f"worker {run.get('worker_id')} 心跳超时（>{settings.worker_stale_seconds}s）")
            logs.append(run["id"], "\n[ci] worker heartbeat timed out; run marked failed (worker-lost)\n")
            notify("run.finished", store.get_run(run["id"]))
            reaped.append(run["id"])
        return reaped

    def reaper_loop() -> None:
        while not stop_event.wait(min(60, max(5, settings.worker_stale_seconds // 4))):
            try:
                reap_stale()
            except Exception:  # noqa: BLE001
                logger.exception("stale-run reaper failed")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        thread = None
        if start_background:
            thread = threading.Thread(target=reaper_loop, name="ci-reaper", daemon=True)
            thread.start()
        yield
        stop_event.set()

    app = FastAPI(
        title="KoeAxis CI", version=API_VERSION, lifespan=lifespan, root_path=settings.root_path,
        openapi_url="/api/v1/openapi.json", docs_url="/api/docs", redoc_url=None,
        description="字幕流水线 CI 服务：运行队列、worker 协议、产物与对比。",
    )
    app.state.settings, app.state.store, app.state.blobs, app.state.logs = settings, store, blob_store, logs
    app.state.reap_stale = reap_stale
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                           allow_methods=["*"], allow_headers=["*"])

    def notify(event: str, run: dict[str, Any] | None) -> None:
        if run:
            webhooks.send(event, {"run": run_public(run, with_position=False)})

    # ---------------------------------------------------------------------------------- auth deps
    def _client_ip(request: Request) -> str:
        return request.client.host if request.client else ""

    def principal_optional(request: Request) -> Principal | None:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = store.token_by_hash(hash_token(header[7:].strip()))
            if not token:
                raise HTTPException(401, "invalid token")
            if token["kind"] == "worker":
                return Principal(kind="worker", name=token["name"], token_id=token["id"], via="token")
            user = store.get_user(token["user_id"]) if token.get("user_id") else None
            if not user or user["disabled"]:
                raise HTTPException(401, "token owner disabled")
            return Principal(kind="user", name=user["username"], user=user, token_id=token["id"], via="token")
        if settings.trusted_user_header:
            name = request.headers.get(settings.trusted_user_header, "").strip()
            ip = _client_ip(request)
            if name and ip_trusted(ip, settings.trusted_proxy_ips):
                user = store.get_user_by_name(name)
                if not user and settings.trusted_header_auto_create and USERNAME_RE.match(name):
                    user = store.create_user(name, "", "member")
                    store.add_audit("system", "user.auto_create", name)
                if user and not user["disabled"]:
                    return Principal(kind="user", name=user["username"], user=user, via="header")
        cookie = request.cookies.get(settings.cookie_name)
        if cookie:
            user = store.session_user(hash_token(cookie))
            if user:
                if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get(CSRF_HEADER) != "koeaxis":
                    raise HTTPException(403, f"missing {CSRF_HEADER}: koeaxis header")
                return Principal(kind="user", name=user["username"], user=user, via="session")
        return None

    def current_user(principal: Principal | None = Depends(principal_optional)) -> Principal:
        if not principal or principal.kind != "user":
            raise HTTPException(401, "login required")
        return principal

    def admin_user(principal: Principal = Depends(current_user)) -> Principal:
        if not principal.is_admin:
            raise HTTPException(403, "admin only")
        return principal

    def current_worker(request: Request, principal: Principal | None = Depends(principal_optional)) -> Principal:
        if not principal or principal.kind != "worker":
            raise HTTPException(401, "worker token required")
        worker_id = request.headers.get(WORKER_HEADER, "").strip()
        if not worker_id or not re.match(r"^[A-Za-z0-9_.-]{1,64}$", worker_id):
            raise HTTPException(400, f"{WORKER_HEADER} header required")
        return Principal(kind="worker", name=worker_id, token_id=principal.token_id, via="token")

    def worker_run(run_id: str, worker: Principal = Depends(current_worker)) -> dict[str, Any]:
        run = store.get_run(run_id)
        if not run or run.get("worker_id") != worker.name:
            raise HTTPException(404, "run not assigned to this worker")
        return run

    def get_run_or_404(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        return run

    # ---------------------------------------------------------------------------------- serialization
    def run_public(run: dict[str, Any], *, with_position: bool = True) -> dict[str, Any]:
        data = dict(run)
        data["inputs"] = [{key: item.get(key) for key in ("index", "flag", "name", "size", "source", "blob_id")}
                          for item in run.get("inputs") or []]
        if with_position and run["status"] == "queued":
            data["queue_position"] = store.queue_position(run["id"])
        return data

    def save_upload(upload: UploadFile, actor: str) -> dict[str, Any]:
        try:
            sha, size = blob_store.put(upload.file, max_bytes=settings.max_upload_mb * 1024 * 1024)
        except BlobTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        return store.add_blob(secrets.token_hex(12),
                              pipeline.safe_filename(upload.filename or "upload"), size, sha,
                              upload.content_type or "application/octet-stream", actor)

    def save_file(path: Path, actor: str) -> dict[str, Any]:
        with path.open("rb") as handle:
            sha, size = blob_store.put(handle)
        return store.add_blob(secrets.token_hex(12),
                              pipeline.safe_filename(path.name), size, sha, "application/octet-stream", actor)

    def blob_response(blob: dict[str, Any], filename: str | None = None) -> Response:
        name = filename or blob["name"]
        local = blob_store.local_path(blob["sha256"])
        if local is not None:
            return FileResponse(local, filename=name, media_type=blob.get("content_type") or "application/octet-stream")

        def stream():
            with blob_store.open(blob["sha256"]) as handle:
                while chunk := handle.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(stream(), media_type="application/octet-stream",
                                 headers={"Content-Disposition": f"attachment; filename*=UTF-8''{name}"})

    def presets() -> dict[str, pipeline.Preset]:
        return pipeline.discover_presets(settings.presets_dirs)

    def snapshot_inputs(preset: pipeline.Preset, user_inputs: list[InputRef], actor: str) -> list[dict[str, Any]]:
        """Copy preset files into the blob store so the run is reproducible even if the preset changes."""
        inputs: list[dict[str, Any]] = []
        for item in preset.files:
            if not item.path.is_file():
                raise HTTPException(500, f"preset file missing on server: {item.name}")
            blob = save_file(item.path, actor)
            inputs.append({"flag": item.flag, "blob_id": blob["id"], "name": blob["name"], "sha256": blob["sha256"],
                           "size": blob["size"], "source": "preset"})
        for ref in user_inputs:
            if ref.flag not in settings.input_flags:
                raise HTTPException(400, f"input flag {ref.flag} not allowed; allowed: {settings.input_flags}")
            blob = store.get_blob(ref.blob_id)
            if not blob:
                raise HTTPException(400, f"unknown blob {ref.blob_id}")
            # a user file replaces the preset file passed with the same flag
            inputs = [entry for entry in inputs if entry["flag"] != ref.flag]
            inputs.append({"flag": ref.flag, "blob_id": blob["id"], "name": blob["name"], "sha256": blob["sha256"],
                           "size": blob["size"], "source": "user"})
        for index, entry in enumerate(inputs):
            entry["index"] = index
        return inputs

    def check_extra_args(principal: Principal, extra_args: list[str]) -> list[str]:
        if extra_args and not (principal.is_admin or settings.member_extra_args):
            raise HTTPException(403, "extra_args require admin (or KOEAXIS_CI_MEMBER_EXTRA_ARGS=1)")
        if any(not isinstance(item, str) or len(item) > 500 or "\n" in item for item in extra_args):
            raise HTTPException(400, "invalid extra_args")
        if any(item in {"--workdir", "--media", "--export-path", "--export-mode"} for item in extra_args):
            raise HTTPException(400, "extra_args may not override --workdir/--media/--export-*")
        return list(extra_args)

    # ---------------------------------------------------------------------------------- public
    @app.get("/api/v1/health", tags=["meta"])
    def health() -> dict[str, Any]:
        return {"ok": True, "version": API_VERSION, "time": utcnow().isoformat(timespec="seconds")}

    @app.get("/api/v1/meta", tags=["meta"])
    def meta(_: Principal = Depends(current_user)) -> dict[str, Any]:
        return {
            "stages": [{"name": name, "phase": phase} for name, phase in pipeline.STAGES],
            "retry_stages": list(pipeline.RETRY_STAGES),
            "options": sorted([*pipeline.OPTION_FLAGS, "language"]),
            "input_flags": settings.input_flags,
            "max_upload_mb": settings.max_upload_mb,
            "member_extra_args": settings.member_extra_args,
        }

    @app.post("/api/v1/auth/login", tags=["auth"])
    def login(body: LoginBody, request: Request, response: Response) -> dict[str, Any]:
        key = f"{_client_ip(request)}|{body.username}"
        if throttle.blocked(key):
            raise HTTPException(429, "too many failed logins, try later")
        user = store.get_user_by_name(body.username)
        if not user or user["disabled"] or not verify_password(body.password, user.get("password_hash") or ""):
            throttle.fail(key)
            raise HTTPException(401, "用户名或密码错误")
        throttle.reset(key)
        token = new_token("session")
        ttl = timedelta(hours=settings.session_ttl_hours)
        store.create_session(hash_token(token), user["id"], ttl)
        response.set_cookie(settings.cookie_name, token, max_age=int(ttl.total_seconds()), httponly=True,
                            samesite="lax", secure=settings.cookie_secure, path="/")
        store.add_audit(user["username"], "auth.login", detail={"ip": _client_ip(request)})
        return {"user": _public_user(user)}

    @app.post("/api/v1/auth/logout", tags=["auth"])
    def logout(request: Request, response: Response) -> dict[str, Any]:
        cookie = request.cookies.get(settings.cookie_name)
        if cookie:
            store.delete_session(hash_token(cookie))
        response.delete_cookie(settings.cookie_name, path="/")
        return {"ok": True}

    @app.get("/api/v1/me", tags=["auth"])
    def me(principal: Principal = Depends(current_user)) -> dict[str, Any]:
        return {"user": _public_user(principal.user or {}), "via": principal.via}

    @app.post("/api/v1/me/password", tags=["auth"])
    def change_password(body: PasswordBody, principal: Principal = Depends(current_user)) -> dict[str, Any]:
        user = principal.user or {}
        if user.get("password_hash") and not verify_password(body.old_password, user["password_hash"]):
            raise HTTPException(400, "旧密码不正确")
        try:
            store.update_user(user["id"], password_hash=hash_password(body.new_password))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        store.add_audit(principal.name, "user.password")
        return {"ok": True}

    @app.get("/api/v1/me/tokens", tags=["auth"])
    def my_tokens(principal: Principal = Depends(current_user)) -> list[dict[str, Any]]:
        return [token for token in store.list_tokens() if token.get("user_id") == principal.user["id"]]

    @app.post("/api/v1/me/tokens", tags=["auth"])
    def create_my_token(body: TokenCreate, principal: Principal = Depends(current_user)) -> dict[str, Any]:
        token = new_token("user")
        record = store.create_token(body.name[:128], hash_token(token), "user", principal.user["id"])
        store.add_audit(principal.name, "token.create", str(record["id"]), {"kind": "user"})
        return {"token": token, **record}

    @app.delete("/api/v1/me/tokens/{token_id}", tags=["auth"])
    def revoke_my_token(token_id: int, principal: Principal = Depends(current_user)) -> dict[str, Any]:
        token = store.get_token(token_id)
        if not token or token.get("user_id") != principal.user["id"]:
            raise HTTPException(404, "token not found")
        store.revoke_token(token_id)
        return {"ok": True}

    # ---------------------------------------------------------------------------------- presets / blobs
    @app.get("/api/v1/presets", tags=["runs"])
    def list_presets(_: Principal = Depends(current_user)) -> list[dict[str, Any]]:
        return [preset.public() for preset in presets().values()]

    @app.post("/api/v1/blobs", tags=["runs"])
    def upload_blob(file: UploadFile = File(...), principal: Principal = Depends(current_user)) -> dict[str, Any]:
        return save_upload(file, principal.name)

    # ---------------------------------------------------------------------------------- runs
    @app.get("/api/v1/runs", tags=["runs"])
    def list_runs(status: str | None = None, preset_id: str | None = None, q: str | None = None,
                  mine: bool = False, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                  principal: Principal = Depends(current_user)) -> dict[str, Any]:
        items, total = store.list_runs(status=status, preset_id=preset_id, query=q,
                                       created_by=principal.name if mine else None, limit=limit, offset=offset)
        return {"items": [run_public(item) for item in items], "total": total}

    @app.post("/api/v1/runs", tags=["runs"], status_code=201)
    def create_run(body: RunCreate, principal: Principal = Depends(current_user)) -> dict[str, Any]:
        preset = presets().get(body.preset_id)
        if not preset:
            raise HTTPException(400, f"unknown preset {body.preset_id}")
        media = store.get_blob(body.media_blob_id)
        if not media:
            raise HTTPException(400, "unknown media blob")
        try:
            options = pipeline.validate_options(body.options)
        except pipeline.PresetError as exc:
            raise HTTPException(400, str(exc)) from exc
        extra_args = check_extra_args(principal, body.extra_args)
        run = store.create_run(
            title=(body.title or Path(media["name"]).stem)[:500], preset_id=preset.id, preset=preset.public(),
            options=options, extra_args=extra_args, inputs=snapshot_inputs(preset, body.inputs, principal.name),
            labels=body.labels, media_blob_id=media["id"], priority=body.priority, created_by=principal.name,
        )
        logs.append(run["id"], f"[ci] queued by {principal.name}: preset={preset.id} media={media['name']}\n")
        store.add_audit(principal.name, "run.create", run["id"])
        notify("run.queued", run)
        return run_public(run)

    @app.get("/api/v1/runs/{run_id}", tags=["runs"])
    def get_run(run_id: str, _: Principal = Depends(current_user)) -> dict[str, Any]:
        run = get_run_or_404(run_id)
        data = run_public(run)
        data["artifacts"] = store.list_artifacts(run_id)
        data["media"] = store.get_blob(run["media_blob_id"])
        data["log_size"] = logs.size(run_id)
        return data

    @app.patch("/api/v1/runs/{run_id}", tags=["runs"])
    def patch_run(run_id: str, body: RunPatch, principal: Principal = Depends(current_user)) -> dict[str, Any]:
        get_run_or_404(run_id)
        run = store.update_run(run_id, title=body.title, labels=body.labels, priority=body.priority)
        store.add_audit(principal.name, "run.update", run_id)
        return run_public(run)

    @app.post("/api/v1/runs/{run_id}/cancel", tags=["runs"])
    def cancel_run(run_id: str, principal: Principal = Depends(current_user)) -> dict[str, Any]:
        before = get_run_or_404(run_id)
        run = store.request_cancel(run_id)
        logs.append(run_id, f"[ci] cancel requested by {principal.name}\n")
        store.add_audit(principal.name, "run.cancel", run_id)
        if before["status"] == "queued":
            notify("run.finished", run)
        return run_public(run)

    @app.post("/api/v1/runs/{run_id}/retry", tags=["runs"], status_code=201)
    def retry_run(run_id: str, body: RunRetry, principal: Principal = Depends(current_user)) -> dict[str, Any]:
        parent = get_run_or_404(run_id)
        if parent["status"] not in FINAL_STATUSES:
            raise HTTPException(409, "run is still active; cancel it first")
        if body.reset_from and body.reset_from not in pipeline.RETRY_STAGES:
            raise HTTPException(400, f"reset_from must be one of {pipeline.RETRY_STAGES}")
        preset_public = parent["preset"]
        inputs = parent["inputs"]
        if body.refresh_preset:
            preset = presets().get(parent["preset_id"])
            if not preset:
                raise HTTPException(400, "preset no longer exists")
            preset_public = preset.public()
            user_inputs = [InputRef(flag=item["flag"], blob_id=item["blob_id"]) for item in inputs
                           if item.get("source") == "user"]
            inputs = snapshot_inputs(preset, user_inputs, principal.name)
        try:
            options = pipeline.validate_options({**parent["options"], **(body.options or {})})
        except pipeline.PresetError as exc:
            raise HTTPException(400, str(exc)) from exc
        extra_args = parent["extra_args"] if body.extra_args is None else check_extra_args(principal, body.extra_args)
        run = store.create_run(
            title=(body.title or parent["title"])[:500], preset_id=parent["preset_id"], preset=preset_public,
            options=options, extra_args=extra_args, inputs=inputs, labels=parent["labels"],
            media_blob_id=parent["media_blob_id"], priority=parent["priority"], created_by=principal.name,
            parent_run_id=parent["id"],
            workspace_key=None if body.fresh_workspace else parent["workspace_key"],
            worker_hint=None if body.fresh_workspace else parent.get("worker_id"),
            reset_from=None if body.fresh_workspace else body.reset_from,
        )
        logs.append(run["id"], f"[ci] retry of {parent['id']} by {principal.name}; reset_from={body.reset_from}\n")
        store.add_audit(principal.name, "run.retry", run["id"], {"parent": parent["id"], "reset_from": body.reset_from})
        notify("run.queued", run)
        return run_public(run)

    @app.delete("/api/v1/runs/{run_id}", tags=["runs"])
    def delete_run(run_id: str, principal: Principal = Depends(current_user)) -> dict[str, Any]:
        run = get_run_or_404(run_id)
        if run["status"] not in FINAL_STATUSES:
            raise HTTPException(409, "cancel the run before deleting it")
        if run["created_by"] != principal.name and not principal.is_admin:
            raise HTTPException(403, "only the creator or an admin may delete a run")
        for blob_id in store.delete_run(run_id):
            if not store.blob_in_use(blob_id):
                sha = store.delete_blob(blob_id)
                if sha:
                    blob_store.delete(sha)
        logs.delete(run_id)
        store.add_audit(principal.name, "run.delete", run_id)
        return {"ok": True}

    @app.get("/api/v1/runs/{run_id}/log", tags=["runs"])
    def read_log(run_id: str, offset: int = Query(0, ge=0), limit: int = Query(256 * 1024, ge=1, le=4 * 1024 * 1024),
                 _: Principal = Depends(current_user)) -> dict[str, Any]:
        run = get_run_or_404(run_id)
        text, next_offset = logs.read(run_id, offset, limit)
        return {"text": text, "offset": offset, "next_offset": next_offset, "size": logs.size(run_id),
                "complete": run["status"] in FINAL_STATUSES}

    @app.get("/api/v1/runs/{run_id}/log.txt", tags=["runs"])
    def download_log(run_id: str, _: Principal = Depends(current_user)) -> Response:
        get_run_or_404(run_id)
        path = logs.path(run_id)
        if not path.exists():
            raise HTTPException(404, "no log yet")
        return FileResponse(path, filename=f"{run_id}.log", media_type="text/plain; charset=utf-8")

    @app.get("/api/v1/runs/{run_id}/artifacts", tags=["runs"])
    def list_artifacts(run_id: str, _: Principal = Depends(current_user)) -> list[dict[str, Any]]:
        get_run_or_404(run_id)
        return store.list_artifacts(run_id)

    @app.get("/api/v1/runs/{run_id}/artifacts/{name}", tags=["runs"])
    def download_artifact(run_id: str, name: str, _: Principal = Depends(current_user)) -> Response:
        artifact = store.get_artifact(run_id, name)
        blob = store.get_blob(artifact["blob_id"]) if artifact else None
        if not blob:
            raise HTTPException(404, "artifact not found")
        return blob_response(blob, artifact["name"])

    @app.get("/api/v1/runs/{run_id}/media", tags=["runs"])
    def download_media(run_id: str, _: Principal = Depends(current_user)) -> Response:
        run = get_run_or_404(run_id)
        blob = store.get_blob(run["media_blob_id"])
        if not blob:
            raise HTTPException(404, "media missing")
        return blob_response(blob)

    @app.get("/api/v1/compare", tags=["runs"])
    def compare(a: str, b: str, artifact: str | None = None, _: Principal = Depends(current_user)) -> dict[str, Any]:
        run_a, run_b = get_run_or_404(a), get_run_or_404(b)

        def stage_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
            return {item.get("name"): item for item in run.get("stages") or [] if isinstance(item, dict)}

        stages_a, stages_b = stage_map(run_a), stage_map(run_b)
        stages = [{"name": name, "a": stages_a.get(name), "b": stages_b.get(name)}
                  for name in pipeline.STAGE_NAMES if name in stages_a or name in stages_b]
        checks_a = {item.get("name"): item for item in run_a.get("checks") or []}
        checks_b = {item.get("name"): item for item in run_b.get("checks") or []}
        checks = [{"name": name, "a": checks_a.get(name), "b": checks_b.get(name)}
                  for name in dict.fromkeys([*checks_a, *checks_b])]
        metric_keys = dict.fromkeys([*(run_a.get("metrics") or {}), *(run_b.get("metrics") or {})])
        metrics = [{"name": key, "a": (run_a.get("metrics") or {}).get(key), "b": (run_b.get("metrics") or {}).get(key)}
                   for key in metric_keys]

        def subtitle(run: dict[str, Any]) -> dict[str, Any] | None:
            items = store.list_artifacts(run["id"])
            if artifact:
                return next((item for item in items if item["name"] == artifact), None)
            return next((item for item in items if item["kind"] == "subtitle"), None)

        srt = None
        art_a, art_b = subtitle(run_a), subtitle(run_b)
        if art_a and art_b:
            blob_a, blob_b = store.get_blob(art_a["blob_id"]), store.get_blob(art_b["blob_id"])
            with blob_store.open(blob_a["sha256"]) as handle:
                cues_a = _srt_texts(handle.read())
            with blob_store.open(blob_b["sha256"]) as handle:
                cues_b = _srt_texts(handle.read())
            matcher = difflib.SequenceMatcher(a=cues_a, b=cues_b, autojunk=False)
            diff = list(difflib.unified_diff(cues_a, cues_b, art_a["name"], art_b["name"], n=1, lineterm=""))
            srt = {"a": art_a["name"], "b": art_b["name"], "a_count": len(cues_a), "b_count": len(cues_b),
                   "similarity": round(matcher.ratio(), 4), "diff": diff[:4000], "truncated": len(diff) > 4000}
        return {"a": run_public(run_a), "b": run_public(run_b), "stages": stages, "checks": checks,
                "metrics": metrics, "srt": srt}

    # ---------------------------------------------------------------------------------- admin
    @app.get("/api/v1/admin/users", tags=["admin"])
    def admin_users(_: Principal = Depends(admin_user)) -> list[dict[str, Any]]:
        return [_public_user(user) for user in store.list_users()]

    @app.post("/api/v1/admin/users", tags=["admin"], status_code=201)
    def admin_create_user(body: UserCreate, principal: Principal = Depends(admin_user)) -> dict[str, Any]:
        if not USERNAME_RE.match(body.username):
            raise HTTPException(400, "invalid username")
        if store.get_user_by_name(body.username):
            raise HTTPException(409, "user exists")
        try:
            user = store.create_user(body.username, hash_password(body.password), body.role, body.display_name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        store.add_audit(principal.name, "user.create", body.username, {"role": body.role})
        return _public_user(user)

    @app.patch("/api/v1/admin/users/{user_id}", tags=["admin"])
    def admin_patch_user(user_id: int, body: UserPatch, principal: Principal = Depends(admin_user)) -> dict[str, Any]:
        if not store.get_user(user_id):
            raise HTTPException(404, "user not found")
        try:
            user = store.update_user(user_id, display_name=body.display_name, role=body.role, disabled=body.disabled,
                                     password_hash=hash_password(body.password) if body.password else None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        store.add_audit(principal.name, "user.update", str(user_id),
                        body.model_dump(exclude_none=True, exclude={"password"}))
        return _public_user(user)

    @app.get("/api/v1/admin/tokens", tags=["admin"])
    def admin_tokens(_: Principal = Depends(admin_user)) -> list[dict[str, Any]]:
        return store.list_tokens()

    @app.post("/api/v1/admin/tokens", tags=["admin"], status_code=201)
    def admin_create_token(body: TokenCreate, principal: Principal = Depends(admin_user)) -> dict[str, Any]:
        if body.kind not in {"worker", "user"}:
            raise HTTPException(400, "kind must be worker or user")
        user_id = None
        if body.kind == "user":
            user = store.get_user_by_name(body.username or principal.name)
            if not user:
                raise HTTPException(400, "unknown user")
            user_id = user["id"]
        token = new_token(body.kind)
        record = store.create_token(body.name[:128], hash_token(token), body.kind, user_id)
        store.add_audit(principal.name, "token.create", str(record["id"]), {"kind": body.kind})
        return {"token": token, **record}

    @app.delete("/api/v1/admin/tokens/{token_id}", tags=["admin"])
    def admin_revoke_token(token_id: int, principal: Principal = Depends(admin_user)) -> dict[str, Any]:
        store.revoke_token(token_id)
        store.add_audit(principal.name, "token.revoke", str(token_id))
        return {"ok": True}

    @app.get("/api/v1/workers", tags=["admin"])
    def list_workers(_: Principal = Depends(current_user)) -> list[dict[str, Any]]:
        running, _total = store.list_runs(status="running", limit=500)
        result = []
        for worker in store.list_workers():
            worker["running"] = [run["id"] for run in running if run.get("worker_id") == worker["id"]]
            result.append(worker)
        return result

    @app.get("/api/v1/admin/audit", tags=["admin"])
    def admin_audit(limit: int = Query(200, ge=1, le=2000), _: Principal = Depends(admin_user)) -> list[dict[str, Any]]:
        return store.list_audit(limit)

    @app.post("/api/v1/admin/reap", tags=["admin"])
    def admin_reap(_: Principal = Depends(admin_user)) -> dict[str, Any]:
        return {"reaped": reap_stale()}

    # ---------------------------------------------------------------------------------- worker protocol
    def worker_payload(run: dict[str, Any]) -> dict[str, Any]:
        media = store.get_blob(run["media_blob_id"]) or {}
        base = f"/api/v1/worker/runs/{run['id']}"
        return {
            "id": run["id"], "title": run["title"], "preset_id": run["preset_id"], "preset": run["preset"],
            "options": run["options"], "extra_args": run["extra_args"], "workspace_key": run["workspace_key"],
            "reset_from": run.get("reset_from"), "parent_run_id": run.get("parent_run_id"),
            "env": (run.get("preset") or {}).get("env") or {},
            "media": {"name": media.get("name"), "sha256": media.get("sha256"), "size": media.get("size"),
                      "url": f"{base}/media"},
            "inputs": [{**{key: item.get(key) for key in ("index", "flag", "name", "sha256", "size", "source")},
                        "url": f"{base}/inputs/{item['index']}"} for item in run.get("inputs") or []],
        }

    @app.post("/api/v1/worker/claim", tags=["worker"])
    def worker_claim(body: ClaimBody, worker: Principal = Depends(current_worker)) -> Response:
        store.touch_worker(worker.name, body.name or None, worker.token_id, body.info)
        run = store.claim_run(worker.name, preset_ids=body.preset_ids,
                              affinity=timedelta(seconds=settings.worker_affinity_seconds))
        if not run:
            return Response(status_code=204)
        logs.append(run["id"], f"[ci] claimed by worker {worker.name}\n")
        notify("run.started", run)
        return JSONResponse({"run": worker_payload(run)})

    @app.get("/api/v1/worker/runs/{run_id}", tags=["worker"])
    def worker_get_run(run: dict[str, Any] = Depends(worker_run)) -> dict[str, Any]:
        return {"run": worker_payload(run), "cancel_requested": bool(run["cancel_requested"])}

    @app.get("/api/v1/worker/runs/{run_id}/media", tags=["worker"])
    def worker_media(run: dict[str, Any] = Depends(worker_run)) -> Response:
        blob = store.get_blob(run["media_blob_id"])
        if not blob:
            raise HTTPException(404, "media missing")
        return blob_response(blob)

    @app.get("/api/v1/worker/runs/{run_id}/inputs/{index}", tags=["worker"])
    def worker_input(index: int, run: dict[str, Any] = Depends(worker_run)) -> Response:
        items = run.get("inputs") or []
        if not 0 <= index < len(items):
            raise HTTPException(404, "input not found")
        blob = store.get_blob(items[index]["blob_id"])
        if not blob:
            raise HTTPException(404, "input blob missing")
        return blob_response(blob)

    @app.post("/api/v1/worker/runs/{run_id}/heartbeat", tags=["worker"])
    def worker_heartbeat(body: HeartbeatBody, run: dict[str, Any] = Depends(worker_run),
                         worker: Principal = Depends(current_worker)) -> dict[str, Any]:
        if run["status"] != "running":
            return {"cancel_requested": True, "status": run["status"]}
        values = {key: value for key, value in body.model_dump().items() if value is not None}
        store.update_run(run["id"], heartbeat_at=utcnow(), **values)
        store.touch_worker(worker.name, None, worker.token_id, None)
        return {"cancel_requested": bool(run["cancel_requested"]), "status": run["status"]}

    @app.post("/api/v1/worker/runs/{run_id}/log", tags=["worker"])
    async def worker_log(request: Request, run: dict[str, Any] = Depends(worker_run)) -> dict[str, Any]:
        body = await request.body()
        if len(body) > LOG_APPEND_LIMIT:
            raise HTTPException(413, "log chunk too large")
        size = logs.append(run["id"], body.decode("utf-8", errors="replace"))
        return {"size": size}

    @app.post("/api/v1/worker/runs/{run_id}/artifacts", tags=["worker"])
    def worker_artifact(file: UploadFile = File(...), name: str = Form(""), kind: str = Form("file"),
                        run: dict[str, Any] = Depends(worker_run), worker: Principal = Depends(current_worker)
                        ) -> dict[str, Any]:
        blob = save_upload(file, f"worker:{worker.name}")
        artifact_name = pipeline.safe_filename(name or blob["name"])
        return store.add_artifact(run["id"], artifact_name, kind[:32] or "file", blob["id"], blob["size"])

    @app.post("/api/v1/worker/runs/{run_id}/complete", tags=["worker"])
    def worker_complete(body: CompleteBody, run: dict[str, Any] = Depends(worker_run)) -> dict[str, Any]:
        if body.status not in FINAL_STATUSES:
            raise HTTPException(400, f"status must be one of {FINAL_STATUSES}")
        if run["status"] in FINAL_STATUSES:
            return run_public(run)
        status = "cancelled" if run["cancel_requested"] and body.status != "succeeded" else body.status
        values = {key: value for key, value in body.model_dump(exclude={"status"}).items() if value is not None}
        updated = store.update_run(run["id"], status=status, finished_at=utcnow(), heartbeat_at=utcnow(), **values)
        logs.append(run["id"], f"[ci] finished: {status}\n")
        notify("run.finished", updated)
        return run_public(updated)

    # ---------------------------------------------------------------------------------- web UI
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> Response:
            return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
