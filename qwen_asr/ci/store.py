"""Persistent state of the CI server: users, sessions, tokens, blobs, runs, artifacts, workers, audit log.

Built on SQLAlchemy Core so any SQLAlchemy URL works (SQLite by default, PostgreSQL/MySQL for larger
deployments). JSON fields are stored as text for portability.
"""
from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    event,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.engine import Engine

RUN_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
FINAL_STATUSES = ("succeeded", "failed", "cancelled")
ROLES = ("admin", "member")
TOKEN_KINDS = ("worker", "user")

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(64), unique=True, nullable=False),
    Column("display_name", String(128), nullable=False, default=""),
    Column("password_hash", String(256), nullable=False, default=""),
    Column("role", String(16), nullable=False, default="member"),
    Column("disabled", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
sessions = Table(
    "sessions", metadata,
    Column("token_hash", String(64), primary_key=True),
    Column("user_id", Integer, nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
)
tokens = Table(
    "tokens", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(128), nullable=False),
    Column("token_hash", String(64), unique=True, nullable=False),
    Column("kind", String(16), nullable=False),
    Column("user_id", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_used_at", DateTime(timezone=True), nullable=True),
    Column("revoked", Boolean, nullable=False, default=False),
)
blobs = Table(
    "blobs", metadata,
    Column("id", String(40), primary_key=True),
    Column("name", String(512), nullable=False),
    Column("size", Integer, nullable=False),
    Column("sha256", String(64), nullable=False, index=True),
    Column("content_type", String(128), nullable=False, default="application/octet-stream"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("created_by", String(128), nullable=False, default=""),
)
runs = Table(
    "runs", metadata,
    Column("id", String(32), primary_key=True),
    Column("title", String(512), nullable=False),
    Column("preset_id", String(64), nullable=False, index=True),
    Column("preset_json", Text, nullable=False, default="{}"),
    Column("options_json", Text, nullable=False, default="{}"),
    Column("extra_args_json", Text, nullable=False, default="[]"),
    Column("inputs_json", Text, nullable=False, default="[]"),
    Column("labels_json", Text, nullable=False, default="{}"),
    Column("media_blob_id", String(40), nullable=False),
    Column("status", String(16), nullable=False, index=True),
    Column("priority", Integer, nullable=False, default=0),
    Column("created_by", String(128), nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("heartbeat_at", DateTime(timezone=True), nullable=True),
    Column("worker_id", String(64), nullable=True),
    Column("worker_hint", String(64), nullable=True),   # retries prefer the worker that holds the workspace
    Column("cancel_requested", Boolean, nullable=False, default=False),
    Column("parent_run_id", String(32), nullable=True),
    Column("workspace_key", String(64), nullable=False),
    Column("reset_from", String(32), nullable=True),
    Column("stages_json", Text, nullable=False, default="[]"),
    Column("metrics_json", Text, nullable=False, default="{}"),
    Column("checks_json", Text, nullable=False, default="[]"),
    Column("summary", Text, nullable=False, default=""),
    Column("error", Text, nullable=False, default=""),
)
artifacts = Table(
    "artifacts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), nullable=False, index=True),
    Column("name", String(512), nullable=False),
    Column("kind", String(32), nullable=False, default="file"),
    Column("blob_id", String(40), nullable=False),
    Column("size", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
workers = Table(
    "workers", metadata,
    Column("id", String(64), primary_key=True),
    Column("name", String(128), nullable=False, default=""),
    Column("token_id", Integer, nullable=True),
    Column("last_seen_at", DateTime(timezone=True), nullable=False),
    Column("info_json", Text, nullable=False, default="{}"),
)
audit = Table(
    "audit", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", DateTime(timezone=True), nullable=False),
    Column("actor", String(128), nullable=False),
    Column("action", String(64), nullable=False),
    Column("target", String(128), nullable=False, default=""),
    Column("detail_json", Text, nullable=False, default="{}"),
)

_JSON_FIELDS = {
    "preset_json": "preset", "options_json": "options", "extra_args_json": "extra_args", "inputs_json": "inputs",
    "labels_json": "labels", "stages_json": "stages", "metrics_json": "metrics", "checks_json": "checks",
    "info_json": "info", "detail_json": "detail",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _row(row: Any) -> dict[str, Any]:
    data = dict(row._mapping)
    for column, key in _JSON_FIELDS.items():
        if column in data:
            try:
                data[key] = json.loads(data.pop(column) or "null")
            except ValueError:
                data[key] = None
    for key, value in list(data.items()):
        if isinstance(value, datetime):
            data[key] = _aware(value).isoformat(timespec="seconds")
    return data


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class Store:
    def __init__(self, url: str, *, engine: Engine | None = None) -> None:
        self.engine = engine or create_engine(url, future=True, pool_pre_ping=True,
                                              connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
        if self.engine.dialect.name == "sqlite":
            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_connection, _record):  # noqa: ANN001
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=10000")
                cursor.close()
        metadata.create_all(self.engine)
        self._claim_lock = threading.Lock()

    # ---------------------------------------------------------------- users / sessions / tokens
    def user_count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(users)).scalar_one())

    def create_user(self, username: str, password_hash: str, role: str = "member", display_name: str = "") -> dict[str, Any]:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        with self.engine.begin() as conn:
            result = conn.execute(insert(users).values(
                username=username, display_name=display_name or username, password_hash=password_hash,
                role=role, disabled=False, created_at=utcnow()))
            user_id = result.inserted_primary_key[0]
        return self.get_user(user_id)

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(users).where(users.c.id == user_id)).first()
        return _row(row) if row else None

    def get_user_by_name(self, username: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(users).where(users.c.username == username)).first()
        return _row(row) if row else None

    def list_users(self) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(users).order_by(users.c.id)).all()
        return [_row(row) for row in rows]

    def update_user(self, user_id: int, **values: Any) -> dict[str, Any] | None:
        allowed = {key: value for key, value in values.items()
                   if key in {"display_name", "password_hash", "role", "disabled"} and value is not None}
        if "role" in allowed and allowed["role"] not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if allowed:
            with self.engine.begin() as conn:
                conn.execute(update(users).where(users.c.id == user_id).values(**allowed))
                if allowed.get("disabled") or "password_hash" in allowed:
                    conn.execute(delete(sessions).where(sessions.c.user_id == user_id))
        return self.get_user(user_id)

    def create_session(self, token_hash: str, user_id: int, ttl: timedelta) -> None:
        now = utcnow()
        with self.engine.begin() as conn:
            conn.execute(delete(sessions).where(sessions.c.expires_at < now))
            conn.execute(insert(sessions).values(token_hash=token_hash, user_id=user_id, created_at=now,
                                                 expires_at=now + ttl))

    def session_user(self, token_hash: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(sessions).where(sessions.c.token_hash == token_hash)).first()
        if not row or _aware(row.expires_at) < utcnow():
            return None
        user = self.get_user(row.user_id)
        return user if user and not user["disabled"] else None

    def delete_session(self, token_hash: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(sessions).where(sessions.c.token_hash == token_hash))

    def create_token(self, name: str, token_hash: str, kind: str, user_id: int | None = None) -> dict[str, Any]:
        if kind not in TOKEN_KINDS:
            raise ValueError(f"kind must be one of {TOKEN_KINDS}")
        with self.engine.begin() as conn:
            result = conn.execute(insert(tokens).values(name=name, token_hash=token_hash, kind=kind, user_id=user_id,
                                                        created_at=utcnow(), revoked=False))
            token_id = result.inserted_primary_key[0]
        return self.get_token(token_id)

    def get_token(self, token_id: int) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(tokens).where(tokens.c.id == token_id)).first()
        if not row:
            return None
        data = _row(row)
        data.pop("token_hash", None)
        return data

    def token_by_hash(self, token_hash: str) -> dict[str, Any] | None:
        with self.engine.begin() as conn:
            row = conn.execute(select(tokens).where(and_(tokens.c.token_hash == token_hash,
                                                         tokens.c.revoked.is_(False)))).first()
            if not row:
                return None
            conn.execute(update(tokens).where(tokens.c.id == row.id).values(last_used_at=utcnow()))
        data = _row(row)
        data.pop("token_hash", None)
        return data

    def list_tokens(self) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(tokens).order_by(tokens.c.id)).all()
        result = []
        for row in rows:
            data = _row(row)
            data.pop("token_hash", None)
            result.append(data)
        return result

    def revoke_token(self, token_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(tokens).where(tokens.c.id == token_id).values(revoked=True))

    # ---------------------------------------------------------------- blobs
    def add_blob(self, blob_id: str, name: str, size: int, sha256: str, content_type: str, created_by: str) -> dict[str, Any]:
        with self.engine.begin() as conn:
            conn.execute(insert(blobs).values(id=blob_id, name=name, size=size, sha256=sha256,
                                              content_type=content_type, created_at=utcnow(), created_by=created_by))
        return self.get_blob(blob_id)

    def get_blob(self, blob_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(blobs).where(blobs.c.id == blob_id)).first()
        return _row(row) if row else None

    def blob_by_sha(self, sha256: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(blobs).where(blobs.c.sha256 == sha256).limit(1)).first()
        return _row(row) if row else None

    def delete_blob(self, blob_id: str) -> str | None:
        """Delete a blob row; return its sha256 when no other row/run references the content any more."""
        with self.engine.begin() as conn:
            row = conn.execute(select(blobs).where(blobs.c.id == blob_id)).first()
            if not row:
                return None
            conn.execute(delete(blobs).where(blobs.c.id == blob_id))
            still_used = conn.execute(select(func.count()).select_from(blobs).where(blobs.c.sha256 == row.sha256)).scalar_one()
        return None if still_used else row.sha256

    def blob_in_use(self, blob_id: str) -> bool:
        with self.engine.connect() as conn:
            if conn.execute(select(runs.c.id).where(runs.c.media_blob_id == blob_id).limit(1)).first():
                return True
            if conn.execute(select(artifacts.c.id).where(artifacts.c.blob_id == blob_id).limit(1)).first():
                return True
            like = f'%"{blob_id}"%'
            return conn.execute(select(runs.c.id).where(runs.c.inputs_json.like(like)).limit(1)).first() is not None

    # ---------------------------------------------------------------- runs
    def create_run(self, **values: Any) -> dict[str, Any]:
        run_id = values.pop("id", None) or secrets.token_hex(6)
        row = {
            "id": run_id,
            "title": values["title"],
            "preset_id": values["preset_id"],
            "preset_json": _dumps(values.get("preset") or {}),
            "options_json": _dumps(values.get("options") or {}),
            "extra_args_json": _dumps(values.get("extra_args") or []),
            "inputs_json": _dumps(values.get("inputs") or []),
            "labels_json": _dumps(values.get("labels") or {}),
            "media_blob_id": values["media_blob_id"],
            "status": "queued",
            "priority": int(values.get("priority") or 0),
            "created_by": values.get("created_by") or "",
            "created_at": utcnow(),
            "parent_run_id": values.get("parent_run_id"),
            "workspace_key": values.get("workspace_key") or run_id,
            "worker_hint": values.get("worker_hint"),
            "reset_from": values.get("reset_from"),
            "cancel_requested": False,
        }
        with self.engine.begin() as conn:
            conn.execute(insert(runs).values(**row))
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(runs).where(runs.c.id == run_id)).first()
        return _row(row) if row else None

    def list_runs(self, *, status: str | None = None, preset_id: str | None = None, query: str | None = None,
                  created_by: str | None = None, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        conditions = []
        if status:
            conditions.append(runs.c.status.in_(status.split(",")))
        if preset_id:
            conditions.append(runs.c.preset_id == preset_id)
        if created_by:
            conditions.append(runs.c.created_by == created_by)
        if query:
            like = f"%{query}%"
            conditions.append(or_(runs.c.title.like(like), runs.c.id.like(like)))
        where = and_(*conditions) if conditions else None
        statement = select(runs).order_by(runs.c.created_at.desc()).limit(max(1, min(limit, 500))).offset(max(0, offset))
        count = select(func.count()).select_from(runs)
        if where is not None:
            statement = statement.where(where)
            count = count.where(where)
        with self.engine.connect() as conn:
            rows = conn.execute(statement).all()
            total = int(conn.execute(count).scalar_one())
        return [_row(row) for row in rows], total

    def queue_position(self, run_id: str) -> int | None:
        run = self.get_run(run_id)
        if not run or run["status"] != "queued":
            return None
        with self.engine.connect() as conn:
            ahead = conn.execute(select(func.count()).select_from(runs).where(and_(
                runs.c.status == "queued",
                or_(runs.c.priority > run["priority"],
                    and_(runs.c.priority == run["priority"], runs.c.created_at < datetime.fromisoformat(run["created_at"])))
            ))).scalar_one()
        return int(ahead) + 1

    def claim_run(self, worker_id: str, *, preset_ids: list[str] | None = None,
                  affinity: timedelta = timedelta(seconds=300)) -> dict[str, Any] | None:
        """Atomically move the best queued run to `running` for this worker.

        Order: priority desc, then age. A run with a `worker_hint` for another worker (a retry that can
        reuse that worker's workspace) is skipped until it has waited longer than `affinity`.
        """
        now = utcnow()
        with self._claim_lock, self.engine.begin() as conn:
            statement = select(runs.c.id).where(and_(
                runs.c.status == "queued",
                or_(runs.c.worker_hint.is_(None), runs.c.worker_hint == worker_id, runs.c.created_at < now - affinity),
            ))
            if preset_ids:
                statement = statement.where(runs.c.preset_id.in_(preset_ids))
            statement = statement.order_by(runs.c.priority.desc(), runs.c.created_at).limit(1)
            row = conn.execute(statement).first()
            if not row:
                return None
            result = conn.execute(update(runs).where(and_(runs.c.id == row.id, runs.c.status == "queued")).values(
                status="running", worker_id=worker_id, started_at=now, heartbeat_at=now))
            if result.rowcount != 1:
                return None
        return self.get_run(row.id)

    def update_run(self, run_id: str, **values: Any) -> dict[str, Any] | None:
        row: dict[str, Any] = {}
        for key, value in values.items():
            if key in {"stages", "metrics", "checks", "labels"}:
                row[f"{key}_json"] = _dumps(value)
            elif key in {"title", "status", "summary", "error", "heartbeat_at", "finished_at", "cancel_requested",
                         "priority", "worker_id", "started_at"}:
                row[key] = value
        if row:
            with self.engine.begin() as conn:
                conn.execute(update(runs).where(runs.c.id == run_id).values(**row))
        return self.get_run(run_id)

    def request_cancel(self, run_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as conn:
            conn.execute(update(runs).where(and_(runs.c.id == run_id, runs.c.status == "queued")).values(
                status="cancelled", cancel_requested=True, finished_at=utcnow(), summary="已取消（未开始）"))
            conn.execute(update(runs).where(and_(runs.c.id == run_id, runs.c.status == "running")).values(
                cancel_requested=True))
        return self.get_run(run_id)

    def delete_run(self, run_id: str) -> list[str]:
        with self.engine.begin() as conn:
            blob_ids = [row.blob_id for row in conn.execute(select(artifacts.c.blob_id).where(artifacts.c.run_id == run_id))]
            conn.execute(delete(artifacts).where(artifacts.c.run_id == run_id))
            conn.execute(delete(runs).where(runs.c.id == run_id))
        return blob_ids

    def stale_runs(self, older_than: timedelta) -> list[dict[str, Any]]:
        cutoff = utcnow() - older_than
        with self.engine.connect() as conn:
            rows = conn.execute(select(runs).where(and_(runs.c.status == "running", runs.c.heartbeat_at < cutoff))).all()
        return [_row(row) for row in rows]

    # ---------------------------------------------------------------- artifacts
    def add_artifact(self, run_id: str, name: str, kind: str, blob_id: str, size: int) -> dict[str, Any]:
        with self.engine.begin() as conn:
            conn.execute(delete(artifacts).where(and_(artifacts.c.run_id == run_id, artifacts.c.name == name)))
            result = conn.execute(insert(artifacts).values(run_id=run_id, name=name, kind=kind, blob_id=blob_id,
                                                           size=size, created_at=utcnow()))
            artifact_id = result.inserted_primary_key[0]
        with self.engine.connect() as conn:
            return _row(conn.execute(select(artifacts).where(artifacts.c.id == artifact_id)).first())

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(artifacts).where(artifacts.c.run_id == run_id).order_by(artifacts.c.id)).all()
        return [_row(row) for row in rows]

    def get_artifact(self, run_id: str, name: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(artifacts).where(and_(artifacts.c.run_id == run_id, artifacts.c.name == name))).first()
        return _row(row) if row else None

    # ---------------------------------------------------------------- workers / audit
    def touch_worker(self, worker_id: str, name: str | None, token_id: int | None,
                     info: dict[str, Any] | None) -> None:
        """Record that a worker is alive; `name`/`info` of None keep the stored values."""
        now = utcnow()
        with self.engine.begin() as conn:
            exists = conn.execute(select(workers.c.id).where(workers.c.id == worker_id)).first()
            values: dict[str, Any] = {"token_id": token_id, "last_seen_at": now}
            if name is not None or not exists:
                values["name"] = name or worker_id
            if info is not None or not exists:
                values["info_json"] = _dumps(info or {})
            if exists:
                conn.execute(update(workers).where(workers.c.id == worker_id).values(**values))
            else:
                conn.execute(insert(workers).values(id=worker_id, **values))

    def list_workers(self) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(workers).order_by(workers.c.last_seen_at.desc())).all()
        return [_row(row) for row in rows]

    def add_audit(self, actor: str, action: str, target: str = "", detail: dict[str, Any] | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(audit).values(at=utcnow(), actor=actor, action=action, target=target,
                                              detail_json=_dumps(detail or {})))

    def list_audit(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(audit).order_by(audit.c.id.desc()).limit(limit)).all()
        return [_row(row) for row in rows]
