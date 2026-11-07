"""Command line of the CI service.

    python -m qwen_asr.ci server  [--host 0.0.0.0 --port 8710 --data-dir ci-data]
    python -m qwen_asr.ci worker  --server URL --token kxw_... [--id gpu-01 --default-run-args "..."]
    python -m qwen_asr.ci phase   --spec spec.json --phase gpu|api [--gpu-lock PATH]   (used by workers)
    python -m qwen_asr.ci create-user  --username admin --role admin   (password: prompt or $KOEAXIS_PASSWORD)
    python -m qwen_asr.ci create-token --kind worker --name gpu-01
    python -m qwen_asr.ci openapi [--output openapi.json]

Server settings come from ``KOEAXIS_CI_*`` environment variables (see qwen_asr/ci/config.py); the flags
below only override the most common ones.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import shlex
import sys
from pathlib import Path


def _settings(args: argparse.Namespace):
    from qwen_asr.ci.config import Settings

    overrides = {"data_dir": getattr(args, "data_dir", None), "db_url": getattr(args, "db_url", None)}
    if getattr(args, "presets_dir", None):
        overrides["presets_dirs"] = [Path(item) for item in args.presets_dir]
    return Settings.from_env(**overrides)


def cmd_server(args: argparse.Namespace) -> int:
    import uvicorn

    from qwen_asr.ci.api import create_app

    settings = _settings(args)
    app = create_app(settings)
    if app.state.store.user_count() == 0:
        print("[ci] no users yet: create an admin with `python -m qwen_asr.ci create-user --role admin ...`",
              file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=True,
                forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"), log_level=args.log_level)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    from qwen_asr.ci.worker import Worker, WorkerConfig

    token = args.token or os.environ.get("KOEAXIS_WORKER_TOKEN", "")
    if not args.server or not token:
        print("--server and --token (or KOEAXIS_WORKER_TOKEN) are required", file=sys.stderr)
        return 2
    env = {}
    for item in args.env or []:
        key, _, value = item.partition("=")
        env[key] = value
    config = WorkerConfig(
        server=args.server, token=token, data_dir=Path(args.data_dir), max_runs=args.max_runs,
        poll_interval_s=args.poll_interval, default_run_args=shlex.split(args.default_run_args or "", posix=True),
        env=env, preset_ids=args.preset or None,
        model_cache_dir=Path(args.model_cache_dir) if args.model_cache_dir else None,
        upload_manifests=not args.no_manifests,
        **({"worker_id": args.id} if args.id else {}), **({"name": args.name} if args.name else {}),
        **({"phase_command": shlex.split(args.phase_command)} if args.phase_command else {}),
    )
    Worker(config).serve(once=args.once)
    return 0


def cmd_phase(args: argparse.Namespace, rest: list[str]) -> int:
    from qwen_asr.ci.phase import main

    return main(rest)


def cmd_create_user(args: argparse.Namespace) -> int:
    from qwen_asr.ci.auth import hash_password
    from qwen_asr.ci.store import Store

    settings = _settings(args)
    settings.ensure_dirs()
    store = Store(settings.db_url)
    password = os.environ.get("KOEAXIS_PASSWORD") or getpass.getpass(f"password for {args.username}: ")
    existing = store.get_user_by_name(args.username)
    if existing:
        store.update_user(existing["id"], password_hash=hash_password(password), role=args.role, disabled=False)
        print(f"updated user {args.username} ({args.role})")
    else:
        store.create_user(args.username, hash_password(password), args.role, args.display_name or "")
        print(f"created user {args.username} ({args.role})")
    store.add_audit("cli", "user.create", args.username, {"role": args.role})
    return 0


def cmd_create_token(args: argparse.Namespace) -> int:
    from qwen_asr.ci.auth import hash_token, new_token
    from qwen_asr.ci.store import Store

    settings = _settings(args)
    settings.ensure_dirs()
    store = Store(settings.db_url)
    user_id = None
    if args.kind == "user":
        user = store.get_user_by_name(args.username or "")
        if not user:
            print("--username of an existing user is required for user tokens", file=sys.stderr)
            return 2
        user_id = user["id"]
    token = new_token(args.kind)
    record = store.create_token(args.name, hash_token(token), args.kind, user_id)
    store.add_audit("cli", "token.create", str(record["id"]), {"kind": args.kind})
    print(token)
    print(f"(token id {record['id']}; shown once — store it in the worker/client environment)", file=sys.stderr)
    return 0


def cmd_openapi(args: argparse.Namespace) -> int:
    import tempfile

    from qwen_asr.ci.api import create_app
    from qwen_asr.ci.config import Settings

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Settings(data_dir=Path(tmp), presets_dirs=[]), start_background=False)
        text = json.dumps(app.openapi(), ensure_ascii=False, indent=2)
        app.state.store.engine.dispose()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qwen_asr.ci", description="KoeAxis subtitle CI service")
    sub = parser.add_subparsers(dest="command", required=True)

    def storage_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", default=None, help="server data directory (env KOEAXIS_CI_DATA_DIR)")
        p.add_argument("--db-url", default=None, help="SQLAlchemy URL (env KOEAXIS_CI_DB_URL)")
        p.add_argument("--presets-dir", action="append", help="preset directory (repeatable)")

    server = sub.add_parser("server", help="run the control server (API + web UI)")
    storage_flags(server)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8710)
    server.add_argument("--log-level", default="info")

    worker = sub.add_parser("worker", help="run a worker that pulls runs from the server")
    worker.add_argument("--server", default=os.environ.get("KOEAXIS_SERVER", ""))
    worker.add_argument("--token", default="", help="worker token (or env KOEAXIS_WORKER_TOKEN)")
    worker.add_argument("--id", default="", help="stable worker id (default: host name)")
    worker.add_argument("--name", default="")
    worker.add_argument("--data-dir", default=os.environ.get("KOEAXIS_WORKER_DATA_DIR", "ci-worker"))
    worker.add_argument("--max-runs", type=int, default=2)
    worker.add_argument("--poll-interval", type=float, default=10.0)
    worker.add_argument("--default-run-args", default=os.environ.get("KOEAXIS_WORKER_RUN_ARGS", ""),
                        help='CLI defaults for `main.py run` on this machine, e.g. "--device cuda:0 --dtype fp16"')
    worker.add_argument("--env", action="append", help="KEY=VALUE for phase processes (repeatable)")
    worker.add_argument("--preset", action="append", help="only claim runs of these presets (repeatable)")
    worker.add_argument("--model-cache-dir", default=os.environ.get("KOEAXIS_MODEL_CACHE_DIR", ""))
    worker.add_argument("--phase-command", default="", help="override the phase command prefix")
    worker.add_argument("--no-manifests", action="store_true", help="do not upload manifests.zip")
    worker.add_argument("--once", action="store_true", help="exit after one run")
    worker.add_argument("--log-level", default="INFO")

    sub.add_parser("phase", help="(internal) run one phase of a run", add_help=False)

    user = sub.add_parser("create-user", help="create or reset a user")
    storage_flags(user)
    user.add_argument("--username", required=True)
    user.add_argument("--role", choices=["admin", "member"], default="member")
    user.add_argument("--display-name", default="")

    token = sub.add_parser("create-token", help="create an API token (printed once)")
    storage_flags(token)
    token.add_argument("--kind", choices=["worker", "user"], default="worker")
    token.add_argument("--name", required=True)
    token.add_argument("--username", default="")

    openapi = sub.add_parser("openapi", help="print the OpenAPI schema")
    openapi.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "phase":
        return cmd_phase(argparse.Namespace(), argv[1:])
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(args, "log_level", "INFO").upper() if args.command == "worker" else "INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = {"server": cmd_server, "worker": cmd_worker, "create-user": cmd_create_user,
                "create-token": cmd_create_token, "openapi": cmd_openapi}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
