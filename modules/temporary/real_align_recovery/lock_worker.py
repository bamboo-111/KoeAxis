"""Two-process filelock probe. Prints one JSON object. Does not touch the legacy repo."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("real_align_execution", HERE / "execution.py")
execution = importlib.util.module_from_spec(spec)
sys.modules["real_align_execution"] = execution
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
spec.loader.exec_module(execution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--hold-ms", type=int, default=0)
    parser.add_argument("--ready", default="")
    parser.add_argument("--wait-ready", default="")
    parser.add_argument("--wait-release", default="")
    parser.add_argument("--exit-without-release", action="store_true")
    parser.add_argument("--touch-lock", action="store_true")
    args = parser.parse_args()
    if args.wait_ready:
        deadline = time.time() + 10
        while not Path(args.wait_ready).exists():
            if time.time() > deadline:
                print(json.dumps({"role": args.role, "pid": os.getpid(), "error": "ready-timeout"}), flush=True)
                return 2
            time.sleep(0.02)
    started = time.time()
    if not args.touch_lock:
        if args.hold_ms:
            time.sleep(args.hold_ms / 1000)
        print(json.dumps({"role": args.role, "pid": os.getpid(), "acquired": None,
                          "touched_lock": False, "elapsed_ms": int((time.time() - started) * 1000)}), flush=True)
        return 0
    path = Path(args.lock)
    path.parent.mkdir(parents=True, exist_ok=True)
    outcome = execution.try_acquire(path, args.timeout)
    event = {"role": args.role, "pid": os.getpid(), "lock_path": str(path.resolve()),
             "attempt_at": started, "acquire_end_at": time.time(), "acquired": outcome["acquired"],
             "error": outcome["error"], "implementation": outcome["implementation"],
             "touched_lock": True, "elapsed_ms": int((time.time() - started) * 1000)}
    if args.ready and outcome["acquired"]:
        Path(args.ready).write_text(str(os.getpid()), encoding="utf-8", newline="\n")
    if outcome["acquired"] and args.wait_release:
        deadline = time.monotonic() + 15
        while not Path(args.wait_release).exists():
            if time.monotonic() >= deadline:
                outcome["lock"].release()
                event.update(error="release-timeout", released=True, release_at=time.time())
                print(json.dumps(event), flush=True)
                return 2
            time.sleep(0.02)
    if outcome["acquired"] and args.hold_ms:
        time.sleep(args.hold_ms / 1000)
        event["elapsed_ms"] = int((time.time() - started) * 1000)
    if outcome["acquired"] and args.exit_without_release:
        print(json.dumps(event), flush=True)
        os._exit(9)
    if outcome["lock"] is not None:
        outcome["lock"].release()
        event["release_at"] = time.time()
        event["released"] = True
    else:
        event["released"] = False
    event["elapsed_ms"] = int((time.time() - started) * 1000)
    print(json.dumps(event), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
