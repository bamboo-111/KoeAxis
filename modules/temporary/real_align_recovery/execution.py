"""Real filelock acquire/release for resource.execution. Not the E2E-P1 fake."""
import sys
from pathlib import Path

from host.contracts import ModuleError, effect

LEGACY = Path(__file__).resolve().parents[3] / "compat" / "legacy".resolve()
FILELOCK_SITE = LEGACY / ".venv312" / "Lib" / "site-packages"


def filelock_api():
    """Reuse the installed filelock package. Read-only import; no install."""
    try:
        import filelock
    except ImportError:
        site = str(FILELOCK_SITE)
        if site not in sys.path:
            sys.path.insert(0, site)
        try:
            import filelock
        finally:
            if site in sys.path:
                sys.path.remove(site)
    return filelock.FileLock, filelock.Timeout


def outside_legacy(path):
    resolved = Path(path).resolve()
    return resolved != LEGACY and LEGACY not in resolved.parents


def resolve_lock_path(lock_path, output_root):
    path = Path(lock_path)
    root = Path(output_root).resolve()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ModuleError("PermissionDenied", "lock path is outside the host output",
                          partial={"acquired": False})
    if not outside_legacy(path):
        raise ModuleError("PermissionDenied", "lock path is inside the legacy repository",
                          partial={"acquired": False})
    return path


def try_acquire(path, timeout_seconds):
    FileLock, Timeout = filelock_api()
    if type(timeout_seconds) not in (int, float) or isinstance(timeout_seconds, bool):
        raise ModuleError("InvalidTimeout", "timeout must be a finite number",
                          partial={"acquired": False})
    if timeout_seconds < 0 or timeout_seconds > 30:
        raise ModuleError("InvalidTimeout", "timeout must be finite and at most 30 seconds",
                          partial={"acquired": False})
    lock = FileLock(str(path), timeout=timeout_seconds)
    try:
        lock.acquire()
    except Timeout as exc:
        return {"acquired": False, "lock": None, "error": type(exc).__name__,
                "implementation": type(lock).__module__ + "." + type(lock).__name__}
    return {"acquired": True, "lock": lock, "error": None,
            "implementation": type(lock).__module__ + "." + type(lock).__name__}


class Implementation:
    def __init__(self, settings):
        self.settings = settings
        self.held = {}

    def invoke(self, operation, arguments, api):
        if operation == "acquire_gpu_lock":
            return self.acquire(arguments, api)
        if operation == "release_gpu_lock":
            return self.release(arguments, api)
        raise ModuleError("UnknownOperation", operation)

    def _holder(self, api):
        parent = api.context["parent_call_id"] or api.context["call_id"]
        if type(parent) is not str or not parent:
            raise ModuleError("PermissionDenied", "envelope has no holder call",
                              partial={"acquired": False})
        return parent

    def acquire(self, arguments, api):
        path = resolve_lock_path(arguments["lock_path"], arguments["output_root"])
        key = str(path)
        if key in self.held:
            raise ModuleError("DoubleAcquire", "this process already holds the lock",
                              partial={"acquired": True, "lock_path": key})
        path.parent.mkdir(parents=True, exist_ok=True)
        outcome = try_acquire(path, arguments["timeout_seconds"])
        if outcome["acquired"] is not True:
            raise ModuleError("LockTimeout", outcome["error"] or "lock timeout",
                              partial={"acquired": False, "timed_out": True,
                                       "implementation": outcome["implementation"],
                                       "lock_path": key})
        holder = self._holder(api)
        self.held[key] = {"lock": outcome["lock"], "holder": holder}
        lease = {
            "acquired": True,
            "lock_path": key,
            "holder": holder,
            "stage": arguments["stage"],
            "timeout_seconds": arguments["timeout_seconds"],
            "implementation": outcome["implementation"],
            "cooperative_only": True,
            "rejects_external_gpu_process": False,
        }
        return {"lease": lease}, [effect(
            {"locator": "gpu.lock", "kind": "filelock"}, "write", "committed",
            "filelock acquired")]

    def release(self, arguments, api):
        path = Path(arguments["lock_path"]).resolve()
        key = str(path)
        if not outside_legacy(path):
            raise ModuleError("PermissionDenied", "lock path is inside the legacy repository",
                              partial={"released": False})
        item = self.held.get(key)
        if item is None:
            raise ModuleError("NotHeld", "lock is not held by this process",
                              partial={"released": False, "lock_path": key})
        holder = self._holder(api)
        if item["holder"] != holder or arguments["holder"] != holder:
            raise ModuleError("WrongHolder", "release holder does not match acquire",
                              partial={"released": False, "lock_path": key})
        item["lock"].release()
        del self.held[key]
        result = {"released": True, "held_after": key in self.held, "lock_path": key,
                  "holder": holder}
        return {"result": result}, [effect(
            {"locator": "gpu.lock", "kind": "filelock"}, "write", "committed",
            "filelock released")]
