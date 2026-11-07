"""Fixture JSON read/write. Same host file rule as the other temporary packages."""
import hashlib
import json

from host.contracts import ModuleError, effect, json_bytes


class Implementation:
    def __init__(self, settings):
        self.settings = settings

    def invoke(self, operation, arguments, api):
        resource = arguments["path"]
        if operation == "write_fixture_json":
            path = api.path(resource, "write")
            data = json_bytes(arguments["payload"]) + b"\n"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    written = stream.write(data)
                    if written != len(data):
                        raise OSError("short fixture write")
            except OSError as exc:
                raise ModuleError("FixtureWriteError", str(exc),
                                  effects=[effect(resource, "write", "unknown",
                                                   "write failed; residue not rolled back")],
                                  legacy_identity=type(exc).__name__) from exc
            receipt = {"path": resource["locator"], "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}
            return {"result": receipt}, [effect(resource, "write", "committed",
                                                 "new fixture file closed")]
        if operation == "read_fixture_json":
            path = api.path(resource, "read")
            data = path.read_bytes()
            receipt = {"path": resource["locator"], "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "payload": json.loads(data)}
            return {"result": receipt}, [effect(resource, "read", "committed",
                                                 "read exact fixture bytes")]
        raise ModuleError("UnknownOperation", operation)
