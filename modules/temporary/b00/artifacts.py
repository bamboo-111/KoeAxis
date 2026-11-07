"""storage.artifacts/write_fixture_json: NEW controlled operation, not legacy write_json."""
import hashlib
from host.contracts import ModuleError, effect, json_bytes


class Implementation:
    def __init__(self, settings):
        pass

    def invoke(self, operation, arguments, api):
        resource = arguments["path"]
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
                              effects=[effect(resource, "write", "unknown", "write failed; residue not rolled back")],
                              legacy_identity=type(exc).__name__) from exc
        receipt = {"path": resource["locator"], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return {"result": receipt}, [effect(resource, "write", "committed", "new fixture file closed; no crash-atomicity claim")]
