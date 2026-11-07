"""B00 public data contracts only; no module business logic or dispatch."""
import hashlib
import json
import math
import re

VERSION = "0.1.0-b00"
EMPTY_CONFIG = {"entries": {}, "provenance": [], "evaluation_records": []}


class ContractError(ValueError):
    pass


def json_bytes(value):
    def check(v, parents):
        if v is None or type(v) in (str, bool, int):
            return
        if type(v) is float and math.isfinite(v):
            return
        if type(v) not in (dict, list) or id(v) in parents or len(parents) > 64:
            raise ContractError("finite acyclic JSON required")
        parents = parents | {id(v)}
        if type(v) is dict:
            if any(type(k) is not str for k in v):
                raise ContractError("JSON object keys must be strings")
            for child in v.values():
                check(child, parents)
        else:
            for child in v:
                check(child, parents)
    check(value, set())
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def clone(value):
    return json.loads(json_bytes(value))


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def identity(record):
    return {k: record[k] for k in ("module_id", "implementation_id", "version", "content_digest")}


def ref(module, operation, version=VERSION):
    return {"module_id": module, "operation_id": operation, "interface_version": version}


def artifact(path, owner="test.b00", content_identity=None):
    return {"artifact_id": path.replace("/", "."), "locator": path,
            "representation": "application/json; charset=utf-8",
            "permissions": ["read", "write"], "content_identity": content_identity,
            "state_owner": owner, "writer_operation": ref("storage.artifacts", "write_fixture_json"),
            "lifetime": "run"}


def effect(resource, operation, state, observation):
    return {"resource_ref": clone(resource), "operation": operation,
            "commit_state": state, "observation": observation, "evidence_ref": None}


class ModuleError(Exception):
    def __init__(self, category, message, *, partial=None, effects=None, causes=None,
                 legacy_identity="b00.controlled"):
        super().__init__(message)
        self.error = {"category": category, "legacy_identity": legacy_identity,
                      "message": message, "cause_chain": causes or [],
                      "partial_result": partial, "effects": effects or []}


def returned(result):
    """Explicit caller choice to propagate a child's failure; never a host retry."""
    if result["outcome"] == "returned":
        return result["returned"]
    if result["outcome"] == "raised":
        e = result["raised"]
        raise ModuleError(e["category"], e["message"], partial=e["partial_result"],
                          effects=e["effects"], causes=e["cause_chain"],
                          legacy_identity=e["legacy_identity"])
    raise ModuleError("ChildRejected", result["host_rejected"]["code"], partial=result)


class Validator:
    def __init__(self, profile):
        self.types = profile["types"]

    def check(self, value, spec):
        if type(spec) is str:
            if spec in self.types:
                return self.check(value, self.types[spec])
            if spec == "json":
                json_bytes(value)
                return
            valid = ((spec == "null" and value is None) or
                     (spec == "text" and type(value) is str and bool(value)) or
                     (spec == "id" and type(value) is str and value != "latest" and
                      re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)) or
                     (spec == "sha256" and type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)))
            if not valid:
                raise ContractError("invalid " + spec)
            return
        if "object" in spec:
            props = spec["object"]
            if type(value) is not dict or set(value) != set(props):
                raise ContractError("object fields must exactly match declaration")
            for key, child in props.items():
                self.check(value[key], child)
        elif "array" in spec:
            if type(value) is not list:
                raise ContractError("array required")
            for child in value:
                self.check(child, spec["array"])
        elif "const" in spec:
            if json_bytes(value) != json_bytes(spec["const"]):
                raise ContractError("constant mismatch")
        elif "enum" in spec:
            if json_bytes(value) not in [json_bytes(x) for x in spec["enum"]]:
                raise ContractError("enum mismatch")
        elif "union" in spec:
            for child in spec["union"]:
                try:
                    self.check(value, child)
                    return
                except ContractError:
                    pass
            raise ContractError("no allowed alternative")
        else:
            raise ContractError("unsupported contract schema")
