"""Trusted local B00 host: explicit loading/binding/dispatch, never workflow scheduling."""
import builtins
import contextlib
import importlib.metadata
import importlib.util
import stat
import threading

HOST_VERSION = "0.1.7"
import hashlib
import re
import types
import json
import pathlib
import platform
import sys
import os
from collections.abc import Mapping

from .contracts import (ContractError, EMPTY_CONFIG, ModuleError, Validator,
                        clone, digest, effect, identity, ref)


def validate_access_declaration(declaration):
    """0.1.7 additions are optional for frozen pre-0.1.7 trusted modules."""
    names = declaration.get("environment", [])
    secrets = declaration.get("secret_environment", [])
    for values in (names, secrets):
        if (type(values) is not list or any(type(n) is not str or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in values)
                or len(values) != len(set(values))):
            raise ContractError("invalid environment access declaration")
    if not set(secrets) <= set(names):
        raise ContractError("secret environment must be declared readable")
    for key in ("secret_inputs", "secret_outputs"):
        paths = declaration.get(key, [])
        if (type(paths) is not list or any(type(p) is not list or not p or
                any(type(s) is not str or not s for s in p) for p in paths)):
            raise ContractError("invalid secret field declaration")


def secret_fields(value, paths):
    def visit(node, path):
        if not path:
            yield node
        elif path[0] == "*" and type(node) in (list, dict):
            for child in (node.values() if type(node) is dict else node):
                yield from visit(child, path[1:])
        elif type(node) is dict and path[0] in node:
            yield from visit(node[path[0]], path[1:])
    for path in paths:
        yield from visit(value, path)


class SecretObservations:
    """In-memory observation projection for trusted callers, not a sandbox.

    Track common case/strip and repr/ascii/JSON forms. With known secrets, opaque
    *_repr diagnostics are digested structurally, not interpreted as Python.
    Other arbitrary encodings, native I/O and pre-declaration emissions remain
    outside this facade. Declare inputs before use, not only returned fields.
    """
    def __init__(self):
        self.values = {}
        self.representations = set()
        self.markers = set()

    def _marker(self, text):
        marker = {"secret": True, "length": len(text),
                  "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        self.markers.add((marker["length"], marker["sha256"]))
        return marker

    def _is_marker(self, value):
        return (type(value) is dict and set(value) == {"secret", "length", "sha256"}
                and value["secret"] is True and type(value["length"]) is int
                and type(value["sha256"]) is str
                and (value["length"], value["sha256"]) in self.markers)

    def remember(self, value):
        if type(value) is str and value.strip():
            for base in (value, value.strip()):
                for text in {base, base.lower(), base.upper(), base.casefold(),
                             base.swapcase(), base.capitalize(), base.title()}:
                    if text in self.values:
                        continue
                    self.values[text] = self._marker(text)
                    self.representations.add(text)
                    # Include unquoted payloads for strings embedded in messages,
                    # plus quoted forms. No eval/exec or escape decoding is used.
                    for encoded in (repr(text), ascii(text), json.dumps(text),
                                    json.dumps(text, ensure_ascii=False)):
                        self.representations.update((encoded, encoded[1:-1]))
        elif type(value) is dict:
            for child in value.values():
                self.remember(child)
        elif type(value) is list:
            for child in value:
                self.remember(child)

    def _contains(self, text):
        return any(secret in text for secret in self.representations)

    def redact(self, value):
        if type(value) is str:
            if value in self.values:
                return clone(self.values[value])
            # Digest the whole observation, never substitute inside its fragments.
            return self._marker(value) if self._contains(value) else value
        if type(value) is list:
            return [self.redact(x) for x in value]
        if type(value) is dict:
            if self._is_marker(value):
                return clone(value)
            result = {}
            for key, child in value.items():
                projected_key = ("secret-key-" + hashlib.sha256(key.encode("utf-8")).hexdigest()
                                 if self._contains(key) else key)
                if self.values and (key == "repr" or key.endswith("_repr")) and not self._is_marker(child):
                    # Legacy exception projections may nest/re-encode arbitrary
                    # objects. They are diagnostics, never a secret transport.
                    text = child if type(child) is str else json.dumps(child, ensure_ascii=True, sort_keys=True)
                    result[projected_key] = self._marker(text)
                else:
                    result[projected_key] = self.redact(child)
            return result
        return value


class DeclaredEnvironment(Mapping):
    """Read-only view: deny before touching the backing mapping, including misses."""
    def __init__(self, api, source):
        self.api, self.source = api, source

    def __getitem__(self, name):
        self.api._check_environment(name)
        value = self.source[name]
        if name in self.api.declaration.get("secret_environment", []):
            self.api._host._secrets.remember(value)
        return value

    def __iter__(self):
        raise ModuleError("PermissionDenied", "environment enumeration is not declared")

    def __len__(self):
        raise ModuleError("PermissionDenied", "environment enumeration is not declared")


_ENVIRONMENT_LOCK = threading.RLock()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_source(path, expected_sha256):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ModuleError("BindingMismatch", "content file mismatch: " + str(path))
    return source


def execute_source(name, path, verified_bytes):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(verified_bytes, str(path), "exec"), module.__dict__)
    return module


def safe_path(root, relative):
    return sys.modules["host"].guarded_path(root, relative, ContractError)


def capture_legacy_sources(spec):
    """0.1.5: read declared read-only legacy files once; exact SHA256 or BindingMismatch."""
    root = pathlib.Path(spec["root"])
    captured = {}
    for item in spec["files"]:
        path = safe_path(root, item["path"])
        if item["module"] in captured:
            raise ModuleError("BindingMismatch", "duplicate legacy module")
        captured[item["module"]] = (path, capture_source(path, item["sha256"]))
    return captured


# All legacy registrations are serialized; unrelated Python imports are not sandboxed.
_REGISTRY_LOCK = threading.RLock()
_ACTIVE_CLOSURE = None


class LegacyClosure(dict):
    """Captured modules; use activate() for runtime dataclasses/typing introspection.

    Construction is automatically activated. Between activations sys.modules has no
    residue. Distinct overlapping activations in one thread fail closed; other threads
    serialize. Escaped module references are trusted caller responsibility.
    """
    def __init__(self):
        super().__init__()
        self.closed = False
        self._depth = 0
        self._registered = {}

    def _register(self, name, module):
        if name in sys.modules and sys.modules[name] is not module:
            raise ImportError("legacy module name already occupied: " + name)
        if name not in self._registered:
            self._registered[name] = (name in sys.modules, sys.modules.get(name))
        sys.modules[name] = module

    @contextlib.contextmanager
    def activate(self):
        global _ACTIVE_CLOSURE
        with _REGISTRY_LOCK:
            if self.closed:
                raise ModuleError("PermissionDenied", "legacy closure is released")
            if _ACTIVE_CLOSURE is not None and _ACTIVE_CLOSURE is not self:
                raise ModuleError("PermissionDenied", "distinct legacy closures cannot overlap")
            outer = self._depth == 0
            self._depth += 1
            _ACTIVE_CLOSURE = self
            try:
                if outer:
                    for name, module in self.items():
                        self._register(name, module)
                yield self
            finally:
                self._depth -= 1
                if outer:
                    for name, (existed, original) in self._registered.items():
                        sys.modules.pop(name, None)
                        if existed:
                            sys.modules[name] = original
                    self._registered.clear()
                    _ACTIVE_CLOSURE = None

    def close(self):
        with _REGISTRY_LOCK:
            if self._depth:
                raise ModuleError("PermissionDenied", "cannot release an active legacy closure")
            self.clear()
            self.closed = True


def legacy_closure(captured, package, resource_modules=()):
    """Fresh captured-byte closure. `package` accepts one top-level name or a list.

    Captured legacy code never uses path loaders or pyc. A resource module is an
    explicitly verified trusted root loaded normally (not a Python security sandbox).
    """
    packages = [package] if type(package) is str else package
    if (type(packages) is not list or not packages or len(set(packages)) != len(packages)
            or any(type(p) is not str or not p.isidentifier() or p in sys.stdlib_module_names
                   or p == "host" for p in packages)):
        raise ModuleError("BindingMismatch", "invalid legacy package list")
    if set(packages) & set(resource_modules):
        raise ModuleError("BindingMismatch", "legacy/resource package overlap")
    if any(name.split(".")[0] not in packages for name in captured):
        raise ModuleError("BindingMismatch", "captured module outside declared packages")
    modules = LegacyClosure()
    namespace_builtins = dict(builtins.__dict__)

    def load(name):
        if name in modules:
            return modules[name]
        if name not in captured:
            raise ImportError("legacy module outside declared closure: " + name)
        parent = load(name.rsplit(".", 1)[0]) if "." in name else None
        # A package initializer may itself have loaded this child.
        if name in modules:
            return modules[name]
        path, data = captured[name]
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = name if path.name == "__init__.py" else name.rpartition(".")[0]
        if path.name == "__init__.py":
            module.__path__ = []  # normal importers must not discover legacy files
        module.__builtins__ = namespace_builtins
        modules[name] = module
        modules._register(name, module)
        exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[1], module)
        return module

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            raise ImportError("relative legacy import is not declared")
        top = name.split(".")[0]
        if top in packages:
            if modules.closed:
                raise ImportError("legacy closure is released")
            module = load(name)
            for item in fromlist or ():
                if not hasattr(module, item) and name + "." + item in captured:
                    load(name + "." + item)
            return module if fromlist else load(top)
        if top not in sys.stdlib_module_names and top not in resource_modules:
            raise ImportError("undeclared non-stdlib import from legacy closure: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace_builtins["__import__"] = guarded_import
    try:
        with modules.activate():
            for name in captured:
                load(name)
    except BaseException:
        modules.close()
        raise
    return modules


def marker_environment():
    """Finite, local marker vocabulary; no implicit packaging environment."""
    return {"python_version": ".".join(map(str, sys.version_info[:2])),
            "python_full_version": platform.python_version(),
            "sys_platform": sys.platform, "platform_system": platform.system(),
            "platform_python_implementation": platform.python_implementation(),
            "implementation_name": sys.implementation.name, "extra": ""}


def validate_marker_profile(profile):
    environment = marker_environment()
    if type(profile) is not dict or not profile:
        raise ValueError("marker profile must be a nonempty explicit mapping")
    for name, value in profile.items():
        if name not in environment or type(value) is not str or value != environment[name]:
            raise ValueError("unsupported or mismatched marker profile: " + str(name))
    return profile


def evaluate_marker(marker, profile):
    """Restricted grammar: variable comparison quoted literal; and/or/parentheses.

    Parse/evaluate every operand (no short-circuit omission of undeclared names).
    Only numeric release versions are ordered; other fields support ==/!= only.
    Extras are disabled by the explicit empty value. No eval, exec or fallback.
    """
    validate_marker_profile(profile)
    if type(marker) is not str or len(marker) > 2048:
        raise ValueError("invalid marker size")
    token_re = re.compile(r'''\s*(?:(?P<word>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>==|!=|<=|>=|<|>)|(?P<literal>'[^'\\]*'|"[^"\\]*")|(?P<paren>[()]))''')
    text, tokens, offset = marker.strip(), [], 0
    while offset < len(text):
        match = token_re.match(text, offset)
        if not match:
            raise ValueError("unsupported marker syntax")
        tokens.append((match.lastgroup, match.group(match.lastgroup)))
        offset = match.end()
    if not tokens or len(tokens) > 256:
        raise ValueError("invalid marker token count")
    index = 0

    def take(kind):
        nonlocal index
        if index == len(tokens) or tokens[index][0] != kind:
            raise ValueError("unsupported marker grammar")
        value = tokens[index][1]
        index += 1
        return value

    def atom():
        nonlocal index
        if index < len(tokens) and tokens[index] == ("paren", "("):
            index += 1
            value = expression()
            if take("paren") != ")":
                raise ValueError("unbalanced marker parentheses")
            return value
        name, operator, quoted = take("word"), take("op"), take("literal")
        if name not in profile:
            raise ValueError("undeclared marker variable: " + name)
        left, right = profile[name], quoted[1:-1]
        if name in ("python_version", "python_full_version"):
            if not all(re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", x) for x in (left, right)):
                raise ValueError("unsupported marker version literal")
            def release(value):
                parts = tuple(map(int, value.split(".")))
                return parts + (0,) * (3 - len(parts))
            left, right = release(left), release(right)
        elif operator not in ("==", "!="):
            raise ValueError("only equality supported for non-version marker")
        return {"==": lambda: left == right, "!=": lambda: left != right,
                "<": lambda: left < right, "<=": lambda: left <= right,
                ">": lambda: left > right, ">=": lambda: left >= right}[operator]()

    def conjunction():
        nonlocal index
        value = atom()
        while index < len(tokens) and tokens[index] == ("word", "and"):
            index += 1
            other = atom()
            value = value and other
        return value

    def expression():
        nonlocal index
        value = conjunction()
        while index < len(tokens) and tokens[index] == ("word", "or"):
            index += 1
            other = conjunction()
            value = value or other
        return value

    result = expression()
    if index != len(tokens):
        raise ValueError("trailing marker tokens")
    return result


def verify_resource_pack(pack):
    """Verify complete RECORD-style file lists before allowing normal imports.

    Includes metadata, data and existing pyc files. No installation, rewriting or
    path execution. Normal importer/cached code is a verified-then-loaded trust root;
    transitive third-party imports are not sandboxed. Required distributions must
    also be declared. Optional extras are not enabled by this profile.
    """
    if type(pack) is not list:
        raise ModuleError("BindingMismatch", "resource_pack must be a list")
    allowed, evidence, distributions, profiles = set(), [], {}, {}
    canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    try:
        for entry in pack:
            if set(entry) not in ({"name", "version", "modules", "files"},
                                  {"name", "version", "modules", "files", "marker_profile"}):
                raise ValueError("resource declaration fields")
            profile = validate_marker_profile(entry["marker_profile"]) if "marker_profile" in entry else None
            name = canonical(entry["name"])
            if name in distributions:
                raise ValueError("duplicate resource distribution")
            dist = importlib.metadata.distribution(entry["name"])
            if dist.version != entry["version"]:
                raise ValueError("resource version mismatch: " + entry["name"])
            declared = entry["files"]
            paths = [f["path"] for f in declared]
            actual = {str(f).replace("\\", "/") for f in dist.files or []}
            if not actual or len(paths) != len(set(paths)) or set(paths) != actual:
                raise ValueError("resource manifest must cover the entire distribution")
            verified = set()
            for item in declared:
                if set(item) != {"path", "sha256"} or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                    raise ValueError("invalid resource file record")
                path = pathlib.Path(dist.locate_file(item["path"]))
                # RECORD can legitimately contain ../Scripts/*.exe. The path must
                # be exactly listed by the installed distribution; no guessed files.
                for part in (path, *path.parents):
                    info = part.lstat()
                    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                        raise ValueError("resource manifest links are not supported")
                if file_hash(path) != item["sha256"]:
                    raise ValueError("resource hash mismatch: " + item["path"])
                verified.add(path.resolve())
            tops = entry["modules"]
            if type(tops) is not list or not tops or len(tops) != len(set(tops)):
                raise ValueError("resource top-level modules required")
            for top in tops:
                if (type(top) is not str or not top.isidentifier() or top in allowed
                        or top in sys.stdlib_module_names or top == "host"):
                    raise ValueError("invalid resource module grant")
                spec = importlib.util.find_spec(top)
                if spec is None or spec.origin is None or pathlib.Path(spec.origin).resolve() not in verified:
                    raise ValueError("resource module origin is not in verified manifest: " + top)
                allowed.add(top)
            distributions[name] = dist
            profiles[name] = profile
            evidence.append({"name": entry["name"], "version": dist.version,
                             "modules": list(tops), "manifest_digest": digest(entry),
                             "trust": "verified-then-normal-import; no TOCTOU or cached-code protection"})
        for name, dist in distributions.items():
            profile = profiles[name]
            audit = next(item for item in evidence if canonical(item["name"]) == name)
            if profile is not None:
                audit["marker_evaluations"] = []
            for requirement in dist.requires or []:
                if ";" in requirement:
                    requirement, marker = requirement.split(";", 1)
                    if profile is None:
                        # Preserve 0.1.5's disabled simple extra exception only.
                        if re.fullmatch(r"\s*extra\s*==\s*['\"][A-Za-z0-9_.-]+['\"]\s*", marker):
                            continue
                        raise ValueError("conditional dependency needs an explicit supported profile")
                    evaluation = {"requirement": requirement.strip(), "marker": marker.strip(),
                                  "profile": dict(profile), "result": None}
                    audit["marker_evaluations"].append(evaluation)
                    try:
                        evaluation["result"] = evaluate_marker(marker, profile)
                    except ValueError as exc:
                        evaluation["error"] = str(exc)
                        raise
                    if not evaluation["result"]:
                        continue
                match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
                if not match or canonical(match[1]) not in distributions:
                    raise ValueError("undeclared resource dependency: " + requirement)
    except (ValueError, TypeError, KeyError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        partial = {"resource_pack": evidence} if any(e.get("marker_evaluations") for e in evidence) else None
        raise ModuleError("BindingMismatch", str(exc), partial=partial) from exc
    return frozenset(allowed), evidence


class ResourceHandles:
    """Host-only objects; operations see opaque IDs and detached JSON records only."""
    def __init__(self):
        self.records = {}
        self._objects = {}
        self._kinds = {}
        self._sequence = 0
        self._busy = set()

    def register_kind(self, name, factory, actions, release, cancel=None):
        if name in self._kinds:
            raise ModuleError("BindingMismatch", "resource kind is immutable")
        self._kinds[name] = (factory, dict(actions), release, cancel)

    def _state(self, record, state, reason):
        self._sequence += 1
        record["state"] = state
        record["states"].append({"sequence": self._sequence, "state": state, "reason": reason})

    def _grant(self, api, kind, owner, scope, action):
        if not api.active:
            raise ModuleError("PermissionDenied", "resource call frame has ended")
        grants = api.declaration.get("resource_handles", [])
        if not any(g.get("access") == "resource_handle" and g.get("kind") == kind
                   and g.get("owner") == owner and scope in g.get("scopes", [])
                   and action in g.get("actions", []) for g in grants):
            raise ModuleError("PermissionDenied", "undeclared resource_handle access: " + action)

    def create(self, api, kind, scope, options):
        owner = api._request["target"]["module_id"]
        self._grant(api, kind, owner, scope, "create")
        if kind == "legacy_closure":
            api._require("legacy_source")
        if kind not in self._kinds or scope not in ("call", "stage", "run"):
            raise ModuleError("PermissionDenied", "unsupported resource kind or scope")
        scope_id = {"call": api.context["call_id"], "stage": api._stage_id,
                    "run": api.context["run_id"]}[scope]
        if scope_id is None:
            raise ModuleError("PermissionDenied", "no active resource stage")
        obj = self._kinds[kind][0](clone(options))
        token = "resource-" + str(len(self.records) + 1)
        record = {"id": token, "owner": owner, "run_id": api.context["run_id"],
                  "kind": kind, "scope": scope, "scope_id": scope_id, "state": "created", "states": []}
        self.records[token] = record
        self._objects[token] = obj
        self._state(record, "created", "explicit-create")
        return token

    def _authorize(self, api, token, action):
        record = self.records.get(token)
        if record is None or record["run_id"] != api.context["run_id"]:
            raise ModuleError("PermissionDenied", "resource is outside this run")
        self._grant(api, record["kind"], record["owner"], record["scope"], action)
        if record["scope"] == "call" and record["scope_id"] != api.context["call_id"]:
            raise ModuleError("PermissionDenied", "resource is outside this call")
        if record["scope"] == "stage" and record["scope_id"] != api._stage_id:
            raise ModuleError("PermissionDenied", "resource is outside this stage")
        # Cross-module use requires both the operation grant and explicit envelope grant.
        if record["owner"] != api._request["target"]["module_id"]:
            if token not in [r["id"] for r in api._request["resource_handles"]]:
                raise ModuleError("PermissionDenied", "resource not granted in call envelope")
        return record

    def borrow(self, api, token, action, arguments):
        record = self._authorize(api, token, "borrow:" + action)
        if record["state"] in ("cancelled", "released") or token in self._busy:
            raise ModuleError("PermissionDenied", "resource cannot be borrowed")
        handler = self._kinds[record["kind"]][1].get(action)
        if handler is None:
            raise ModuleError("PermissionDenied", "unknown resource action")
        self._state(record, "in_use", "borrow:" + action)
        self._busy.add(token)
        try:
            return clone(handler(self._objects[token], clone(arguments)))
        finally:
            self._busy.remove(token)

    def cancel(self, api, token):
        record = self._authorize(api, token, "cancel")
        if record["state"] not in ("cancelled", "released"):
            self._state(record, "cancelled", "cancel-request; not proof of termination")
            callback = self._kinds[record["kind"]][3]
            if callback is not None:
                callback(self._objects[token])
        return clone(record)

    def _release(self, token, reason):
        record = self.records[token]
        if record["state"] == "released":
            return clone(record)
        if token in self._busy:
            raise ModuleError("PermissionDenied", "resource borrow is still active")
        self._kinds[record["kind"]][2](self._objects[token])
        del self._objects[token]
        self._state(record, "released", reason)
        return clone(record)

    def release(self, api, token):
        self._authorize(api, token, "release")
        return self._release(token, "explicit-release")

    def end_scope(self, run_id, scope, scope_id):
        failures = []
        for token, record in self.records.items():
            if (record["run_id"], record["scope"], record["scope_id"]) == (run_id, scope, scope_id):
                try:
                    self._release(token, "scope-end:" + scope)
                except Exception as exc:
                    failures.append({"id": token, "error": type(exc).__name__, "message": str(exc)})
        if failures:
            raise ModuleError("ResourceReleaseFailed", "scope cleanup failed", partial=failures)

    def snapshot(self):
        return clone(list(self.records.values()))


class Host:
    def __init__(self, project, output, registration):
        sys.dont_write_bytecode = True
        self.project = pathlib.Path(project).resolve()
        # 0.1.5: plan/profile/catalog come from the registered plan entry.
        self.registration = clone(registration)
        self.profile = json.loads(safe_path(self.project, registration["profile"]).read_text("utf-8"))
        self.interface_version = self.profile["interface_version"]
        for module in self.profile["modules"].values():
            for declaration in module["operations"].values():
                validate_access_declaration(declaration)
        self.validator = Validator(self.profile)
        self.catalog = json.loads(safe_path(self.project, registration["catalog"]).read_text("utf-8"))
        self.records = {}
        self.instances = {}
        self.bindings = {}
        self.tasks = {}
        self.trace = []
        self._secrets = SecretObservations()
        self.used_calls = set()
        self.sequence = 0
        self._legacy_sources = None
        self.resources = ResourceHandles()
        self._active_calls = 0
        self._resource_stage_sequence = 0
        self._closed_runs = set()
        self._resource_modules, resource_evidence = verify_resource_pack(self.profile.get("resource_pack", []))
        self.resources.register_kind("legacy_closure", lambda options: self.legacy_modules(),
                                     {"call": self._legacy_resource_call}, lambda closure: closure.close())
        self.environment = {"python": sys.version, "platform": platform.platform(),
                            "plan": registration["plan"], "profile": registration["profile"],
                            "profile_sha256": file_hash(self.project / registration["profile"]),
                            "catalog": registration["catalog"],
                            "catalog_sha256": file_hash(self.project / registration["catalog"]),
                            "admission": "temporary", "network": False, "models": False}
        if resource_evidence:
            self.environment["resource_pack"] = resource_evidence
        for record in self.catalog["implementations"]:
            self._verify_record(record, allow_new=True)
            key = (record["implementation_id"], record["version"])
            if key in self.records:
                raise ContractError("duplicate implementation identity")
            self.records[key] = clone(record)
        boot_records = [self.records[tuple(key)] for key in self.catalog["bootstrap"]]
        manifest = {"binding_id": "bootstrap", "implementations": [identity(r) for r in boot_records],
                    "root_operations": [clone(registration["entry"])]}
        self.install_binding(manifest, boot_records)
        # Only a new directory under runs/experiments; never read historical workspaces.
        if not output.startswith("runs/experiments/") or len(output.split("/")) != 3:
            raise ContractError("output must be one new runs/experiments/<id> directory")
        self.output = safe_path(self.project, output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.environment["output"] = output
        self.new_task("bootstrap-run", "bootstrap")

    def redact(self, value):
        return self._secrets.redact(clone(value))

    def _remember_fields(self, value, paths):
        before = len(self._secrets.values)
        for secret in secret_fields(value, paths):
            self._secrets.remember(secret)
        # A returned secret may previously have appeared in a parent input.
        if len(self._secrets.values) != before:
            self.trace[:] = self.redact(self.trace)

    def legacy_modules(self):
        spec = self.profile.get("legacy_sources")
        if spec is None:
            raise ModuleError("PermissionDenied", "profile declares no legacy sources")
        if self._legacy_sources is None:
            self._legacy_sources = capture_legacy_sources({**spec, "root": str(self.project / spec["root"])})
        return legacy_closure(self._legacy_sources, spec.get("packages", spec.get("package")),
                              self._resource_modules)

    @staticmethod
    def _legacy_resource_call(closure, arguments):
        with closure.activate():
            function = closure[arguments["module"]]
            for part in arguments["function"].split("."):
                if not part.isidentifier():
                    raise ModuleError("PermissionDenied", "invalid legacy function name")
                function = getattr(function, part)
            return clone(function(*arguments.get("args", []), **arguments.get("kwargs", {})))

    def end_run(self, run_id):
        if self._active_calls:
            raise ModuleError("PermissionDenied", "cannot end run during active calls")
        if run_id not in self.tasks:
            raise ModuleError("BindingMismatch", "unknown run")
        self.resources.end_scope(run_id, "run", run_id)
        self._closed_runs.add(run_id)

    def end_runs(self):
        """Finish every live run in creation order; one failure cannot skip others."""
        if self._active_calls:
            raise ModuleError("PermissionDenied", "cannot end runs during active calls")
        failures = []
        for run_id in tuple(self.tasks):
            if run_id not in self._closed_runs:
                try:
                    self.end_run(run_id)
                except ModuleError as exc:
                    failures.append({"run_id": run_id, "error": clone(exc.error)})
        if failures:
            raise ModuleError("ResourceReleaseFailed", "run cleanup failed", partial=failures)

    def _verify_record(self, record, allow_new=False):
        self.validator.check(record, "ImplementationRecord")
        body = {k: v for k, v in record.items() if k != "content_digest"}
        if digest(body) != record["content_digest"]:
            raise ModuleError("BindingMismatch", "implementation content digest mismatch")
        if record["module_id"] not in self.profile["modules"]:
            raise ModuleError("BindingMismatch", "module not in frozen profile")
        paths = [x["path"] for x in record["content_files"]]
        required = {"host/__main__.py", "host/contracts.py", "host/runtime.py",
                    self.registration["profile"], record["source"]}
        if record["module_id"] == self.registration["entry"]["module_id"]:
            required.add(self.registration["plan"])
        if len(paths) != len(set(paths)) or not required <= set(paths):
            raise ModuleError("BindingMismatch", "incomplete or duplicate content closure")
        if not re.fullmatch(r"modules/temporary/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.py", record["source"]):
            raise ModuleError("BindingMismatch", "only controlled temporary module sources allowed")
        captured = {}
        for item in record["content_files"]:
            path = safe_path(self.project, item["path"])
            captured[item["path"]] = capture_source(path, item["sha256"])
            if item["path"].startswith("host/") and captured[item["path"]] != sys.modules["host"]._verified_host_sources.get(item["path"]):
                raise ModuleError("BindingMismatch", "executed host source does not match binding")
        if not allow_new and record != self.records.get((record["implementation_id"], record["version"])):
            raise ModuleError("BindingMismatch", "record is not the frozen approved candidate")
        return captured

    def inspect_binding(self, manifest, records):
        self.validator.check(manifest, "AssemblyManifest")
        if not manifest["implementations"]:
            raise ModuleError("BindingMismatch", "empty combination")
        if [identity(r) for r in records] != manifest["implementations"]:
            raise ModuleError("BindingMismatch", "registry response does not match manifest")
        selected = {}
        for record in records:
            self._verify_record(record)
            if record["module_id"] in selected:
                raise ModuleError("BindingMismatch", "duplicate module in combination")
            selected[record["module_id"]] = record
        edges = []
        interfaces = []
        for module in selected:
            for operation, contract in self.profile["modules"][module]["operations"].items():
                interfaces.append(ref(module, operation, self.interface_version))
                for target in contract["calls"]:
                    target_module, target_op = target.split("/")
                    if target_module not in selected or target_op not in self.profile["modules"][target_module]["operations"]:
                        raise ModuleError("BindingMismatch", "missing declared dependency: " + target)
                    edges.append({"caller": ref(module, operation, self.interface_version),
                                  "target": ref(target_module, target_op, self.interface_version)})
        roots = manifest["root_operations"]
        if not roots or len({digest(r) for r in roots}) != len(roots) or any(r not in interfaces for r in roots):
            raise ModuleError("BindingMismatch", "invalid root operation or exact interface version")
        return {**clone(manifest), "interfaces": interfaces, "dependency_edges": edges,
                "environment_ref": clone(self.environment), "combination_status": "temporary"}

    def install_binding(self, manifest, records):
        binding = self.inspect_binding(manifest, records)
        key = binding["binding_id"]
        if key in self.bindings:
            raise ModuleError("BindingMismatch", "binding ID is immutable and cannot be reused")
        # Load exactly these sources, without import of any legacy code or automatic invocation.
        for record in records:
            ident = record["content_digest"]
            if ident not in self.instances:
                captured = self._verify_record(record)
                module = execute_source("_kax_b00_" + ident, self.project / record["source"], captured[record["source"]])
                self.instances[ident] = module.Implementation(clone(record["settings"]))
        self.bindings[key] = clone(binding)
        return clone(binding)

    def new_task(self, run_id, binding_id):
        self.validator.check(run_id, "id")
        if run_id in self._closed_runs:
            raise ModuleError("BindingMismatch", "run has ended")
        if binding_id not in self.bindings:
            raise ModuleError("BindingMismatch", "unknown binding")
        if run_id in self.tasks and self.tasks[run_id] != binding_id:
            raise ModuleError("BindingMismatch", "existing run cannot drift to another binding")
        self.tasks[run_id] = binding_id
        return {"run_id": run_id, "binding_id": binding_id}

    def envelope(self, task, target, arguments, artifact_refs=None, parent=None, resources=None):
        self.sequence += 1
        return {"protocol_version": "0.1.0-draft", "target": clone(target),
                "context": {"run_id": task["run_id"], "binding_id": task["binding_id"],
                            "call_id": f"call-{self.sequence:05d}", "parent_call_id": parent,
                            "cancellation_ref": None, "trace_ref": "host-trace", "resource_scope_id": "isolated-run"},
                "arguments": arguments, "config_view": clone(EMPTY_CONFIG),
                "artifact_refs": artifact_refs or [],
                "resource_handles": [clone(self.resources.records[token]) for token in resources or []]}

    def invoke(self, envelope, parent=None):
        # Pre-0.1.7 trusted embedders/tests can construct dispatch state directly.
        if not hasattr(self, "_secrets"):
            self._secrets = SecretObservations()
        # Parent is an internal active frame, never supplied by an external JSON envelope.
        context = envelope.get("context", {}) if type(envelope) is dict else {}
        call_id = context.get("call_id", "invalid") if type(context) is dict else "invalid"
        binding_id = context.get("binding_id", "invalid") if type(context) is dict else "invalid"
        base = {"call_id": call_id if type(call_id) is str else "invalid",
                "binding_id": binding_id if type(binding_id) is str else "invalid"}

        def reject(code, detail):
            result = {**base, "outcome": "host_rejected", "effects": [],
                      "host_rejected": {"code": code, "execution_started": False,
                                        "detail": detail, "external_mapping": None}}
            self.trace.append({"event": "rejected", "result": self.redact(result)})
            return result
        try:
            request = clone(envelope)
            self.validator.check(request, "CallEnvelope")
        except (ContractError, UnicodeError, RecursionError):
            return reject("INVALID_ENVELOPE", "request is outside the frozen JSON envelope")
        ctx = request["context"]
        if ctx["call_id"] in self.used_calls:
            return reject("INVALID_ENVELOPE", "call ID reused")
        self.used_calls.add(ctx["call_id"])
        binding = self.bindings.get(binding_id)
        if binding is None or self.tasks.get(ctx["run_id"]) != binding_id or ctx["run_id"] in self._closed_runs:
            return reject("BINDING_MISMATCH", "run has no matching immutable binding")
        if ctx["trace_ref"] != "host-trace" or ctx["resource_scope_id"] != "isolated-run":
            return reject("INVALID_ENVELOPE", "unsupported trace or resource scope")
        if parent is None:
            if ctx["parent_call_id"] is not None:
                return reject("BINDING_MISMATCH", "external root cannot forge a parent")
        elif not parent.active or any(ctx[k] != parent.context[k] for k in
                ("run_id", "binding_id", "trace_ref", "resource_scope_id")) or ctx["parent_call_id"] != parent.context["call_id"]:
            return reject("BINDING_MISMATCH", "child must inherit active parent context")
        target = request["target"]
        module, operation = target["module_id"], target["operation_id"]
        selected = {r["module_id"]: r for r in binding["implementations"]}
        if module not in selected:
            return reject("UNKNOWN_MODULE", "module is outside this binding")
        declaration = self.profile["modules"][module]["operations"].get(operation)
        if declaration is None:
            return reject("UNKNOWN_OPERATION", "operation not declared")
        if target["interface_version"] != self.interface_version:
            return reject("VERSION_MISMATCH", "exact interface version required")
        if parent is None and target not in binding["root_operations"]:
            return reject("CAPABILITY_DENIED", "root operation not authorized")
        if parent is not None and module + "/" + operation not in parent.declaration["calls"]:
            return reject("UNDECLARED_DEPENDENCY", "caller did not declare this operation")
        try:
            self.validator.check(request["arguments"], declaration["input"])
        except ContractError:
            return reject("INVALID_ENVELOPE", "arguments do not match the declared signature")
        api = CallAPI(self, request, declaration)
        if parent is not None:
            api._stage_id = parent._stage_id
        for record in request["resource_handles"]:
            if (type(record) is not dict or type(record.get("id")) is not str
                    or record != self.resources.records.get(record["id"])
                    or record["run_id"] != ctx["run_id"] or record["state"] == "released"):
                return reject("CAPABILITY_DENIED", "invalid resource handle grant")
        try:
            # ArtifactRef inputs require an explicit grant even when only forwarded.
            for name, schema in declaration["input"].get("object", {}).items():
                if schema == "ArtifactRef" and request["arguments"][name] not in request["artifact_refs"]:
                    raise ContractError("forwarded artifact reference is not granted")
            for name, access in declaration["files"].items():
                api.path(request["arguments"][name], access)
        except (ContractError, OSError):
            return reject("CAPABILITY_DENIED", "artifact reference or path not authorized")
        self._active_calls += 1
        self._remember_fields(request["arguments"], declaration.get("secret_inputs", []))
        self.trace.append({"event": "started", "request": self.redact(request),
                           "implementation": clone(selected[module])})
        try:
            arguments = clone(request["arguments"])
            # Evidence/file-writing operations only receive the observable projection.
            # No writer allowlist and no path authorization bypass is introduced.
            if "write" in declaration["files"].values():
                arguments = self.redact(arguments)
                self.validator.check(arguments, declaration["input"])
            value, effects = self.instances[selected[module]["content_digest"]].invoke(operation, arguments, api)
            self._remember_fields(value, declaration.get("secret_outputs", []))
            self.validator.check(value, declaration["output"])
            self.validator.check(effects, {"array": "SideEffectRecord"})
            result = {**base, "outcome": "returned", "returned": value, "effects": effects}
        except ModuleError as exc:
            self.validator.check(exc.error, "BusinessError")
            result = {**base, "outcome": "raised", "raised": clone(exc.error), "effects": clone(exc.error["effects"])}
        except Exception as exc:
            unknown = effect({"module_id": module}, "write", "unknown", "unexpected module exception; side effects unknown")
            error = ModuleError("UnexpectedModuleError", str(exc), effects=[unknown], legacy_identity=type(exc).__name__).error
            result = {**base, "outcome": "raised", "raised": error, "effects": [unknown]}
        finally:
            try:
                self.resources.end_scope(ctx["run_id"], "call", ctx["call_id"])
            except ModuleError as exc:
                result = {**base, "outcome": "raised", "raised": clone(exc.error), "effects": clone(exc.error["effects"])}
            finally:
                api.active = False
                self._active_calls -= 1
        if self.resources.records:
            self.trace.append({"event": "resource_handles", "call_id": call_id,
                               "resource_handles": self.resources.snapshot()})
        if self._secrets.values:
            self.trace[:] = self.redact(self.trace)
        self.trace.append({"event": "finished", "result": self.redact(result)})
        return clone(result)


class CallAPI:
    """Capability-limited trusted-code facade. Not an untrusted Python sandbox."""
    def __init__(self, host, request, declaration):
        self._host = host
        self._request = clone(request)
        self.declaration = clone(declaration)
        self.context = clone(request["context"])
        self.active = True
        self._stage_id = None

    def fixture_directory(self):
        self._require("source_link_probes")
        path = safe_path(self._host.output, "fixtures/integrity")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def capture_fixture_source(self, relative, expected_sha256):
        self._require("source_link_probes")
        path = safe_path(self.fixture_directory(), relative)
        source = capture_source(path, expected_sha256)
        if not hasattr(self, "_captured_probes"):
            self._captured_probes = {}
        token = "source-" + str(len(self._captured_probes))
        self._captured_probes[token] = (path, source)
        return token

    def inspect_fixture_source(self, token):
        self._require("source_link_probes")
        path, source = self._captured_probes.pop(token)
        module = execute_source("_b00_source_probe", path, source)
        return clone(module.PROBE_VALUE)

    def inspect_fixture_path(self, locator):
        self._require("source_link_probes")
        try:
            safe_path(self._host.output, locator)
            return {"rejected": False, "reason": None}
        except ContractError as exc:
            return {"rejected": True, "reason": str(exc)}

    def copy_startup_fixture(self):
        self._require("source_link_probes")
        root = self.fixture_directory() / "startup-project"
        root.mkdir()
        files = sys.modules["host"]._source_tree_snapshot()
        plan = "experiments/specifications/b00-minimal.json"
        files[plan] = file_hash(safe_path(self._host.project, plan))
        for relative, expected in files.items():
            source = capture_source(safe_path(self._host.project, relative), expected)
            target = safe_path(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source)
        return root

    def source_tree_snapshot(self):
        self._require("source_link_probes")
        bootstrap = sys.modules["host"]
        return clone({"before": bootstrap._startup_files, "after": bootstrap._source_tree_snapshot(),
                      "dont_write_bytecode": sys.dont_write_bytecode, "startup_B": bool(sys.flags.dont_write_bytecode)})

    def _require(self, capability):
        if not self.active or capability not in self.declaration["host_capabilities"]:
            raise ModuleError("PermissionDenied", "host capability denied: " + capability)

    def _check_environment(self, name):
        if not self.active or type(name) is not str or name not in self.declaration.get("environment", []):
            raise ModuleError("PermissionDenied", "environment variable is not declared readable")

    def environment(self, name, default=None):
        """Missing names are still checked before reading; no ambient enumeration."""
        self._check_environment(name)
        with _ENVIRONMENT_LOCK:
            source = os.environ
            while isinstance(source, DeclaredEnvironment):
                source = source.source
            return DeclaredEnvironment(self, source).get(name, default)

    @contextlib.contextmanager
    def environment_scope(self):
        """Opt-in trusted legacy os.environ/os.getenv scope; restore identity always.

        Nested scopes use the underlying mapping, not the parent's grant: a child
        gets only its own declaration. Unrelated Python threads/native getenv and
        escaped pre-scope references are not sandboxed. No writes are supported.
        """
        if not self.active:
            raise ModuleError("PermissionDenied", "inactive environment scope")
        with _ENVIRONMENT_LOCK:
            previous = os.environ
            source = previous
            while isinstance(source, DeclaredEnvironment):
                source = source.source
            os.environ = DeclaredEnvironment(self, source)
            try:
                yield
            finally:
                os.environ = previous

    def redact(self, value):
        if not self.active:
            raise ModuleError("PermissionDenied", "inactive call")
        return self._host.redact(value)

    def legacy_modules(self):
        """0.1.5: fresh private legacy closure from SHA256-verified declared bytes."""
        self._require("legacy_source")
        return self._host.legacy_modules()

    def legacy_handle(self, scope="call"):
        """Opaque host-owned closure, including any unchanged legacy cache objects."""
        self._require("legacy_source")
        return self.resource_create("legacy_closure", scope, {})

    @property
    def resource_handles(self):
        """Detached records granted by this call envelope; never resource objects."""
        if not self.active:
            raise ModuleError("PermissionDenied", "resource call frame has ended")
        return clone(self._request["resource_handles"])

    def resource_create(self, kind, scope, options=None):
        return self._host.resources.create(self, kind, scope, {} if options is None else options)

    def resource_borrow(self, token, action, arguments):
        return self._host.resources.borrow(self, token, action, arguments)

    def resource_cancel(self, token):
        return self._host.resources.cancel(self, token)

    def resource_release(self, token):
        return self._host.resources.release(self, token)

    @contextlib.contextmanager
    def resource_stage(self):
        if not self.active:
            raise ModuleError("PermissionDenied", "inactive resource stage")
        prior = self._stage_id
        self._host._resource_stage_sequence += 1
        self._stage_id = "stage-" + str(self._host._resource_stage_sequence)
        try:
            yield self._stage_id
        finally:
            try:
                self._host.resources.end_scope(self.context["run_id"], "stage", self._stage_id)
            finally:
                self._stage_id = prior

    def call(self, target, arguments, artifacts=None, resources=None):
        if not self.active:
            raise ModuleError("PermissionDenied", "call frame has ended")
        for token in resources or []:
            self._host.resources._authorize(self, token, "forward")
        envelope = self._host.envelope(self.context, target, arguments, artifacts, self.context["call_id"], resources)
        return self._host.invoke(envelope, self)

    def path(self, resource, access):
        if not self.active:
            raise ContractError("inactive file capability")
        self._host.validator.check(resource, "ArtifactRef")
        permitted = [self._request["arguments"][k] for k, mode in self.declaration["files"].items() if mode == access]
        if resource not in permitted or resource not in self._request["artifact_refs"] or access not in resource["permissions"]:
            raise ContractError("undeclared artifact access")
        path = safe_path(self._host.output, resource["locator"])
        if access == "write" and path.exists():
            raise ContractError("fixture overwrite forbidden")
        if access == "write" and resource["writer_operation"] != self._request["target"]:
            raise ContractError("writer operation mismatch")
        if access == "read" and resource["content_identity"] is not None and file_hash(path) != resource["content_identity"]:
            raise ContractError("artifact content identity mismatch")
        return path

    def validate_record(self, record):
        self._require("validate_record")
        self._host._verify_record(record)

    def inspect_binding(self, manifest, records):
        self._require("inspect_binding")
        return self._host.inspect_binding(manifest, records)

    def install_binding(self, manifest, records):
        self._require("install_binding")
        return self._host.install_binding(manifest, records)

    def catalog_snapshot(self):
        self._require("catalog_snapshot")
        return clone(self._host.catalog["implementations"])

    def new_task(self, run_id, binding_id):
        self._require("new_task")
        return self._host.new_task(run_id, binding_id)

    def invoke_root(self, task, target, arguments, artifacts=None, overrides=None):
        self._require("invoke_root")
        envelope = self._host.envelope(task, target, arguments, artifacts)
        # Explicit malformed-envelope fault injection is confined to the test module.
        if overrides:
            envelope.update(clone(overrides))
        return self._host.invoke(envelope)

    def snapshot(self):
        self._require("snapshot")
        snapshot = {"environment": self._host.environment, "bindings": self._host.bindings,
                    "tasks": self._host.tasks, "trace": self._host.trace}
        if self._host.resources.records:
            snapshot["resource_handles"] = self._host.resources.snapshot()
        return self._host.redact(snapshot)
