"""Trusted bootstrap and shared no-follow path guard; -B is mandatory."""
import sys

sys.dont_write_bytecode = True
if not sys.flags.dont_write_bytecode:
    raise RuntimeError("B00 startup requires python -B -m host")

import hashlib
import importlib.util
import json
import pathlib
import re
import stat


def guarded_path(root, relative, error_type=ValueError):
    """Validate lexical containment and lstat every component before any access.

    Shared by bootstrap reads/walks and runtime. Not a concurrent-race sandbox.
    """
    if type(relative) is not str or not relative or "\\" in relative:
        raise error_type("relative portable path required")
    parts = relative.split("/")
    for part in parts:
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in (".", "..") or
                part.endswith((".", " ")) or part.split(".")[0].upper() in
                {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                 *[f"LPT{i}" for i in range(1, 10)]}):
            raise error_type("unsafe path component")
    root = pathlib.Path(root).absolute()
    current = root
    paths = [*reversed(root.parents), root]
    for part in parts:
        current = current / part
        paths.append(current)
    for path in paths:
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise error_type("links/reparse points are not allowed")
    return current


_project = pathlib.Path(__file__).absolute().parent.parent
# 0.1.5: plans/profiles/catalogs are registered here; every registered catalog
# must agree on host source identities (one trusted host for all plans).
PLAN_REGISTRY = "module-registry/plans.json"
_REGISTERED_PATH = r"module-registry/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.json"
_plan_registry = json.loads(guarded_path(_project, PLAN_REGISTRY).read_text("utf-8"))
_expected = {}
for _plan in _plan_registry["plans"]:
    if not (re.fullmatch(_REGISTERED_PATH, _plan["profile"]) and re.fullmatch(_REGISTERED_PATH, _plan["catalog"])):
        raise RuntimeError("profile/catalog must be module-registry/<pkg>/<file>.json")
    _catalog = json.loads(guarded_path(_project, _plan["catalog"]).read_text("utf-8"))
    for _record in _catalog["implementations"]:
        for _file in _record["content_files"]:
            if _file["path"].startswith("host/"):
                if _file["path"] in _expected and _expected[_file["path"]] != _file["sha256"]:
                    raise RuntimeError("inconsistent host source identities")
                _expected[_file["path"]] = _file["sha256"]
_verified_host_sources = {}
for _name in ("__main__", "contracts", "runtime"):
    _path = "host/" + _name + ".py"
    _data = guarded_path(_project, _path).read_bytes()
    if hashlib.sha256(_data).hexdigest() != _expected.get(_path):
        raise RuntimeError("host source identity mismatch: " + _path)
    _verified_host_sources[_path] = _data


def _source_tree_snapshot():
    files = {}

    def visit(relative):
        path = guarded_path(_project, relative)
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            for child in sorted(path.iterdir()):
                visit(child.relative_to(_project).as_posix())
        elif stat.S_ISREG(mode):
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            raise ValueError("unsupported source tree entry")

    for tree in ("host", "modules", "module-registry"):
        visit(tree)
    return files


_startup_files = _source_tree_snapshot()


class _SnapshotLoader:
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def get_code(self, fullname):
        return compile(_verified_host_sources[self.path], str(_project / self.path), "exec")

    def exec_module(self, module):
        module.__file__ = str(_project / self.path)
        exec(self.get_code(module.__name__), module.__dict__)


class _SnapshotFinder:
    def find_spec(self, fullname, path=None, target=None):
        relative = fullname.replace(".", "/") + ".py"
        if relative in _verified_host_sources:
            loader = _SnapshotLoader(relative)
            return importlib.util.spec_from_loader(fullname, loader, origin=str(_project / relative))
        return None


sys.meta_path.insert(0, _SnapshotFinder())
