from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = ROOT / "reports"
CURRENT_BASELINE_WORKSPACES = {"p6-final-regression-20260714-160541"}


@dataclass(frozen=True, slots=True)
class TreeStats:
    files: int
    bytes: int
    latest_mtime: float | None


def tree_stats(path: Path) -> TreeStats:
    if path.is_file():
        stat = path.stat()
        return TreeStats(1, stat.st_size, stat.st_mtime)
    files = 0
    size = 0
    latest: float | None = None
    if not path.exists():
        return TreeStats(0, 0, None)
    for item in path.rglob("*"):
        try:
            stat = item.stat()
        except OSError:
            continue
        latest = stat.st_mtime if latest is None else max(latest, stat.st_mtime)
        if item.is_file():
            files += 1
            size += stat.st_size
    return TreeStats(files, size, latest)


def iso_mtime(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def reference_documents(root: Path = ROOT) -> list[tuple[str, str]]:
    paths = list((root / "docs").glob("*.md"))
    paths.extend(
        path
        for path in (
            root / "reports" / "final_acceptance.md",
            root / "reports" / "rejected_feature_inventory.md",
        )
        if path.exists()
    )
    paths.extend((root / "benchmarks").glob("*.md"))
    documents = []
    for path in sorted(set(paths)):
        try:
            documents.append((path.relative_to(root).as_posix(), read_utf8(path)))
        except OSError:
            continue
    return documents


def references_for(relative_path: str, documents: list[tuple[str, str]]) -> list[str]:
    forward = relative_path.replace("\\", "/")
    backward = forward.replace("/", "\\")
    return [name for name, text in documents if forward in text or backward in text]


def classify_workspace(name: str, references: list[str]) -> tuple[str, str]:
    if name in CURRENT_BASELINE_WORKSPACES:
        return "CURRENT_BASELINE", "Latest named P6 final-regression workspace; retain as the current acceptance baseline."
    if references:
        return "KEY_EVIDENCE", "Referenced by a current or historical acceptance, benchmark, or planning document."
    return "UNKNOWN", "No authoritative document reference found; no archive or delete action is inferred."


def backup_topic(name: str) -> str:
    return re.sub(r"-\d{8}-\d{6}$", "", name)


def path_record(path: Path, root: Path = ROOT) -> dict[str, object]:
    stats = tree_stats(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "files": stats.files,
        "size_bytes": stats.bytes,
        "modified_at": iso_mtime(stats.latest_mtime),
    }


def parse_pyvenv(path: Path) -> dict[str, str]:
    config = path / "pyvenv.cfg"
    if not config.exists():
        return {}
    result: dict[str, str] = {}
    for line in read_utf8(config).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def dist_version(env_path: Path, distribution: str) -> str:
    site_packages = env_path / "Lib" / "site-packages"
    prefix = distribution.lower().replace("-", "_") + "-"
    for path in site_packages.glob("*.dist-info"):
        name = path.name.lower()
        if name.startswith(prefix) and name.endswith(".dist-info"):
            return path.name[len(prefix) : -len(".dist-info")]
    return ""


def model_revision(model_dir: Path) -> str:
    ref = model_dir / "refs" / "main"
    return ref.read_text(encoding="ascii").strip() if ref.exists() else ""


def build_workspace_records(root: Path, documents: list[tuple[str, str]]) -> list[dict[str, object]]:
    workspace_root = root / "workspaces"
    records = []
    for path in sorted((item for item in workspace_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        relative = path.relative_to(root).as_posix()
        references = references_for(relative, documents)
        status, reason = classify_workspace(path.name, references)
        records.append({**path_record(path, root), "status": status, "reason": reason, "references": references})
    return records


def build_tmp_records(root: Path, documents: list[tuple[str, str]]) -> list[dict[str, object]]:
    tmp_root = root / "tmp_manual_tests"
    records = []
    for path in sorted((item for item in tmp_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        relative = path.relative_to(root).as_posix()
        references = references_for(relative, documents)
        status = "KEY_EVIDENCE" if references else "UNKNOWN"
        reason = (
            "Referenced by an acceptance or historical evidence document; retain in place."
            if references
            else "No authoritative replacement mapping found; no archive or delete action is inferred."
        )
        records.append({**path_record(path, root), "status": status, "reason": reason, "references": references})
    return records


def build_backup_records(root: Path) -> list[dict[str, object]]:
    backup_root = root / "backups"
    records = []
    for path in sorted((item for item in backup_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        records.append(
            {
                **path_record(path, root),
                "topic": backup_topic(path.name),
                "status": "KEEP_ROLLBACK",
                "reason": "Retain until the associated code is in a stable commit and has passed a later full regression.",
            }
        )
    return records


def local_state_entry(
    root: Path,
    relative: str,
    *,
    owner: str,
    purpose: str,
    rebuild: str,
    retention: str,
    status: str,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    path = root / relative
    return {
        **path_record(path, root),
        "owner": owner,
        "purpose": purpose,
        "rebuild": rebuild,
        "retention": retention,
        "status": status,
        "details": details or {},
    }


def build_models(root: Path) -> list[dict[str, object]]:
    cache = root / ".model-cache"
    specs = [
        (
            "Qwen/Qwen3-ASR-1.7B",
            cache / "models--Qwen--Qwen3-ASR-1.7B",
            "Populate through the Qwen ASR loader with local_files_only disabled once, or download the pinned revision into .model-cache.",
        ),
        (
            "Qwen/Qwen3-ForcedAligner-0.6B",
            cache / "models--Qwen--Qwen3-ForcedAligner-0.6B",
            "Populate through the Qwen aligner loader with local_files_only disabled once, or download the pinned revision into .model-cache.",
        ),
        (
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            cache / "faster-whisper" / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
            "Re-download with faster-whisper/WhisperX into .model-cache/faster-whisper when that experiment is explicitly needed.",
        ),
    ]
    records = []
    for model_id, path, rebuild in specs:
        records.append(
            {
                "model_id": model_id,
                **path_record(path, root),
                "revision": model_revision(path),
                "status": "KEEP_REBUILDABLE",
                "rebuild": rebuild,
            }
        )
    mdx = cache / "mdx-net"
    records.append(
        {
            "model_id": "UVR-MDX-NET-Inst_HQ_3.onnx",
            **path_record(mdx, root),
            "revision": "local model filename",
            "status": "KEEP_REBUILDABLE",
            "rebuild": "Re-download through the configured audio-separator model provider into .model-cache/mdx-net.",
        }
    )
    return records


def build_reports(root: Path = ROOT) -> tuple[dict[str, object], dict[str, object]]:
    documents = reference_documents(root)
    workspaces = build_workspace_records(root, documents)
    tmp_records = build_tmp_records(root, documents)
    backups = build_backup_records(root)

    primary_env = parse_pyvenv(root / ".venv312")
    whisperx_legacy = parse_pyvenv(root / ".venv-whisperx")
    whisperx_312 = parse_pyvenv(root / ".venv-whisperx312")
    mfa_meta = next((root / "tools" / "mfa-env" / "conda-meta").glob("montreal-forced-aligner-*.json"), None)
    mfa_version = "3.4.0" if mfa_meta and "3.4.0" in mfa_meta.name else ""

    entries = [
        local_state_entry(
            root,
            ".venv312",
            owner="Project development and production runtime",
            purpose="Primary Python environment for compile, tests, Ruff, CLI, and model execution.",
            rebuild="Create with Python 3.12, then install requirements.txt and requirements-dev.txt.",
            retention="KEEP_PRIMARY; do not move because virtual environments may contain absolute paths.",
            status="KEEP_PRIMARY",
            details={"python_version": primary_env.get("version", ""), "ruff_version": "0.15.21"},
        ),
        local_state_entry(
            root,
            ".venv-whisperx",
            owner="Legacy WhisperX experiment",
            purpose="Python 3.14 virtual-environment shell; no WhisperX, faster-whisper, or torch distribution was detected.",
            rebuild="No verified rebuild is required because no active code or document references this shell.",
            retention="UNKNOWN; retain in place until a path-level decision is approved.",
            status="UNKNOWN",
            details={"python_version": whisperx_legacy.get("version", ""), "whisperx_version": ""},
        ),
        local_state_entry(
            root,
            ".venv-whisperx312",
            owner="WhisperX/faster-whisper experiment",
            purpose="Functional Python 3.12 experiment environment for WhisperX 3.8.6 and faster-whisper 1.2.1; not used by the production Qwen path.",
            rebuild="Create with Python 3.12 and install whisperx==3.8.6 with its compatible torch/faster-whisper stack.",
            retention="KEEP_LOCAL_EXPERIMENT until its historical evidence is no longer required and a path-level decision is approved.",
            status="KEEP_LOCAL_EXPERIMENT",
            details={
                "python_version": whisperx_312.get("version", ""),
                "whisperx_version": dist_version(root / ".venv-whisperx312", "whisperx"),
                "faster_whisper_version": dist_version(root / ".venv-whisperx312", "faster_whisper"),
                "torch_version": dist_version(root / ".venv-whisperx312", "torch"),
            },
        ),
        local_state_entry(
            root,
            ".model-cache",
            owner="Project model runtime",
            purpose="Project-local Qwen, faster-whisper, and MDX model cache used to keep local_files_only execution deterministic.",
            rebuild="Download the listed model IDs/revisions into .model-cache with network access explicitly enabled.",
            retention="KEEP_REBUILDABLE; large downloads are ignored and never auto-deleted.",
            status="KEEP_REBUILDABLE",
        ),
        local_state_entry(
            root,
            "tools/mfa-env",
            owner="Optional MFA experiment runtime",
            purpose="Micromamba prefix containing Montreal Forced Aligner and its native dependencies.",
            rebuild="Follow docs/PIPELINE.md to create MFA 3.4.0 under tools/mfa-env.",
            retention="KEEP_LOCAL_EXPERIMENT; ignored and not part of production source.",
            status="KEEP_LOCAL_EXPERIMENT",
            details={"mfa_version": mfa_version, "python_version": "3.13.14"},
        ),
        local_state_entry(
            root,
            "tools/mfa-root",
            owner="Optional MFA experiment runtime",
            purpose="Project-local MFA models, corpora, caches, configuration, and command history.",
            rebuild="Set MFA_ROOT_DIR and download japanese_mfa acoustic and dictionary models as documented in docs/PIPELINE.md.",
            retention="KEEP_KEY_EVIDENCE; command history and models support historical MFA reports.",
            status="KEEP_KEY_EVIDENCE",
        ),
        local_state_entry(
            root,
            "tools/micromamba",
            owner="Optional MFA experiment runtime",
            purpose="Project-local micromamba bootstrap and executable.",
            rebuild="Download the current Windows micromamba package and extract it as documented in docs/PIPELINE.md.",
            retention="KEEP_REBUILDABLE; ignored and not production source.",
            status="KEEP_REBUILDABLE",
            details={"verified_version": "2.8.1"},
        ),
        local_state_entry(
            root,
            "workspaces",
            owner="Pipeline experiments and acceptance evidence",
            purpose="127 independent workspaces containing current baselines, historical evidence, and unclassified experiments.",
            rebuild="Individual workspaces require their recorded input media, commands, model revisions, and manifests; they are not globally rebuildable from source alone.",
            retention="Mixed: CURRENT_BASELINE and KEY_EVIDENCE are retained; UNKNOWN remains in place pending review.",
            status="MIXED_REVIEWED",
            details={
                "workspace_count": len(workspaces),
                "status_counts": count_statuses(workspaces),
            },
        ),
        local_state_entry(
            root,
            "backups",
            owner="Rollback evidence for uncommitted cleanup work",
            purpose=f"{len(backups)} topic/timestamp backup directories created before file edits.",
            rebuild="Not rebuildable by design; each backup preserves a pre-edit state.",
            retention="KEEP_ROLLBACK until a stable commit exists and a later full regression passes.",
            status="KEEP_ROLLBACK",
        ),
        local_state_entry(
            root,
            "tmp_manual_tests",
            owner="Historical manual experiments",
            purpose="96 manual-test directories containing mixed derived data and at least one still-referenced reliable ASS path.",
            rebuild="Mixed and not globally reproducible; use per-directory evidence before considering archive.",
            retention="Mixed KEY_EVIDENCE/UNKNOWN; the directory is not proven replaceable by formal workspaces.",
            status="MIXED_REVIEWED",
            details={"directory_count": len(tmp_records), "status_counts": count_statuses(tmp_records)},
        ),
    ]

    register = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "models": build_models(root),
        "backup_directories": backups,
    }
    reference_map = {
        "generated_at": register["generated_at"],
        "reference_documents": [name for name, _text in documents],
        "workspaces": workspaces,
        "tmp_manual_tests": tmp_records,
        "summary": {
            "workspace_count": len(workspaces),
            "workspace_status_counts": count_statuses(workspaces),
            "tmp_directory_count": len(tmp_records),
            "tmp_status_counts": count_statuses(tmp_records),
            "missing_referenced_paths": sum(
                not (root / record["path"]).exists()
                for record in [*workspaces, *tmp_records]
                if record["references"]
            ),
        },
    }
    return register, reference_map


def count_statuses(records: list[dict[str, object]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        result[status] = result.get(status, 0) + 1
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def write_register_markdown(path: Path, register: dict[str, object]) -> None:
    lines = [
        "# Local State Register",
        "",
        f"Generated: `{register['generated_at']}`",
        "",
        "No path in this report was moved, archived, or deleted.",
        "",
        "| Path | Size | Status | Owner | Purpose | Rebuild | Retention |",
        "|---|---:|---|---|---|---|---|",
    ]
    for item in register["entries"]:
        lines.append(
            f"| `{item['path']}` | {format_size(item['size_bytes'])} | {item['status']} | {item['owner']} | "
            f"{item['purpose']} | {item['rebuild']} | {item['retention']} |"
        )
    lines.extend(["", "## Model Cache", "", "| Model | Revision | Size | Status | Rebuild |", "|---|---|---:|---|---|"])
    for item in register["models"]:
        lines.append(
            f"| `{item['model_id']}` | `{item['revision']}` | {format_size(item['size_bytes'])} | "
            f"{item['status']} | {item['rebuild']} |"
        )
    lines.extend(["", "## Backup Directories", "", f"Count: `{len(register['backup_directories'])}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reference_markdown(path: Path, reference_map: dict[str, object]) -> None:
    summary = reference_map["summary"]
    lines = [
        "# Evidence Reference Map",
        "",
        f"Generated: `{reference_map['generated_at']}`",
        "",
        f"Workspaces: `{summary['workspace_count']}`",
        f"Workspace statuses: `{json.dumps(summary['workspace_status_counts'], sort_keys=True)}`",
        f"Manual-test directories: `{summary['tmp_directory_count']}`",
        f"Manual-test statuses: `{json.dumps(summary['tmp_status_counts'], sort_keys=True)}`",
        f"Missing referenced paths: `{summary['missing_referenced_paths']}`",
        "",
        "## Workspaces",
        "",
        "| Path | Status | References | Reason |",
        "|---|---|---|---|",
    ]
    for item in reference_map["workspaces"]:
        refs = "<br>".join(f"`{ref}`" for ref in item["references"])
        lines.append(f"| `{item['path']}` | {item['status']} | {refs} | {item['reason']} |")
    lines.extend(["", "## tmp_manual_tests", "", "| Path | Status | References | Reason |", "|---|---|---|---|"])
    for item in reference_map["tmp_manual_tests"]:
        refs = "<br>".join(f"`{ref}`" for ref in item["references"])
        lines.append(f"| `{item['path']}` | {item['status']} | {refs} | {item['reason']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit protected local state and evidence references.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--check", action="store_true", help="Build the audit in memory without writing reports.")
    args = parser.parse_args()

    register, reference_map = build_reports()
    summary = {
        "backups": len(register["backup_directories"]),
        **reference_map["summary"],
    }
    if args.check:
        print(json.dumps(summary, sort_keys=True))
        return 0

    reports_dir = Path(args.reports_dir)
    write_json(reports_dir / "local_state_register.json", register)
    write_register_markdown(reports_dir / "local_state_register.md", register)
    write_json(reports_dir / "evidence_reference_map.json", reference_map)
    write_reference_markdown(reports_dir / "evidence_reference_map.md", reference_map)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
