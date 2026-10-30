"""Validate an explicitly supplied current-run finish handoff before any stage runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class InvalidHandoff(ValueError):
    pass


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(path: Path, expected: object) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise InvalidHandoff("invalid SHA256")
    if not path.is_file() or _digest(path) != expected:
        raise InvalidHandoff(f"missing file or SHA256 mismatch: {path}")


def validate_handoff(manifest_path: str | Path, current: bool = False) -> dict:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != "real-chain-finish-handoff-v1" or manifest.get("status") != "ready":
            raise InvalidHandoff("handoff is not ready or has the wrong schema")
        workdir = Path(manifest["workdir"])
        if not workdir.is_absolute() or not workdir.is_dir():
            raise InvalidHandoff("workdir must be an existing absolute directory")
        workdir = workdir.resolve()
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            raise InvalidHandoff("files must be a nonempty mapping")
        required = {"aligned_segments.json", "transcript_segments.json", "segments.json", "split_segments.json", "translated_segments.json", "mimo_proofread_segments.json", "mimo_proofread_report.json"}
        if not required.issubset({Path(name).name for name in files}):
            raise InvalidHandoff("missing required business artifacts")
        for name, sha in files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise InvalidHandoff("file path is not relative to workdir")
            target = (workdir / relative).resolve()
            if workdir not in target.parents:
                raise InvalidHandoff("file path escapes workdir")
            _check(target, sha)
        audio = manifest["source_audio"]
        if not isinstance(audio, dict):
            raise InvalidHandoff("source_audio must be an object")
        audio_path = Path(audio["path"])
        if not audio_path.is_absolute():
            raise InvalidHandoff("source_audio path must be absolute")
        _check(audio_path, audio["sha256"])
        lineage = manifest.get("lineage")
        if not isinstance(lineage, dict) or not lineage.get("llm") or not lineage.get("mimo"):
            raise InvalidHandoff("missing LLM or MiMo lineage")
        resources = manifest.get("resources", [])
        if not isinstance(resources, list):
            raise InvalidHandoff("resources must be a list")
        for resource in resources:
            if not isinstance(resource, dict) or not resource.get("purpose"):
                raise InvalidHandoff("invalid resource")
            resource_path = Path(resource["path"])
            if not resource_path.is_absolute():
                raise InvalidHandoff("resource path must be absolute")
            _check(resource_path, resource["sha256"])
        if current:
            expected = {
                'aligned_segments.json': '2cdd86f4e31946a931541db7b407e7fbc22bbd5451dcd589007996f41b9d71c8',
                'transcript_segments.json': 'b76058c6f4330c577c25eef4e82a91800682a5c31c4a181db4a23c34cb56d2b2',
            }
            for basename, expected_sha in expected.items():
                matches = [sha for name, sha in files.items() if Path(name).name == basename]
                if matches != [expected_sha]:
                    raise InvalidHandoff('current input digest mismatch: ' + basename)
            allowed = Path(__file__).resolve().parents[3] / 'runs/experiments'
            if allowed.resolve() not in workdir.parents or 'CURRENT' not in workdir.relative_to(allowed.resolve()).parts[0]:
                raise InvalidHandoff('workdir is not a current-run output')
            reports = [workdir / name for name in files if Path(name).name == 'mimo_proofread_report.json']
            if len(reports) != 1:
                raise InvalidHandoff('MiMo report is ambiguous')
            report = json.loads(reports[0].read_text(encoding='utf-8'))
            if (report.get('mode') != 'two-stage-nearby' or report.get('stage1_failed') != 0
                    or report.get('stage2_failed') != 0 or report.get('audio_review_unresolved_count') != 0
                    or report.get('stage2_completed') != report.get('audio_review_candidate_count')):
                raise InvalidHandoff('MiMo report does not establish business completion')
        return manifest
    except InvalidHandoff:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise InvalidHandoff(f"invalid handoff: {exc}") from exc
