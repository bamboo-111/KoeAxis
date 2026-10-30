from __future__ import annotations

import hashlib

from scripts import project_inventory


def test_candidate_record_hashes_files_larger_than_old_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(project_inventory, "ROOT", tmp_path)
    candidate = tmp_path / "large.bin"
    with candidate.open("wb") as handle:
        handle.seek(101 * 1024 * 1024)
        handle.write(b"x")

    expected = hashlib.sha256(candidate.read_bytes()).hexdigest()
    record = project_inventory.candidate_record(candidate, "ARCHIVE_CANDIDATE", "test")

    assert record["record_kind"] == "file"
    assert record["sha256_kind"] == "file"
    assert record["sha256"] == expected


def test_p2_status_does_not_count_directory_aggregate_as_missing_file_hash() -> None:
    candidates = [
        {
            "path": "large.bin",
            "record_kind": "file",
            "size_bytes": 10,
            "status": "ARCHIVE_CANDIDATE",
            "approval_required": True,
            "sha256": "abc",
        },
        {
            "path": "workspaces",
            "record_kind": "directory_aggregate",
            "size_bytes": 20,
            "status": "UNKNOWN",
            "approval_required": True,
            "sha256": None,
        },
    ]

    status = project_inventory.build_p2_approval_status(candidates, "2026-07-15T00:00:00+00:00")

    assert status["candidate_count"] == 2
    assert status["file_candidate_count"] == 1
    assert status["directory_aggregate_count"] == 1
    assert status["missing_file_sha256_count"] == 0
