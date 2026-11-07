"""Content storage for uploads, preset snapshots and artifacts, plus append-only run logs.

`BlobStore` is the extension point: the default `LocalBlobStore` keeps content-addressed files
(sha256) on disk; an S3/OSS implementation only needs the same five methods.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol

CHUNK = 1024 * 1024


class BlobTooLarge(ValueError):
    pass


class BlobStore(Protocol):
    def put(self, stream: BinaryIO, *, max_bytes: int | None = None) -> tuple[str, int]:
        """Store the stream; return (sha256, size). Identical content is stored once."""

    def open(self, sha256: str) -> BinaryIO: ...

    def local_path(self, sha256: str) -> Path | None:
        """A filesystem path if the store is local (lets the server use efficient file responses)."""

    def exists(self, sha256: str) -> bool: ...

    def delete(self, sha256: str) -> None: ...


class LocalBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            raise ValueError("invalid blob hash")
        return self.root / sha256[:2] / sha256

    def put(self, stream: BinaryIO, *, max_bytes: int | None = None) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        tmp_dir = self.root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(dir=tmp_dir)
        try:
            with os.fdopen(handle, "wb") as out:
                while True:
                    chunk = stream.read(CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise BlobTooLarge(f"upload exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    out.write(chunk)
            sha = digest.hexdigest()
            target = self._path(sha)
            if target.exists():
                os.unlink(tmp_name)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp_name, target)
            return sha, size
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def open(self, sha256: str) -> BinaryIO:
        return self._path(sha256).open("rb")

    def local_path(self, sha256: str) -> Path | None:
        path = self._path(sha256)
        return path if path.exists() else None

    def exists(self, sha256: str) -> bool:
        return self._path(sha256).exists()

    def delete(self, sha256: str) -> None:
        path = self._path(sha256)
        if path.exists():
            path.unlink()


class LogStore:
    """Append-only plain-text log per run, read incrementally by byte offset."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if not run_id.isalnum():
            raise ValueError("invalid run id")
        return self.root / f"{run_id}.log"

    def append(self, run_id: str, text: str) -> int:
        data = text.encode("utf-8", errors="replace")
        with self._path(run_id).open("ab") as handle:
            handle.write(data)
            return handle.tell()

    def size(self, run_id: str) -> int:
        path = self._path(run_id)
        return path.stat().st_size if path.exists() else 0

    def read(self, run_id: str, offset: int = 0, limit: int = 256 * 1024) -> tuple[str, int]:
        path = self._path(run_id)
        if not path.exists():
            return "", 0
        with path.open("rb") as handle:
            handle.seek(max(0, offset))
            data = handle.read(max(1, limit))
            end = handle.tell()
        cut = _complete_utf8_length(data)
        end -= len(data) - cut
        return data[:cut].decode("utf-8", errors="replace"), end

    def path(self, run_id: str) -> Path:
        return self._path(run_id)

    def delete(self, run_id: str) -> None:
        path = self._path(run_id)
        if path.exists():
            path.unlink()


def copy_stream(source: BinaryIO, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        shutil.copyfileobj(source, out, CHUNK)


def _complete_utf8_length(data: bytes) -> int:
    """Length of `data` without a trailing incomplete UTF-8 sequence."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0xC0 == 0x80:      # continuation byte, keep looking for the lead byte
            continue
        if byte < 0x80:
            return len(data)
        need = 2 if byte >= 0xC0 and byte < 0xE0 else 3 if byte < 0xF0 else 4
        return len(data) if back >= need else len(data) - back
    return len(data)
