from __future__ import annotations

from pathlib import Path

from qwen_asr.models import TranscriptSegment, WorkPaths
from qwen_asr.storage import read_json, write_json_atomic
from tools.mfa_full_alignment import run_mfa_full_alignment


def test_run_mfa_full_alignment_builds_batch_corpus_and_manifest(tmp_path: Path, monkeypatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    audio_1 = tmp_path / "segment_1.wav"
    audio_2 = tmp_path / "segment_2.wav"
    audio_1.write_bytes(b"RIFF")
    audio_2.write_bytes(b"RIFF")
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "tools.mfa_full_alignment.detect_mfa_environment",
        lambda run_version_check=True: {
            "available": True,
            "command": ["micro", "run", "-p", "env", "mfa"],
            "root_dir": str(tmp_path / "mfa-root"),
        },
    )

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        calls.append(command_list)
        output_dir = Path(command_list[9])
        write_json_atomic(
            output_dir / "segment_000001.json",
            {"tiers": {"words": {"entries": [[0.1, 0.3, "\u306f\u3044"]]}}},
        )
        write_json_atomic(
            output_dir / "segment_000002.json",
            {"tiers": {"words": {"entries": [[0.2, 0.6, "\u3046\u3093"]]}}},
        )
        return Completed()

    monkeypatch.setattr("tools.mfa_full_alignment.subprocess.run", fake_run)

    aligned, report = run_mfa_full_alignment(
        paths,
        [
            TranscriptSegment(
                segment_id="segment_000001",
                audio_path=str(audio_1),
                global_start_time=10.0,
                global_end_time=11.0,
                text="\u306f\u3044",
                language="Japanese",
            ),
            TranscriptSegment(
                segment_id="segment_000002",
                audio_path=str(audio_2),
                global_start_time=20.0,
                global_end_time=21.0,
                text="\u3046\u3093",
                language="Japanese",
            ),
        ],
        num_jobs=2,
    )

    assert len(calls) == 1
    assert calls[0][:5] == ["micro", "run", "-p", "env", "mfa"]
    assert calls[0][5] == "align"
    assert calls[0][10:12] == ["--clean", "--single_speaker"]
    assert calls[0][12:14] == ["--num_jobs", "2"]
    assert aligned[0].alignment_backend == "mfa"
    assert aligned[0].alignment_unit == "word"
    assert aligned[0].tokens[0].start_time == 10.1
    assert aligned[1].tokens[0].text == "\u3046\u3093"
    assert report["summary"]["completed_count"] == 2
    assert (paths.workdir / "experiments" / "mfa-full-align" / "corpus" / "segment_000001.lab").read_text(encoding="utf-8") == "\u306f\u3044"
    saved_report = read_json(paths.workdir / "reports" / "mfa_full_align.json")
    assert saved_report["alignment_backend"] == "mfa"


def test_run_mfa_full_alignment_records_rejected_inputs(tmp_path: Path, monkeypatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "tools.mfa_full_alignment.detect_mfa_environment",
        lambda run_version_check=True: {
            "available": True,
            "command": ["micro", "run", "-p", "env", "mfa"],
            "root_dir": "",
        },
    )

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        calls.append([str(value) for value in command])
        return Completed()

    monkeypatch.setattr("tools.mfa_full_alignment.subprocess.run", fake_run)

    aligned, report = run_mfa_full_alignment(
        paths,
        [
            TranscriptSegment(
                segment_id="segment_000001",
                audio_path=str(tmp_path / "missing.wav"),
                global_start_time=0.0,
                global_end_time=1.0,
                text="\u306f\u3044",
            )
        ],
    )

    assert len(calls) == 1
    assert aligned[0].status == "failed"
    assert aligned[0].alignment_failure_reason == "source-audio-missing"
    assert report["summary"]["input_rejected_count"] == 1
