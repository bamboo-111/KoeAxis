from __future__ import annotations

from pathlib import Path

from qwen_asr.mfa_experiment import _ffmpeg_extract_clip, run_local_mfa_alignment_experiments as legacy_run_local
from qwen_asr.mfa_runner import (
    PRUNE_DICTIONARY_ENV,
    PRUNED_DICTIONARY_FILE,
    ffmpeg_extract_clip,
    prefetch_local_mfa_alignments,
    prune_dictionary_enabled,
    read_mfa_batch_cache,
    run_local_mfa_alignment_experiments,
    write_pruned_mfa_dictionary,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


class Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_local_mfa_alignment_skips_missing_command(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")

    result = run_local_mfa_alignment_experiments(
        paths,
        [{"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u3044"}],
        environment={"available": True, "command": []},
        max_run_candidates=1,
        padding_ms=100,
    )

    assert result == [{"status": "skipped", "reason": "mfa-command-missing"}]


def test_ffmpeg_extract_clip_builds_expected_command(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        calls.append([str(value) for value in command])
        return Completed()

    result = ffmpeg_extract_clip(
        tmp_path / "source.wav",
        tmp_path / "clip.wav",
        start_ms=250,
        end_ms=1750,
        run_command=fake_run,
    )

    assert result["status"] == "completed"
    assert calls[0][0] == "ffmpeg"
    assert calls[0][calls[0].index("-ss") + 1] == "0.250"
    assert calls[0][calls[0].index("-t") + 1] == "1.500"


def test_run_local_mfa_alignment_injects_root_and_parses_words(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        calls.append({"command": command_list, "env": kwargs.get("env", {})})
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        output_dir = Path(command_list[command_list.index("--clean") - 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            output_dir / "clip.json",
            {"tiers": {"words": {"entries": [[0.12, 0.34, "\u306f\u3044"]]}}},
        )
        return Completed(stdout="ok")

    result = run_local_mfa_alignment_experiments(
        paths,
        [
            {
                "source": "content-quality",
                "reason": "missing_short_response",
                "severity": "FAIL",
                "start_ms": 1000,
                "end_ms": 1300,
                "text": "\u306f\u3044",
                "details": {"previous_score": 0.2, "current_score": 0.0},
            }
        ],
        environment={"available": True, "command": ["mfa"], "root_dir": str(tmp_path / "mfa-root")},
        max_run_candidates=1,
        padding_ms=100,
        run_command=fake_run,
        monotonic=iter([1.0, 1.25]).__next__,
        environ_factory=lambda: {},
    )

    assert result[0]["status"] == "completed"
    assert result[0]["elapsed_ms"] == 250
    assert result[0]["global_word_ranges"] == [{"start_ms": 1020, "end_ms": 1240, "text": "\u306f\u3044"}]
    assert calls[1]["env"]["MFA_ROOT_DIR"] == str(tmp_path / "mfa-root")


def test_mfa_experiment_runner_compatibility_aliases(tmp_path: Path, monkeypatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        calls.append(command_list)
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        output_dir = Path(command_list[command_list.index("--clean") - 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            output_dir / "clip.json",
            {"tiers": {"words": {"entries": [[0.1, 0.2, "\u306f\u3044"]]}}},
        )
        return Completed()

    monkeypatch.setattr("qwen_asr.mfa_experiment.subprocess.run", fake_run)

    result = legacy_run_local(
        paths,
        [{"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u3044", "details": {"previous_score": 0.0}}],
        environment={"available": True, "command": ["mfa"], "root_dir": ""},
        max_run_candidates=1,
        padding_ms=0,
    )

    assert result[0]["status"] == "completed"
    assert calls[0][0] == "ffmpeg"
    assert calls[1][0] == "mfa"
    assert _ffmpeg_extract_clip(tmp_path / "source.wav", tmp_path / "clip.wav", start_ms=0, end_ms=1)["status"] in {
        "completed",
        "failed",
    }


def _fake_mfa(calls: list[list[str]], *, returncode: int = 0):  # noqa: ANN202
    """ffmpeg writes the clip; mfa writes one json per .lab found in the corpus (batch or single layout)."""

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        calls.append(command_list)
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        corpus_dir = Path(command_list[command_list.index("align") + 1])
        output_dir = Path(command_list[command_list.index("--clean") - 1])
        if returncode == 0:
            for lab in corpus_dir.rglob("*.lab"):
                target = output_dir / lab.parent.relative_to(corpus_dir) / f"{lab.stem}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                text = lab.read_text(encoding="utf-8")
                write_json_atomic(target, {"tiers": {"words": {"entries": [[0.1, 0.4, text]]}}})
        return Completed(returncode=returncode, stderr="" if returncode == 0 else "boom")

    return fake_run


def _candidates() -> list[dict[str, object]]:
    return [
        {"subtitle_id": "a", "start_ms": 1000, "end_ms": 1500, "text": "\u306f\u3044"},
        {"subtitle_id": "b", "start_ms": 5000, "end_ms": 5600, "text": "\u3046\u3093"},
    ]


def test_prefetch_aligns_all_candidates_in_one_mfa_call_then_serves_cache(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []
    environment = {"available": True, "command": ["mfa"], "root_dir": ""}

    summary = prefetch_local_mfa_alignments(
        paths, _candidates(), environment=environment, padding_ms=100, run_command=_fake_mfa(calls),
        environ_factory=lambda: {},
    )

    mfa_calls = [call for call in calls if call[0] == "mfa"]
    assert summary["status"] == "completed" and summary["aligned_count"] == 2
    assert len(mfa_calls) == 1
    # one speaker directory per clip keeps speaker adaptation identical to individual runs
    assert "--single_speaker" not in mfa_calls[0]

    calls.clear()
    results = [
        run_local_mfa_alignment_experiments(
            paths, [candidate], environment=environment, max_run_candidates=1, padding_ms=100,
            run_command=_fake_mfa(calls), environ_factory=lambda: {},
        )[0]
        for candidate in _candidates()
    ]

    assert calls == []
    assert [row["status"] for row in results] == ["completed", "completed"]
    assert results[0]["global_word_ranges"] == [{"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u3044"}]
    assert results[1]["global_word_ranges"] == [{"start_ms": 5000, "end_ms": 5300, "text": "\u3046\u3093"}]
    assert results[0]["batch_cache"]["batch_size"] == 2


def test_cache_miss_on_changed_text_runs_individual_clean_mfa(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []
    environment = {"available": True, "command": ["mfa"], "root_dir": ""}
    prefetch_local_mfa_alignments(
        paths, _candidates(), environment=environment, padding_ms=100, run_command=_fake_mfa(calls),
        environ_factory=lambda: {},
    )
    calls.clear()

    result = run_local_mfa_alignment_experiments(
        paths, [{"start_ms": 1000, "end_ms": 1500, "text": "\u3048\u3048"}], environment=environment,
        max_run_candidates=1, padding_ms=100, run_command=_fake_mfa(calls), environ_factory=lambda: {},
    )[0]

    mfa_calls = [call for call in calls if call[0] == "mfa"]
    assert len(mfa_calls) == 1 and "--clean" in mfa_calls[0] and "--single_speaker" in mfa_calls[0]
    assert "batch_cache" not in result
    assert result["global_word_ranges"][0]["text"] == "\u3048\u3048"


def test_failed_prefetch_caches_nothing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []
    environment = {"available": True, "command": ["mfa"], "root_dir": ""}

    summary = prefetch_local_mfa_alignments(
        paths, _candidates(), environment=environment, padding_ms=100,
        run_command=_fake_mfa(calls, returncode=1), environ_factory=lambda: {},
    )

    assert summary["status"] == "failed" and summary["aligned_count"] == 0
    assert read_mfa_batch_cache(paths) == {}
    again = prefetch_local_mfa_alignments(
        paths, _candidates(), environment=environment, padding_ms=100, run_command=_fake_mfa(calls),
        environ_factory=lambda: {},
    )
    assert again["aligned_count"] == 2
    assert prefetch_local_mfa_alignments(
        paths, _candidates(), environment=environment, padding_ms=100, run_command=_fake_mfa(calls),
        environ_factory=lambda: {},
    ) == {"status": "completed", "aligned_count": 0, "already_cached": 2}


def _mfa_root_with_dictionary(tmp_path: Path) -> Path:
    root = tmp_path / "mfa-root"
    dictionary = root / "pretrained_models" / "dictionary" / "japanese_mfa.dict"
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text(
        "\u306f\u3044\th a i\n\u3046\u3093\tɯ N\n\u3048\u3048\te e\nabc\ta b c\n\u7121\u95a2\u4fc2\tm u\n",
        encoding="utf-8",
    )
    return root


def test_write_pruned_dictionary_keeps_only_lab_substrings_including_nfkc_lowercase(tmp_path: Path) -> None:
    root = _mfa_root_with_dictionary(tmp_path)
    full = root / "pretrained_models" / "dictionary" / "japanese_mfa.dict"

    info = write_pruned_mfa_dictionary(full, ["\u306f\u3044\u3001\uff21\uff22\uff23"], tmp_path / "pruned.dict")

    lines = (tmp_path / "pruned.dict").read_text(encoding="utf-8").splitlines()
    assert [line.split("\t")[0] for line in lines] == ["\u306f\u3044", "abc"]
    assert info["kept_entries"] == 2 and info["total_entries"] == 5


def test_prefetch_uses_pruned_dictionary_when_full_dictionary_is_available(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv(PRUNE_DICTIONARY_ENV, raising=False)
    paths = WorkPaths.from_workdir(tmp_path / "work")
    paths.audio_path.parent.mkdir(parents=True, exist_ok=True)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []
    environment = {"available": True, "command": ["mfa"], "root_dir": str(_mfa_root_with_dictionary(tmp_path))}

    summary = prefetch_local_mfa_alignments(
        paths, _candidates(), environment=environment, padding_ms=100, run_command=_fake_mfa(calls),
        environ_factory=lambda: {},
    )

    mfa_calls = [call for call in calls if call[0] == "mfa"]
    assert len(mfa_calls) == 1
    dictionary_arg = mfa_calls[0][mfa_calls[0].index("align") + 2]
    assert dictionary_arg.endswith(PRUNED_DICTIONARY_FILE)
    assert mfa_calls[0][mfa_calls[0].index("align") + 3] == "japanese_mfa"
    kept = [line.split("\t")[0] for line in Path(dictionary_arg).read_text(encoding="utf-8").splitlines()]
    assert kept == ["\u306f\u3044", "\u3046\u3093"]
    assert summary["dictionary"]["mode"] == "pruned" and summary["dictionary"]["fallback"] is False
    assert summary["aligned_count"] == 2


def test_failed_pruned_run_retries_once_with_full_dictionary(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv(PRUNE_DICTIONARY_ENV, raising=False)
    paths = WorkPaths.from_workdir(tmp_path / "work")
    paths.audio_path.parent.mkdir(parents=True, exist_ok=True)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []
    good = _fake_mfa(calls)
    bad = _fake_mfa(calls, returncode=1)

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        if command_list[0] == "mfa" and command_list[command_list.index("align") + 2].endswith(".dict"):
            return bad(command, **kwargs)
        return good(command, **kwargs)

    environment = {"available": True, "command": ["mfa"], "root_dir": str(_mfa_root_with_dictionary(tmp_path))}
    result = run_local_mfa_alignment_experiments(
        paths, [_candidates()[0]], environment=environment, max_run_candidates=1, padding_ms=100,
        run_command=fake_run, environ_factory=lambda: {},
    )[0]

    mfa_calls = [call for call in calls if call[0] == "mfa"]
    assert [call[call.index("align") + 2].endswith(".dict") for call in mfa_calls] == [True, False]
    assert mfa_calls[1][mfa_calls[1].index("align") + 2] == "japanese_mfa"
    assert result["status"] == "completed"
    assert result["dictionary"]["fallback"] is True and result["dictionary"]["pruned_returncode"] == 1


def test_pruning_can_be_disabled_by_environment_variable(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv(PRUNE_DICTIONARY_ENV, "0")
    paths = WorkPaths.from_workdir(tmp_path / "work")
    paths.audio_path.parent.mkdir(parents=True, exist_ok=True)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []
    environment = {"available": True, "command": ["mfa"], "root_dir": str(_mfa_root_with_dictionary(tmp_path))}

    summary = prefetch_local_mfa_alignments(
        paths, _candidates(), environment=environment, padding_ms=100, run_command=_fake_mfa(calls),
        environ_factory=lambda: {},
    )

    mfa_calls = [call for call in calls if call[0] == "mfa"]
    assert mfa_calls[0][mfa_calls[0].index("align") + 2] == "japanese_mfa"
    assert summary["dictionary"] == {"mode": "full", "fallback": False}
    assert prune_dictionary_enabled({"prune_dictionary": True}) is True
