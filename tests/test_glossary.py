from __future__ import annotations

from pathlib import Path

from qwen_asr.glossary import (
    _read_xlsx_rows,
    _write_single_sheet_xlsx,
    read_xlsx_glossary,
    write_normalized_glossary_xlsx,
)


def _write_input_xlsx(path: Path) -> None:
    _write_single_sheet_xlsx(
        path,
        "Input",
        [
            ["group", "source", "target", "note"],
            ["Names", " Alice  ", "  Alice CN ", " lead "],
            ["Names", "Alice", "Alice CN", "lead"],
            ["Terms", "Radio", "Broadcast", ""],
            ["Terms", "", "Skip", ""],
        ],
    )


def test_write_normalized_glossary_xlsx_creates_canonical_sheet(tmp_path: Path) -> None:
    source = tmp_path / "glossary.xlsx"
    _write_input_xlsx(source)

    result = write_normalized_glossary_xlsx(source)

    assert result.entry_count == 2
    assert result.output_path == tmp_path / "glossary.normalized.xlsx"
    assert result.output_path.exists()

    rows_by_sheet = _read_xlsx_rows(result.output_path)
    sheet_name, rows = rows_by_sheet[0]
    assert sheet_name == "Glossary"
    assert rows == [
        ["group", "source", "target", "note"],
        ["Names", "Alice", "Alice CN", "lead"],
        ["Terms", "Radio", "Broadcast", ""],
    ]


def test_normalized_glossary_xlsx_can_be_read_back(tmp_path: Path) -> None:
    source = tmp_path / "glossary.xlsx"
    _write_input_xlsx(source)

    result = write_normalized_glossary_xlsx(source)
    entries = read_xlsx_glossary(result.output_path)

    assert [(entry.group, entry.source, entry.target, entry.note) for entry in entries] == [
        ("Names", "Alice", "Alice CN", "lead"),
        ("Terms", "Radio", "Broadcast", ""),
    ]


def test_write_normalized_glossary_rejects_empty_input(tmp_path: Path) -> None:
    source = tmp_path / "empty.xlsx"
    _write_single_sheet_xlsx(source, "Input", [["group", "source", "target", "note"]])

    try:
        write_normalized_glossary_xlsx(source)
    except ValueError as exc:
        assert "No usable glossary entries" in str(exc)
    else:
        raise AssertionError("empty glossary should be rejected")
    assert not (tmp_path / "empty.normalized.xlsx").exists()
