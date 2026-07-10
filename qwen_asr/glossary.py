from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape
from xml.etree import ElementTree

NS_MAIN = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SHEET_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

SOURCE_HEADER_KEYWORDS = (
    "日文",
    "通用日语",
    "广播用",
    "劇中成员名（日文）",
    "剧中成员名（日文）",
)
TARGET_HEADER_KEYWORDS = ("中文",)
NOTE_HEADER_KEYWORDS = ("注释", "备注", "角色")
MAX_LOOKAHEAD_COLUMNS = 6
MAX_PROMPT_ENTRIES = 300


@dataclass(frozen=True)
class GlossaryEntry:
    group: str
    source: str
    target: str
    note: str = ""


@dataclass(frozen=True)
class NormalizedGlossaryResult:
    output_path: Path
    entry_count: int


def write_canonical_glossary_xlsx(
    entries: list[GlossaryEntry],
    output_path: str | Path,
) -> NormalizedGlossaryResult:
    target_path = Path(output_path)
    if target_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Canonical glossary output must be .xlsx: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [["group", "source", "target", "note"]]
    seen: set[tuple[str, str, str, str]] = set()
    for entry in entries:
        normalized = GlossaryEntry(
            group=_clean_cell(entry.group),
            source=_clean_cell(entry.source),
            target=_clean_cell(entry.target),
            note=_clean_cell(entry.note),
        )
        if not normalized.source or not normalized.target:
            continue
        key = (normalized.group, normalized.source, normalized.target, normalized.note)
        if key in seen:
            continue
        seen.add(key)
        rows.append([normalized.group, normalized.source, normalized.target, normalized.note])

    _write_single_sheet_xlsx(target_path, "Glossary", rows)
    return NormalizedGlossaryResult(output_path=target_path, entry_count=len(rows) - 1)


def build_glossary_prompt(xlsx_path: str | Path) -> str:
    """Read a non-normalized xlsx glossary and render prompt text."""
    entries = read_xlsx_glossary(xlsx_path)
    if not entries:
        return ""

    lines = [
        "Use the following translation glossary when applicable.",
        "Prefer these terms for names, titles, fixed phrases, and role labels.",
        "If an entry contains notes, use them as contextual guidance.",
    ]
    current_group = None
    for entry in entries[:MAX_PROMPT_ENTRIES]:
        if entry.group != current_group:
            current_group = entry.group
            lines.append(f"\n[{current_group}]")
        line = f"- {entry.source} => {entry.target}"
        if entry.note:
            line += f"; 说明：{entry.note}"
        lines.append(line)

    if len(entries) > MAX_PROMPT_ENTRIES:
        lines.append(f"\nOnly the first {MAX_PROMPT_ENTRIES} glossary entries are shown.")

    return "\n".join(lines).strip()


def read_xlsx_glossary(xlsx_path: str | Path) -> list[GlossaryEntry]:
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Glossary xlsx not found: {path}")
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Glossary file must be .xlsx: {path}")

    rows_by_sheet = _read_xlsx_rows(path)
    entries: list[GlossaryEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for sheet_name, rows in rows_by_sheet:
        if not rows:
            continue
        headers = [_clean_cell(value) for value in rows[0]]
        canonical_columns = _detect_canonical_columns(headers)
        if canonical_columns is not None:
            group_col, source_col, target_col, note_col = canonical_columns
            for row in rows[1:]:
                source = _get_cell(row, source_col)
                target = _get_cell(row, target_col)
                if not source or not target:
                    continue
                group = _get_cell(row, group_col) or sheet_name
                note = _get_cell(row, note_col) if note_col is not None else ""
                key = (group, source, target)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(GlossaryEntry(group=group, source=source, target=target, note=note))
            continue
        for source_col, target_col in _detect_column_pairs(headers):
            note_cols = _detect_note_columns(headers, source_col, target_col)
            group = headers[source_col] or sheet_name
            for row in rows[1:]:
                source = _get_cell(row, source_col)
                target = _get_cell(row, target_col)
                if not source or not target:
                    continue
                note = "；".join(
                    value
                    for value in (_get_cell(row, col) for col in note_cols)
                    if value and value not in {source, target}
                )
                key = (group, source, target)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(
                    GlossaryEntry(
                        group=group,
                        source=source,
                        target=target,
                        note=note,
                    )
                )
    return entries


def write_normalized_glossary_xlsx(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> NormalizedGlossaryResult:
    source_path = Path(input_path)
    entries = read_xlsx_glossary(source_path)
    if not entries:
        raise ValueError(f"No usable glossary entries found: {source_path}")

    target_path = Path(output_path) if output_path is not None else _default_normalized_xlsx_path(source_path)
    if target_path.resolve() == source_path.resolve():
        raise ValueError("Normalized glossary output must not overwrite the source xlsx.")
    if target_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Normalized glossary output must be .xlsx: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [["group", "source", "target", "note"]]
    rows.extend(
        [
            _clean_cell(entry.group),
            _clean_cell(entry.source),
            _clean_cell(entry.target),
            _clean_cell(entry.note),
        ]
        for entry in entries
    )
    _write_single_sheet_xlsx(target_path, "Glossary", rows)
    return NormalizedGlossaryResult(output_path=target_path, entry_count=len(entries))


def _read_xlsx_rows(path: Path) -> list[tuple[str, list[list[str]]]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _load_shared_strings(archive)
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = _load_workbook_relationships(archive)
        result: list[tuple[str, list[list[str]]]] = []
        for sheet in workbook.findall("m:sheets/m:sheet", NS_MAIN):
            name = sheet.attrib.get("name", "Sheet")
            rel_id = sheet.attrib.get(f"{{{SHEET_REL_NS}}}id")
            target = rels.get(rel_id or "")
            if not target:
                continue
            target = target.lstrip("/")
            sheet_path = target if target.startswith("xl/") else "xl/" + target
            rows = _load_sheet_rows(archive, sheet_path, shared_strings)
            result.append((name, rows))
        return result


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for item in root.findall("m:si", NS_MAIN):
        parts = [node.text or "" for node in item.findall(".//m:t", NS_MAIN)]
        strings.append("".join(parts))
    return strings


def _load_workbook_relationships(archive: zipfile.ZipFile) -> dict[str, str]:
    root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rels: dict[str, str] = {}
    for rel in root.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            rels[rel_id] = target
    return rels


def _load_sheet_rows(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[list[str]]:
    root = ElementTree.fromstring(archive.read(sheet_path))
    rows: list[list[str]] = []
    for row in root.findall("m:sheetData/m:row", NS_MAIN):
        values: dict[int, str] = {}
        max_col = -1
        for cell in row.findall("m:c", NS_MAIN):
            col_index = _cell_ref_to_col_index(cell.attrib.get("r", ""))
            if col_index is None:
                continue
            max_col = max(max_col, col_index)
            values[col_index] = _read_cell(cell, shared_strings)
        if max_col < 0:
            rows.append([])
            continue
        rows.append([values.get(index, "") for index in range(max_col + 1)])
    return rows


def _read_cell(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return _clean_cell("".join(node.text or "" for node in cell.findall(".//m:t", NS_MAIN)))

    value_node = cell.find("m:v", NS_MAIN)
    raw_value = value_node.text if value_node is not None else ""
    if cell_type == "s" and raw_value:
        try:
            return _clean_cell(shared_strings[int(raw_value)])
        except (IndexError, ValueError):
            return ""
    return _clean_cell(raw_value)


def _detect_column_pairs(headers: list[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for source_col, header in enumerate(headers):
        if not _is_source_header(header):
            continue
        for target_col in range(source_col + 1, min(len(headers), source_col + MAX_LOOKAHEAD_COLUMNS + 1)):
            if _is_target_header(headers[target_col]):
                pairs.append((source_col, target_col))
                break
    return pairs


def _detect_canonical_columns(headers: list[str]) -> tuple[int, int, int, int | None] | None:
    normalized = [_clean_cell(header).lower() for header in headers]
    try:
        source_col = normalized.index("source")
        target_col = normalized.index("target")
    except ValueError:
        return None
    group_col = normalized.index("group") if "group" in normalized else source_col
    note_col = normalized.index("note") if "note" in normalized else None
    return group_col, source_col, target_col, note_col


def _detect_note_columns(headers: list[str], source_col: int, target_col: int) -> list[int]:
    note_cols: list[int] = []
    end = min(len(headers), target_col + MAX_LOOKAHEAD_COLUMNS + 1)
    for col in range(source_col + 1, end):
        if col == target_col:
            continue
        if any(keyword in headers[col] for keyword in NOTE_HEADER_KEYWORDS):
            note_cols.append(col)
    return note_cols


def _is_source_header(header: str) -> bool:
    return bool(header) and any(keyword in header for keyword in SOURCE_HEADER_KEYWORDS)


def _is_target_header(header: str) -> bool:
    return bool(header) and any(keyword in header for keyword in TARGET_HEADER_KEYWORDS)


def _get_cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return _clean_cell(row[index])


def _clean_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _cell_ref_to_col_index(cell_ref: str) -> int | None:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return None
    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _default_normalized_xlsx_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.normalized{path.suffix}")


def _write_single_sheet_xlsx(path: Path, sheet_name: str, rows: list[list[str]]) -> None:
    sheet_xml = _build_sheet_xml(rows)
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _build_sheet_xml(rows: list[list[str]]) -> str:
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            cell_ref = f"{_column_name(col_index)}{row_index}"
            text = escape(_clean_cell(value))
            cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        "</worksheet>"
    )


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


__all__ = [
    "GlossaryEntry",
    "NormalizedGlossaryResult",
    "build_glossary_prompt",
    "read_xlsx_glossary",
    "write_canonical_glossary_xlsx",
    "write_normalized_glossary_xlsx",
]
