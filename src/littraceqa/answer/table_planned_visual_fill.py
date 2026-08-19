"""Conservative composition of ordinary and visual-consensus planned tables.

The ordinary one-read planned answer is the immutable floor.  A second pass
reuses the exact same :class:`TablePlan` and selected-paper context with three
visual reads, but may contribute only type-correct values for null/missing
schema cells in the floor's exact uniquely keyed rows.  It cannot add, delete,
replace, reorder, or canonicalize a row, and it never consumes replay rows.

When the ordinary pass is completely empty, the visual pass may rescue the
table only under a stricter complete-grid contract: one selected paper, an
explicit nonempty planned row set, exact coverage of that set with no extra
rows or fields, every value cell non-null and type-correct, and every row key
grounded on evidence from that selected paper.  This lets a source image
recover a failed first read without turning the visual model into an
open-ended row generator.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping
import unicodedata

from littraceqa.answer.interfaces import (
    AnswerContext,
    StrategyOutput,
    register_strategy,
)
from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_plan import TablePlan
from littraceqa.answer.table_planned import SchemaPlannedTableAnswerer


def _normalized_key(
    row: Mapping[str, Any], row_key_cols: Iterable[str]
) -> tuple[str, ...] | None:
    columns = tuple(row_key_cols)
    if not columns or any(column not in row for column in columns):
        return None
    key = tuple(normalize_text(row[column]) for column in columns)
    return key if all(key) else None


def _unique_rows_by_key(
    rows: list[dict], row_key_cols: Iterable[str]
) -> dict[tuple[str, ...], dict] | None:
    indexed: dict[tuple[str, ...], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        key = _normalized_key(row, row_key_cols)
        if key is None or key in indexed:
            return None
        indexed[key] = row
    return indexed


def _type_compatible(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str) and bool(value.strip())
    if kind == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if kind == "boolean":
        return isinstance(value, bool)
    return False


def fill_visual_consensus_nulls(
    base_rows: list[dict],
    candidate_rows: list[dict],
    plan: TablePlan,
) -> tuple[list[dict], list[dict[str, Any]]] | None:
    """Return a fill-only merge, or ``None`` on any structural conflict."""

    if not base_rows or len(candidate_rows) != len(base_rows):
        return None
    row_keys = tuple(plan.row_key_cols)
    value_types: dict[str, str] = {}
    for column in plan.value_cols:
        name = column.get("name")
        kind = column.get("type")
        if (
            not isinstance(name, str)
            or not name
            or name in value_types
            or name in row_keys
            or kind not in {"string", "number", "boolean"}
        ):
            return None
        value_types[name] = kind
    if not row_keys or len(row_keys) != len(set(row_keys)) or not value_types:
        return None

    base_by_key = _unique_rows_by_key(base_rows, row_keys)
    candidate_by_key = _unique_rows_by_key(candidate_rows, row_keys)
    if (
        base_by_key is None
        or candidate_by_key is None
        or set(base_by_key) != set(candidate_by_key)
    ):
        return None

    output = [dict(row) for row in base_rows]
    fills: list[dict[str, Any]] = []
    for row_index, (base_row, output_row) in enumerate(
        zip(base_rows, output, strict=True)
    ):
        key = _normalized_key(base_row, row_keys)
        if key is None:
            return None
        candidate = candidate_by_key[key]
        for column, kind in value_types.items():
            base_value = base_row.get(column)
            candidate_present = column in candidate
            candidate_value = candidate.get(column)
            if base_value is not None:
                if (
                    not candidate_present
                    or type(candidate_value) is not type(base_value)
                    or candidate_value != base_value
                ):
                    return None
                continue
            if not candidate_present or candidate_value is None:
                continue
            if not _type_compatible(candidate_value, kind):
                return None
            output_row[column] = candidate_value
            fills.append({"row_index": row_index, "column": column})

    return (output, fills) if fills else None


_MAX_TABLE_ROW_SEGMENT_CHARS = 700
_TABLE_HEADER_WINDOW_CHARS = 700


def _source_normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    # Join only an actual word broken across a line.  A standalone ``-`` is a
    # meaningful missing-value cell in scientific tables and must survive.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = text.replace("­", "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _surface_spans(text: str, value: Any, *, start: int = 0) -> list[tuple[int, int]]:
    surface = _source_normalize(value)
    if not surface:
        return []
    return [
        (match.start(), match.end())
        for match in re.finditer(
            rf"(?<!\w){re.escape(surface)}(?!\w)", text[start:]
        )
    ] if start == 0 else [
        (start + match.start(), start + match.end())
        for match in re.finditer(
            rf"(?<!\w){re.escape(surface)}(?!\w)", text[start:]
        )
    ]


def _row_local_source_support(
    *,
    page_text: str,
    expected_rows: list[dict],
    candidate_by_key: dict[tuple[str, ...], dict],
    row_keys: tuple[str, ...],
    value_types: Mapping[str, str],
) -> bool:
    """Verify each proposed cell inside its own bounded printed row segment."""

    text = _source_normalize(page_text)
    if not text or not expected_rows:
        return False
    first_key = row_keys[0]
    sequences: list[list[tuple[int, int]]] = []
    for first_span in _surface_spans(text, expected_rows[0][first_key]):
        spans = [first_span]
        cursor = first_span[1]
        for expected in expected_rows[1:]:
            matches = _surface_spans(text, expected[first_key], start=cursor)
            if not matches:
                break
            match = matches[0]
            if match[0] - cursor > _MAX_TABLE_ROW_SEGMENT_CHARS:
                break
            spans.append(match)
            cursor = match[1]
        if len(spans) != len(expected_rows):
            continue
        header = text[
            max(0, first_span[0] - _TABLE_HEADER_WINDOW_CHARS):first_span[0]
        ]
        if not all(
            _surface_spans(header, column) for column in value_types
        ):
            continue
        sequences.append(spans)
    # A repeated/ambiguous table is not a safe source for coordinate binding.
    if len(sequences) != 1:
        return False

    spans = sequences[0]
    for index, expected in enumerate(expected_rows):
        row_start = spans[index][0]
        row_end = (
            spans[index + 1][0]
            if index + 1 < len(spans)
            else min(len(text), row_start + _MAX_TABLE_ROW_SEGMENT_CHARS)
        )
        segment = text[row_start:row_end]
        component_cursor = 0
        for column in row_keys:
            component_spans = _surface_spans(
                segment, expected[column], start=component_cursor
            )
            if not component_spans:
                return False
            component_cursor = component_spans[0][1]
        value_segment = segment[component_cursor:]
        key = _normalized_key(expected, row_keys)
        if key is None:
            return False
        source = candidate_by_key[key]
        for column in value_types:
            value = source[column]
            surfaces = [value]
            if (
                isinstance(value, float)
                and math.isfinite(value)
                and value.is_integer()
            ):
                surfaces.append(int(value))
            if not any(_surface_spans(value_segment, item) for item in surfaces):
                return False
    return True


def _word_phrase_matches(
    words: list[tuple], value: Any
) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    tokens = _source_normalize(value).split()
    if not tokens:
        return []
    matches: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    for start in range(0, len(words) - len(tokens) + 1):
        indices = tuple(range(start, start + len(tokens)))
        phrase = words[start:start + len(tokens)]
        line = (int(phrase[0][5]), int(phrase[0][6]))
        if any((int(word[5]), int(word[6])) != line for word in phrase):
            continue
        if [_source_normalize(word[4]) for word in phrase] == tokens:
            matches.append((line, indices))
    return matches


def _coordinate_layout(
    *,
    words: list[tuple],
    expected_rows: list[dict],
    row_keys: tuple[str, ...],
    value_column: str,
) -> tuple[float, float, list[float], list[frozenset[int]]] | None:
    """Resolve one requested PDF column and the y-coordinate of every row."""

    header_matches = _word_phrase_matches(words, value_column)
    if not header_matches:
        return None
    viable: list[
        tuple[
            tuple[int, int], tuple[int, ...], list[float],
            list[frozenset[int]],
        ]
    ] = []
    for header_line, header_indices in header_matches:
        header_y = sum(float(words[index][1]) for index in header_indices) / len(
            header_indices
        )
        row_ys: list[float] = []
        row_key_indices: list[frozenset[int]] = []
        for expected in expected_rows:
            component_matches: list[list[tuple[float, tuple[int, ...]]]] = []
            for column in row_keys:
                component_matches.append([
                    (
                        sum(
                            (float(words[index][1]) + float(words[index][3])) / 2.0
                            for index in indices
                        ) / len(indices),
                        indices,
                    )
                    for _line, indices in _word_phrase_matches(words, expected[column])
                ])
            candidates: list[tuple[float, frozenset[int]]] = []
            for y, first_indices in (
                component_matches[0] if component_matches else []
            ):
                matched_indices = set(first_indices)
                ambiguous = False
                for matches in component_matches[1:]:
                    aligned = [
                        indices for other_y, indices in matches
                        if abs(other_y - y) <= 2.0
                    ]
                    if len(aligned) != 1:
                        ambiguous = True
                        break
                    matched_indices.update(aligned[0])
                if ambiguous:
                    continue
                if header_y < y <= header_y + 140.0:
                    candidates.append((round(y, 3), frozenset(matched_indices)))
            candidates = list(dict.fromkeys(candidates))
            if (
                len(candidates) != 1
                or any(abs(candidates[0][0] - prior) <= 2.0 for prior in row_ys)
            ):
                break
            row_ys.append(candidates[0][0])
            row_key_indices.append(candidates[0][1])
        if len(row_ys) == len(expected_rows):
            viable.append((
                header_line, header_indices, row_ys, row_key_indices
            ))
    if len(viable) != 1:
        return None

    header_line, header_indices, row_ys, row_key_indices = viable[0]
    header_x = sum(
        (float(words[index][0]) + float(words[index][2])) / 2.0
        for index in header_indices
    ) / len(header_indices)
    header_centers = [
        (float(word[0]) + float(word[2])) / 2.0
        for index, word in enumerate(words)
        if (int(word[5]), int(word[6])) == header_line
        and index not in set(header_indices)
    ]
    nearest_header = min(
        (abs(center - header_x) for center in header_centers),
        default=40.0,
    )
    tolerance = max(3.0, min(15.0, nearest_header * 0.4))
    return header_x, tolerance, row_ys, row_key_indices


def _coordinate_column_support(
    *,
    words: list[tuple],
    expected_rows: list[dict],
    candidate_by_key: dict[tuple[str, ...], dict],
    row_keys: tuple[str, ...],
    value_column: str,
) -> bool:
    """Bind a one-column answer to the physical PDF header/row coordinates."""

    layout = _coordinate_layout(
        words=words,
        expected_rows=expected_rows,
        row_keys=row_keys,
        value_column=value_column,
    )
    if layout is None:
        return False
    header_x, tolerance, row_ys, _row_key_indices = layout

    for expected, row_y in zip(expected_rows, row_ys, strict=True):
        key = _normalized_key(expected, row_keys)
        if key is None:
            return False
        value = candidate_by_key[key][value_column]
        surfaces = [value]
        if (
            isinstance(value, float)
            and math.isfinite(value)
            and value.is_integer()
        ):
            surfaces.append(int(value))
        value_matches = [
            indices
            for surface in surfaces
            for _match_line, indices in _word_phrase_matches(words, surface)
            if abs(
                sum(
                    (float(words[index][1]) + float(words[index][3])) / 2.0
                    for index in indices
                ) / len(indices) - row_y
            ) <= 2.0
        ]
        if not value_matches:
            return False
        aligned = []
        for indices in value_matches:
            center = sum(
                (float(words[index][0]) + float(words[index][2])) / 2.0
                for index in indices
            ) / len(indices)
            if abs(center - header_x) <= tolerance:
                aligned.append(indices)
        if len(aligned) != 1:
            return False
    return True


def _coordinate_column_values(
    *,
    words: list[tuple],
    expected_rows: list[dict],
    row_keys: tuple[str, ...],
    value_column: str,
) -> list[str] | None:
    """Read one unambiguous physical cell at each row/header intersection."""

    layout = _coordinate_layout(
        words=words,
        expected_rows=expected_rows,
        row_keys=row_keys,
        value_column=value_column,
    )
    if layout is None:
        return None
    header_x, tolerance, row_ys, row_key_indices = layout
    values: list[str] = []
    for row_y, excluded_indices in zip(
        row_ys, row_key_indices, strict=True
    ):
        aligned = [
            word
            for index, word in enumerate(words)
            if index not in excluded_indices
            if abs(
                (float(word[1]) + float(word[3])) / 2.0 - row_y
            ) <= 2.0
            and abs(
                (float(word[0]) + float(word[2])) / 2.0 - header_x
            ) <= tolerance
        ]
        # A single PDF word is intentionally required. Multi-token cells need
        # a separate span/column-width parser and abstain here.
        if len(aligned) != 1 or not str(aligned[0][4]).strip():
            return None
        values.append(str(aligned[0][4]).strip())
    return values


def _pdf_column_coordinate_support(
    *,
    pdf_bytes: bytes,
    page_number: int,
    expected_rows: list[dict],
    candidate_by_key: dict[tuple[str, ...], dict],
    row_keys: tuple[str, ...],
    value_column: str,
) -> bool:
    if not pdf_bytes or page_number < 1:
        return False
    try:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if page_number > len(document):
                return False
            words = document[page_number - 1].get_text("words")
    except Exception:  # noqa: BLE001 -- coordinate proof is fail-closed
        return False
    return _coordinate_column_support(
        words=list(words),
        expected_rows=expected_rows,
        candidate_by_key=candidate_by_key,
        row_keys=row_keys,
        value_column=value_column,
    )


def _pdf_coordinate_column_values(
    *,
    pdf_bytes: bytes,
    page_number: int,
    expected_rows: list[dict],
    row_keys: tuple[str, ...],
    value_column: str,
) -> list[str] | None:
    if not pdf_bytes or page_number < 1:
        return None
    try:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if page_number > len(document):
                return None
            words = document[page_number - 1].get_text("words")
    except Exception:  # noqa: BLE001 -- coordinate extraction is fail-closed
        return None
    return _coordinate_column_values(
        words=list(words),
        expected_rows=expected_rows,
        row_keys=row_keys,
        value_column=value_column,
    )


def _typed_coordinate_value(raw: str, kind: str) -> Any | None:
    text = raw.strip()
    if not text:
        return None
    if kind == "string":
        return text
    if kind == "boolean":
        normalized = text.casefold()
        if normalized in {"true", "yes"}:
            return True
        if normalized in {"false", "no"}:
            return False
        return None
    if kind == "number":
        cleaned = text.replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1].strip()
        try:
            value = float(cleaned)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        return int(value) if value.is_integer() else value
    return None


def rescue_coordinate_table(
    plan: TablePlan,
    *,
    source_ctx: AnswerContext,
) -> tuple[list[dict], list[Any]] | None:
    """Deterministically read a complete one-column table from PDF geometry."""

    if len(source_ctx.paper_ids) != 1 or not plan.expected_keys:
        return None
    row_keys = tuple(plan.row_key_cols)
    expected_rows = [
        dict(zip(row_keys, values, strict=True))
        for values in plan.expected_keys
        if len(values) == len(row_keys)
    ]
    if (
        len(expected_rows) != len(plan.expected_keys)
        or len(row_keys) != 1
        or _unique_rows_by_key(expected_rows, row_keys) is None
        or len(plan.value_cols) != 1
    ):
        return None
    column = plan.value_cols[0]
    value_column = column.get("name")
    value_kind = column.get("type")
    if (
        not isinstance(value_column, str)
        or not value_column
        or value_column in row_keys
        or value_kind not in {"string", "number", "boolean"}
    ):
        return None

    selected_paper = source_ctx.paper_ids[0]
    table_evidence = [
        item for item in source_ctx.evidence
        if item.paper_id == selected_paper and item.source_type == "table"
    ]
    locator_keys = {
        (item.paper_id, item.source_type, item.page, item.object_id)
        for item in table_evidence
    }
    if len(locator_keys) != 1:
        return None
    selected_locator = next(iter(locator_keys))
    raw_values = _pdf_coordinate_column_values(
        pdf_bytes=source_ctx.pdf_bytes_by_id.get(selected_paper, b""),
        page_number=int(selected_locator[2]),
        expected_rows=expected_rows,
        row_keys=row_keys,
        value_column=value_column,
    )
    if raw_values is None or len(raw_values) != len(expected_rows):
        return None

    rows: list[dict] = []
    for expected, raw in zip(expected_rows, raw_values, strict=True):
        value = _typed_coordinate_value(raw, value_kind)
        if value is None or not _type_compatible(value, value_kind):
            return None
        if not any(
            item.quote
            and _surface_spans(_source_normalize(item.quote), raw)
            for item in table_evidence
        ):
            return None
        rows.append({**expected, value_column: value})
    evidence = [
        next(
            item for item in table_evidence
            if (
                item.paper_id, item.source_type, item.page, item.object_id
            ) == selected_locator
        )
    ]
    return rows, evidence


def rescue_complete_visual_table(
    candidate: StrategyOutput,
    plan: TablePlan,
    *,
    source_ctx: AnswerContext,
) -> tuple[list[dict], list[Any]] | None:
    """Admit a complete source-grounded visual table after an empty base.

    The returned row keys use the plan's exact bytes and order; candidate row
    keys are used only for normalized one-to-one matching.  This avoids making
    a vision model authoritative for scorer-facing row identity.
    """

    if len(source_ctx.paper_ids) != 1 or not plan.expected_keys:
        return None
    row_keys = tuple(plan.row_key_cols)
    expected_rows = [
        dict(zip(row_keys, values, strict=True))
        for values in plan.expected_keys
        if len(values) == len(row_keys)
    ]
    if len(expected_rows) != len(plan.expected_keys):
        return None
    expected_by_key = _unique_rows_by_key(expected_rows, row_keys)
    if expected_by_key is None:
        return None

    value_types: dict[str, str] = {}
    for column in plan.value_cols:
        name = column.get("name")
        kind = column.get("type")
        if (
            not isinstance(name, str)
            or not name
            or name in value_types
            or name in row_keys
            or kind not in {"string", "number", "boolean"}
        ):
            return None
        value_types[name] = kind
    # Empty-base rescue is deliberately narrower than ordinary null filling.
    # With one requested value column, row-local binding plus the localized
    # scalar quote is sufficient; multiple requested columns require a full
    # coordinate-aware grid parser and therefore abstain here.
    if (
        not row_keys
        or len(row_keys) != len(set(row_keys))
        or len(value_types) != 1
    ):
        return None

    rows = candidate.value
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        return None
    candidate_by_key = _unique_rows_by_key(rows, row_keys)
    if candidate_by_key is None or set(candidate_by_key) != set(expected_by_key):
        return None

    allowed_fields = set(row_keys) | set(value_types)
    if (
        not isinstance(candidate.confidence, (int, float))
        or isinstance(candidate.confidence, bool)
        or not math.isfinite(candidate.confidence)
        or candidate.confidence <= 0.0
    ):
        return None
    selected_paper = source_ctx.paper_ids[0]
    table_evidence = [
        item for item in source_ctx.evidence
        if item.paper_id == selected_paper and item.source_type == "table"
    ]
    if not table_evidence:
        return None

    def locator_key(item: Any) -> tuple[Any, ...]:
        return (item.paper_id, item.source_type, item.page, item.object_id)

    locator_keys = {locator_key(item) for item in table_evidence}
    if len(locator_keys) != 1:
        return None
    selected_locator = next(iter(locator_keys))
    parsed = source_ctx.parsed_by_id.get(selected_paper)
    page = parsed.page(selected_locator[2]) if parsed else None
    if page is None or not page.text:
        return None

    diagnostics = candidate.diagnostics
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("visual_consensus_repeats") != 3
        or diagnostics.get("extraction_sources") != "visual_only"
        or diagnostics.get("per_paper_rows") != {
            selected_paper: len(expected_rows)
        }
    ):
        return None

    output: list[dict] = []
    for expected in expected_rows:
        key = _normalized_key(expected, row_keys)
        if key is None:
            return None
        source = candidate_by_key[key]
        if set(source) != allowed_fields:
            return None
        row = dict(expected)
        for column, kind in value_types.items():
            value = source.get(column)
            if value is None or not _type_compatible(value, kind):
                return None
            surfaces = [value]
            if (
                isinstance(value, float)
                and math.isfinite(value)
                and value.is_integer()
            ):
                surfaces.append(int(value))
            # The immutable question-conditioned localizer must also have
            # selected the scalar at this table locator.  Row-local page
            # binding below assigns it to the row; quote membership prevents
            # a visually scrambled adjacent column from being accepted merely
            # because its value occurs on the same printed row.
            if not any(
                item.quote
                and any(
                    _surface_spans(_source_normalize(item.quote), surface)
                    for surface in surfaces
                )
                for item in table_evidence
            ):
                return None
            row[column] = value
        output.append(row)
    if not _row_local_source_support(
        page_text=page.text,
        expected_rows=expected_rows,
        candidate_by_key=candidate_by_key,
        row_keys=row_keys,
        value_types=value_types,
    ):
        return None
    value_column = next(iter(value_types))
    if not _pdf_column_coordinate_support(
        pdf_bytes=source_ctx.pdf_bytes_by_id.get(selected_paper, b""),
        page_number=int(selected_locator[2]),
        expected_rows=expected_rows,
        candidate_by_key=candidate_by_key,
        row_keys=row_keys,
        value_column=value_column,
    ):
        return None
    rescue_evidence = [
        next(item for item in table_evidence if locator_key(item) == selected_locator)
    ]
    return output, rescue_evidence


@register_strategy("table_planned_visual_fill")
class PlannedVisualConsensusFillTableAnswerer:
    """Preserve the ordinary planned table and fill only consensus nulls."""

    answer_type = "table"
    requires_documents = True

    def __init__(
        self,
        *,
        retain_concise_missing_expected_rows: bool = False,
        route_expected_rows_to_papers: bool = False,
        open_ended_one_row_per_paper: bool = False,
        prefer_owned_cells: bool = False,
        retain_source_attested_expected_rows: bool = False,
        visual_extraction_mode: str = "direct",
        visual_retry_owned_rows: bool = False,
        extraction_sources: str = "both",
        text_context_mode: str = "evidence",
        text_page_k: int = 3,
        fill_nulls_from_scalar_evidence: bool = False,
        trace_reassembly_inputs: bool = False,
    ):
        boolean_options = {
            "retain_concise_missing_expected_rows": retain_concise_missing_expected_rows,
            "route_expected_rows_to_papers": route_expected_rows_to_papers,
            "open_ended_one_row_per_paper": open_ended_one_row_per_paper,
            "prefer_owned_cells": prefer_owned_cells,
            "retain_source_attested_expected_rows": retain_source_attested_expected_rows,
            "visual_retry_owned_rows": visual_retry_owned_rows,
            "fill_nulls_from_scalar_evidence": fill_nulls_from_scalar_evidence,
            "trace_reassembly_inputs": trace_reassembly_inputs,
        }
        invalid = [name for name, value in boolean_options.items() if type(value) is not bool]
        if invalid:
            raise ValueError(f"{invalid[0]} must be a boolean")
        common = {
            **boolean_options,
            "visual_extraction_mode": visual_extraction_mode,
            "extraction_sources": extraction_sources,
            "text_context_mode": text_context_mode,
            "text_page_k": text_page_k,
        }
        self._base = SchemaPlannedTableAnswerer(
            **common, visual_consensus_repeats=1
        )
        consensus = {
            **common,
            "extraction_sources": "visual_only",
            "visual_retry_owned_rows": False,
            "fill_nulls_from_scalar_evidence": False,
            "open_ended_one_row_per_paper": False,
        }
        self._consensus = SchemaPlannedTableAnswerer(
            **consensus, visual_consensus_repeats=3
        )

    @staticmethod
    def _isolated_context(ctx: AnswerContext) -> AnswerContext:
        """Clone mutable containers without changing their source contents."""

        return replace(
            ctx,
            paper_ids=list(ctx.paper_ids),
            evidence=deepcopy(ctx.evidence),
            parsed_by_id=deepcopy(ctx.parsed_by_id),
            paper_titles=dict(ctx.paper_titles),
            table_schema=deepcopy(ctx.table_schema),
            pdf_bytes_by_id=dict(ctx.pdf_bytes_by_id),
            frozen_table_rows=None,
        )

    @staticmethod
    def _source_scope(ctx: AnswerContext) -> tuple:
        """Digest the complete ordered selected-paper/source boundary."""

        evidence = [
            {
                "paper_id": item.paper_id,
                "source_type": item.source_type,
                "page": item.page,
                "object_id": item.object_id,
                "quote": item.quote,
                "confidence": item.confidence,
            }
            for item in ctx.evidence
        ]
        parsed = [
            {
                "mapping_paper_id": mapping_paper_id,
                "document_paper_id": document.paper_id,
                "pages": [
                    {
                        "page": page.page,
                        "text": page.text,
                        "objects": [
                            {
                                "source_type": item.source_type,
                                "object_id": item.object_id,
                                "page": item.page,
                            }
                            for item in page.objects
                        ],
                    }
                    for page in document.pages
                ],
            }
            for mapping_paper_id, document in ctx.parsed_by_id.items()
        ]

        def digest(value: Any) -> str:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        return (
            tuple(ctx.paper_ids),
            tuple(ctx.paper_titles.items()),
            digest(evidence),
            digest(parsed),
            tuple(
                (paper_id, hashlib.sha256(pdf_bytes).hexdigest())
                for paper_id, pdf_bytes in ctx.pdf_bytes_by_id.items()
            ),
        )

    def answer(self, ctx: AnswerContext) -> StrategyOutput:
        # Snapshot two independent source contexts before the base call.  A
        # buggy first stage therefore cannot expand the candidate's paper
        # scope or mutate the caller's context.
        try:
            base_ctx = self._isolated_context(ctx)
            candidate_ctx = self._isolated_context(ctx)
            pristine_scope = self._source_scope(base_ctx)
            if self._source_scope(candidate_ctx) != pristine_scope:
                raise ValueError("isolated source scopes differ")
        except Exception:  # noqa: BLE001 -- no unisolated release fallback
            return StrategyOutput(
                value=[], confidence=0.0, attested_evidence=[],
                diagnostics={"status": "source_scope_isolation_failed"},
            )
        base, plan = self._base.answer_with_plan(base_ctx)
        if ctx.frozen_table_rows is not None or plan is None:
            return base
        if self._source_scope(base_ctx) != pristine_scope:
            return base
        if not isinstance(base.value, list) or not all(
            isinstance(row, dict) for row in base.value
        ):
            return base
        if _unique_rows_by_key(base.value, plan.row_key_cols) is None:
            return base
        value_columns = [
            column.get("name")
            for column in plan.value_cols
            if isinstance(column, Mapping)
            and isinstance(column.get("name"), str)
        ]
        rescue_empty_base = not base.value and bool(plan.expected_keys)
        if not value_columns or (
            not rescue_empty_base
            and not any(
                row.get(column) is None
                for row in base.value
                for column in value_columns
            )
        ):
            return base
        if rescue_empty_base:
            rescued = rescue_coordinate_table(plan, source_ctx=candidate_ctx)
            if rescued is None:
                return base
            rows, rescue_evidence = rescued
            return StrategyOutput(
                value=rows,
                confidence=1.0,
                attested_evidence=rescue_evidence,
                diagnostics={
                    "status": "coordinate_table_rescue",
                    "visual_rescued_rows": len(rows),
                    "row_key_cols": list(plan.row_key_cols),
                    "expected_keys": [list(key) for key in plan.expected_keys],
                    "coordinate_value_column": value_columns[0],
                },
            )
        pristine_base: StrategyOutput | None = None
        try:
            pristine_base = deepcopy(base)
            pristine_plan = deepcopy(plan)
            candidate = self._consensus.answer_from_plan(
                candidate_ctx, plan
            )
            if plan != pristine_plan:
                return pristine_base
            if base != pristine_base:
                return pristine_base
            if self._source_scope(candidate_ctx) != pristine_scope:
                return base
            if not isinstance(candidate.value, list) or not all(
                isinstance(row, dict) for row in candidate.value
            ):
                return base
            merged = fill_visual_consensus_nulls(
                base.value, candidate.value, plan
            )
            if merged is None:
                return base
            rows, _fills = merged
        except Exception:  # noqa: BLE001 -- ordinary planned output is the floor
            if pristine_base is not None and base != pristine_base:
                return pristine_base
            return base
        return StrategyOutput(
            value=rows,
            confidence=base.confidence,
            attested_evidence=base.attested_evidence,
            diagnostics=base.diagnostics,
        )
