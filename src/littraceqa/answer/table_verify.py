"""Source-bound verification of a frozen table, one row and cell at a time.

This is deliberately *not* another table generator.  It uses two independent
views of the same PDF page:

* PyMuPDF's native table finder serializes the page's text/grid geometry for a
  text-only reader; and
* the rendered page is read by the vision client.

A frozen value changes only when both readers return the same typed value for
the same requested row and the value is present in the native PDF packet.
Rows and non-table answer components are otherwise immutable.  Missing-row
recovery is a separate opt-in: a row may be appended only within an explicit
question-side cardinality budget and only when the same two-view agreement
holds.  A second, independent opt-in may substitute one source-verified row
for one surplus frozen row.  Substitution is cardinality-preserving and is
eligible only after a one-to-one assignment proves that the frozen row and an
explicit requested row cannot occupy the same contract slot.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from typing import Any, Iterable, Sequence

from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_assemble import matches_expected_key
from littraceqa.answer.table_extract import _parse_rows, _strip_method_attribution
from littraceqa.answer.table_route import route_expected_keys
from littraceqa.localize.pymupdf_runtime import serialized_pymupdf
from littraceqa.localize.render import render_page_png


_MAX_PACKET_CHARS = 18_000
_MAX_PACKETS_PER_TARGET = 4
_VISUAL_LOCATOR_BONUS = 300
_OWNER_MARKERS = re.compile(r"\b(?:ours?|proposed)\b", re.IGNORECASE)
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_AP_PATH_RE = re.compile(
    r"ap\s*\^\s*\{\s*([^}]+?)\s*\}\s*_\s*\{\s*([^}]+?)\s*\}",
    re.IGNORECASE,
)
_UNSAFE_REPLACEMENT_TOKENS = {
    "definition", "equation", "expression", "formula", "objective",
}

_GRID_SYSTEM = (
    "You verify ONE requested row in native PDF table-grid text. Follow the "
    "complete hierarchical column path, including dataset, metric, step/NFE, "
    "IPC, split, model size, and other qualifiers. Never take a nearby value "
    "from the same row or the requested value from a nearby row. Return at "
    "most one row using exactly the supplied schema column names. Copy the "
    "printed row label and printed value; do not infer or paraphrase. Respond "
    "with STRICT minified JSON only: {\"rows\":[{...}]}. Return {\"rows\":[]}"
    "unless both the requested row and value are explicitly present."
)

_VISION_SYSTEM = (
    "You verify ONE requested row on ONE rendered academic-paper page. Locate "
    "the relevant table and follow the complete hierarchical header path, "
    "including dataset, metric, step/NFE, IPC, split, model size, and other "
    "qualifiers. Never use an adjacent row or column. Return at most one row "
    "using exactly the supplied schema column names and copy the printed label "
    "and value. Respond with STRICT minified JSON only: {\"rows\":[{...}]}. "
    "Return {\"rows\":[]} when the exact intersection is not visible."
)

_GRID_CANONICAL_SYSTEM = (
    "You verify ONE requested method row in native PDF table-grid text. The "
    "requested method label may contain a one-character transcription error. "
    "If and only if the requested label is absent and exactly one printed row "
    "label differs by one insertion, deletion, or substitution, return that "
    "printed label and its values. Follow the complete hierarchical column "
    "path and never use an adjacent row or column. A paper may name the "
    "introduced method in the supplied page excerpt but label its table row "
    "as Ours; in that case return the explicitly named method with the Ours "
    "row values only when it is the unique one-edit alternative. Copy every "
    "field exactly. "
    "Respond with STRICT minified JSON only: {\"rows\":[{...}]}. Return "
    "{\"rows\":[]} on ambiguity or when the unique near-label row is absent."
)

_VISION_CANONICAL_SYSTEM = (
    "You verify ONE requested method row on ONE rendered academic-paper page. "
    "The requested method label may contain a one-character transcription "
    "error. If and only if the requested label is absent and exactly one "
    "visible printed row label differs by one insertion, deletion, or "
    "substitution, return that printed label and its values. If the page "
    "explicitly names the introduced method but labels its table row as Ours, "
    "return the named method with the Ours row values only when it is the "
    "unique one-edit alternative. Follow the full "
    "hierarchical header path and never use an adjacent row or column. Copy "
    "every field exactly. Respond with STRICT minified JSON only: "
    "{\"rows\":[{...}]}. Return {\"rows\":[]} on ambiguity or absence."
)


@dataclass(frozen=True)
class NativeTablePacket:
    paper_id: str
    page: int
    strategy: str
    rows: tuple[tuple[str | None, ...], ...]
    text: str
    score: int


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def _target_aliases(target: Sequence[Any]) -> list[str]:
    aliases: list[str] = []
    for value in target:
        raw = str(value or "").strip()
        for candidate in (raw, _PAREN_RE.sub("", raw).strip()):
            compact = _compact(candidate)
            if len(compact) >= 2 and compact not in aliases:
                aliases.append(compact)
    return aliases


def _edit_distance_le1(left: str, right: str) -> bool:
    """Return whether two compact labels differ by at most one edit."""

    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    index = 0
    while index < len(left) and left[index] == right[index]:
        index += 1
    if len(left) == len(right):
        return left[index + 1:] == right[index + 1:]
    return left[index:] == right[index + 1:]


def target_present(text: str, target: Sequence[Any]) -> bool:
    """Return whether text contains a target or one safe method-like typo."""

    normalized_text = normalize_text(text)
    compact_text = _compact(text)
    for value in target:
        raw = str(value or "").strip()
        for candidate in (raw, _PAREN_RE.sub("", raw).strip()):
            normalized = normalize_text(candidate)
            if not normalized:
                continue
            if re.search(
                rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])",
                normalized_text,
            ):
                return True
            compact = _compact(candidate)
            # Compact matching recovers line-broken or hyphen-split labels,
            # but is unsafe for short acronyms (ATT must not match
            # ``attention`` and IMM must not match ``immediate``).
            if len(compact) >= 5 and compact in compact_text:
                return True
            # Question labels occasionally differ from the printed method by
            # one character (for example AP-BPTT vs AT-BPTT). Restrict
            # fuzzy recovery to method-like tokens and labels of length >= 5;
            # the two independent readers still gate every accepted value.
            if len(compact) >= 5:
                candidates = (
                    _compact(token)
                    for token in re.findall(
                        r"[A-Za-z0-9][A-Za-z0-9_.-]{3,}", text
                    )
                )
                if any(
                    len(token) >= 5 and _edit_distance_le1(compact, token)
                    for token in candidates
                ):
                    return True
    return False


def nearest_printed_target(text: str, target: Sequence[Any]) -> str | None:
    """Return one unique exact/one-edit printed method token, else ``None``."""

    expected = {
        _compact(value)
        for value in target
        if len(_compact(value)) >= 5
    }
    if not expected:
        return None
    candidates: dict[str, str] = {}
    for raw in re.findall(r'"([^"\n]+)"', str(text or "")):
        compact = _compact(raw)
        if len(compact) < 5:
            continue
        if any(_edit_distance_le1(compact, value) for value in expected):
            candidates.setdefault(compact, raw.strip())
    return next(iter(candidates.values())) if len(candidates) == 1 else None


# Backward-compatible internal name for the existing verifier call sites.
_target_present = target_present


def _row_contains_target(row: Sequence[Any], target: Sequence[Any]) -> bool:
    return _target_present(" | ".join(str(value or "") for value in row), target)


def _page_score(ctx, paper_id: str, page, target, plan) -> int:
    text = str(getattr(page, "text", "") or "")
    if not _target_present(text, target):
        return -1
    normalized = normalize_text(text)
    score = 100
    score += 8 * sum(
        token in normalized
        for column in plan.value_cols
        for token in _TOKEN_RE.findall(normalize_text(column.get("name")))
        if len(token) >= 2
    )
    score += 4 * sum(
        token in normalized
        for token in _TOKEN_RE.findall(normalize_text(ctx.question))
        if len(token) >= 4
    )
    visual_pages = {
        int(item.page)
        for item in (getattr(ctx, "evidence", None) or [])
        if item.paper_id == paper_id
        and item.source_type in {"table", "figure"}
        and isinstance(item.page, int)
    }
    if int(page.page) in visual_pages:
        # The localizer has already identified this rendered page as a table
        # or figure supporting the selected paper.  It must outrank a broad
        # borderless-text grid from a nearby prose page.
        score += _VISUAL_LOCATOR_BONUS
    return score


def _candidate_pages(ctx, paper_id: str, target, plan, *, max_pages: int):
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(paper_id)
    pages = list(getattr(parsed, "pages", None) or [])
    ranked = [
        (_page_score(ctx, paper_id, page, target, plan), position, page)
        for position, page in enumerate(pages)
    ]
    ranked = [item for item in ranked if item[0] >= 0]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [page for _score, _position, page in ranked[:max_pages]]


@serialized_pymupdf
def _find_native_tables(pdf_bytes: bytes, page_1indexed: int):
    """Yield ``(strategy, rows)`` from PyMuPDF, imported lazily."""

    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if page_1indexed < 1 or page_1indexed > document.page_count:
            return []
        page = document[page_1indexed - 1]
        output = []
        for strategy in ("lines", "text"):
            try:
                finder = page.find_tables(
                    strategy=strategy,
                    min_words_vertical=2,
                    min_words_horizontal=1,
                )
            except Exception:  # noqa: BLE001 -- one detector may fail closed
                continue
            for table in finder.tables:
                try:
                    extracted = table.extract()
                except Exception:  # noqa: BLE001 -- malformed grid is skipped
                    continue
                rows = tuple(
                    tuple(
                        None if value is None else str(value)
                        for value in row
                    )
                    for row in extracted
                    if isinstance(row, (list, tuple))
                )
                if rows:
                    output.append((strategy, rows))
        return output


def _target_row_density(rows: Sequence[Sequence[Any]], target) -> int:
    """Reward table-like requested rows without interpreting their values."""

    counts = [
        sum(value is not None and bool(str(value).strip()) for value in row)
        for row in rows
        if _row_contains_target(row, target)
    ]
    return max(counts, default=0)


def native_packets_for_target(
    ctx,
    paper_id: str,
    plan,
    target: Sequence[Any],
    *,
    max_pages: int = 2,
) -> list[NativeTablePacket]:
    """Return relevance-ranked native grids that explicitly contain target."""

    pdf_bytes = (getattr(ctx, "pdf_bytes_by_id", None) or {}).get(paper_id)
    if not pdf_bytes:
        return []
    packets: list[NativeTablePacket] = []
    seen: set[str] = set()
    for page in _candidate_pages(
        ctx, paper_id, target, plan, max_pages=max_pages
    ):
        page_score = _page_score(ctx, paper_id, page, target, plan)
        for strategy, rows in _find_native_tables(pdf_bytes, int(page.page)):
            serialized = json.dumps(
                rows, ensure_ascii=False, separators=(",", ":")
            )
            if not _target_present(serialized, target):
                continue
            fingerprint = _compact(serialized)
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            target_rows = [row for row in rows if _row_contains_target(row, target)]
            owner_bonus = 120 if any(
                _OWNER_MARKERS.search(" ".join(str(value or "") for value in row))
                for row in target_rows
            ) else 0
            # Ruled-line tables and dense requested rows are substantially
            # less likely to be whole-page prose accidentally segmented into
            # a grid by the borderless ``text`` detector.
            strategy_bonus = 60 if strategy == "lines" else 0
            density_bonus = min(80, 10 * _target_row_density(rows, target))
            # Text-strategy grids can cover a whole dense two-column page.
            # Retain early header rows plus a bounded window around every
            # requested-row occurrence, then serialize valid JSON again.
            packet_rows = rows
            if len(serialized) > _MAX_PACKET_CHARS:
                positions = [
                    position
                    for position, row in enumerate(rows)
                    if _row_contains_target(row, target)
                ]
                keep = set(range(min(14, len(rows))))
                for position in positions:
                    keep.update(range(
                        max(0, position - 6), min(len(rows), position + 7)
                    ))
                packet_rows = tuple(
                    row for position, row in enumerate(rows) if position in keep
                )
            text = json.dumps(
                packet_rows, ensure_ascii=False, separators=(",", ":")
            )
            if len(text) > _MAX_PACKET_CHARS:
                target_only = tuple(rows[:8]) + tuple(target_rows[:4])
                text = json.dumps(
                    target_only, ensure_ascii=False, separators=(",", ":")
                )
            if len(text) > _MAX_PACKET_CHARS:
                continue
            packets.append(NativeTablePacket(
                paper_id=paper_id,
                page=int(page.page),
                strategy=strategy,
                rows=rows,
                text=text,
                score=page_score + owner_bonus + strategy_bonus + density_bonus,
            ))
    packets.sort(key=lambda item: (-item.score, item.page, item.strategy))
    return packets[:_MAX_PACKETS_PER_TARGET]


def _schema_names(plan) -> list[str]:
    return [
        *plan.row_key_cols,
        *(column["name"] for column in plan.value_cols),
    ]


def _target_label(plan, target) -> str:
    return " / ".join(
        f"{column}={value}"
        for column, value in zip(plan.row_key_cols, target)
        if value is not None and str(value).strip()
    )


def _page_text_excerpt(ctx, packet: NativeTablePacket, target) -> str:
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(packet.paper_id)
    page = parsed.page(packet.page) if parsed is not None else None
    text = str(getattr(page, "text", "") or "")
    if not text:
        return ""
    aliases = _target_aliases(target)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    positions = [
        position
        for position, line in enumerate(lines)
        if any(
            _edit_distance_le1(alias, _compact(token))
            for alias in aliases
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{3,}", line)
        )
    ]
    keep: set[int] = set()
    for position in positions:
        keep.update(range(max(0, position - 3), min(len(lines), position + 4)))
    excerpt = "\n".join(lines[position] for position in sorted(keep))
    return excerpt[:4_000]


def _grid_prompt(
    ctx,
    plan,
    target,
    packet,
    *,
    allow_source_label_correction: bool = False,
) -> str:
    page_excerpt = (
        _page_text_excerpt(ctx, packet, target)
        if allow_source_label_correction
        else ""
    )
    excerpt_block = (
        f"Same-page text excerpt:\n{page_excerpt}\n"
        if page_excerpt
        else ""
    )
    return (
        f"Question: {ctx.question}\n"
        f"Schema columns: {json.dumps(_schema_names(plan), ensure_ascii=False)}\n"
        f"Requested row: {_target_label(plan, target)}\n"
        f"Paper: {packet.paper_id}; PDF page: {packet.page}; "
        f"detector: {packet.strategy}\n"
        f"{excerpt_block}"
        f"Native grid JSON:\n{packet.text}\n\n"
        "Return the verified row only."
    )


def _vision_prompt(ctx, plan, target, packet) -> str:
    return (
        f"Question: {ctx.question}\n"
        f"Schema columns: {json.dumps(_schema_names(plan), ensure_ascii=False)}\n"
        f"Requested row: {_target_label(plan, target)}\n"
        f"This is page {packet.page} of paper {packet.paper_id}.\n\n"
        "Return the verified row only."
    )


def _matching_row(rows: Iterable[dict], plan, target) -> dict | None:
    matches = [
        row
        for row in rows
        if matches_expected_key(
            tuple(row.get(column) for column in plan.row_key_cols),
            target,
            plan.row_key_cols,
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _row_attested_by_packet(row: dict, plan, packet: NativeTablePacket) -> bool:
    # The text reader may normalize whitespace/punctuation, but it may not
    # introduce a scalar absent from the native PDF grid.
    compact_packet = _compact(packet.text)
    for column in [*plan.row_key_cols, *(item["name"] for item in plan.value_cols)]:
        value = row.get(column)
        if value is None:
            continue
        compact = _compact(value)
        if compact and compact not in compact_packet:
            return False
    return True


def _canonical_row_attested_by_source(
    row: dict,
    plan,
    packet: NativeTablePacket,
    ctx,
    target,
) -> bool:
    """Require the corrected key in page text and its values in one Ours row."""

    if len(plan.row_key_cols) != 1:
        return False
    source_label = row.get(plan.row_key_cols[0])
    compact_source = _compact(source_label)
    compact_target = _compact(target[0] if target else "")
    if (
        not compact_source
        or not compact_target
        or compact_source == compact_target
        or not _edit_distance_le1(compact_source, compact_target)
    ):
        return False
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(packet.paper_id)
    page = parsed.page(packet.page) if parsed is not None else None
    page_text = str(getattr(page, "text", "") or "")
    normalized_source = normalize_text(source_label)
    normalized_page = normalize_text(page_text)
    exact_source_present = bool(
        normalized_source
        and re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_source)}(?![a-z0-9])",
            normalized_page,
        )
    )
    if not exact_source_present and compact_source not in _compact(page_text):
        return False
    values = [
        row.get(column["name"])
        for column in plan.value_cols
        if row.get(column["name"]) is not None
    ]
    if len(values) != len(plan.value_cols):
        return False
    return any(
        _OWNER_MARKERS.search(" ".join(str(value or "") for value in native_row))
        and all(
            _compact(value)
            and _compact(value) in _compact(" | ".join(str(item or "") for item in native_row))
            for value in values
        )
        for native_row in packet.rows
    )


def _read_grid_row(
    ctx,
    plan,
    target,
    packet,
    llm,
    *,
    allow_source_label_correction: bool = False,
) -> dict | None:
    # A canonicalization read must preserve the printed source label.  An
    # empty expected-key contract keeps parsing schema-bound without rewriting
    # the returned key back to the question-side spelling.
    one_row_plan = replace(
        plan,
        expected_keys=[] if allow_source_label_correction else [tuple(target)],
    )
    try:
        response = llm.complete(
            _grid_prompt(
                ctx,
                one_row_plan,
                target,
                packet,
                allow_source_label_correction=allow_source_label_correction,
            ),
            system=(
                _GRID_CANONICAL_SYSTEM
                if allow_source_label_correction
                else _GRID_SYSTEM
            ),
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001 -- verification abstains on failure
        return None
    parsed = _parse_rows(response, one_row_plan)
    row = (
        parsed[0]
        if allow_source_label_correction and len(parsed) == 1
        else _matching_row(parsed, one_row_plan, target)
    )
    attested = bool(
        row is not None
        and (
            _row_attested_by_packet(row, one_row_plan, packet)
            or (
                allow_source_label_correction
                and _canonical_row_attested_by_source(
                    row, one_row_plan, packet, ctx, target
                )
            )
        )
    )
    if not attested:
        return None
    return row


def _read_vision_row(
    ctx,
    plan,
    target,
    packet,
    vision_llm,
    *,
    allow_source_label_correction: bool = False,
) -> dict | None:
    pdf_bytes = (getattr(ctx, "pdf_bytes_by_id", None) or {}).get(packet.paper_id)
    png = render_page_png(pdf_bytes, packet.page, dpi=180) if pdf_bytes else None
    if png is None:
        return None
    one_row_plan = replace(
        plan,
        expected_keys=[] if allow_source_label_correction else [tuple(target)],
    )
    try:
        response = vision_llm.complete(
            _vision_prompt(ctx, one_row_plan, target, packet),
            system=(
                _VISION_CANONICAL_SYSTEM
                if allow_source_label_correction
                else _VISION_SYSTEM
            ),
            temperature=0.0,
            images=[png],
        )
    except Exception:  # noqa: BLE001 -- verification abstains on failure
        return None
    parsed = _parse_rows(response, one_row_plan)
    if allow_source_label_correction:
        return parsed[0] if len(parsed) == 1 else None
    return _matching_row(parsed, one_row_plan, target)


def _value_key(value: Any):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return ("number", float(value))
    if isinstance(value, str):
        normalized = normalize_text(value)
        return ("string", normalized) if normalized else None
    return None


def _metric_paths(text: Any) -> set[tuple[str, str]]:
    return {
        (_compact(qualifier), _compact(dimension))
        for qualifier, dimension in _AP_PATH_RE.findall(str(text or ""))
        if _compact(qualifier) and _compact(dimension)
    }


def _column_conflicts_with_question(question: str, column: str) -> bool:
    """Detect an explicit symbolic metric mismatch; abstain rather than guess."""

    question_paths = _metric_paths(question)
    column_paths = _metric_paths(column)
    return bool(question_paths and column_paths and question_paths.isdisjoint(column_paths))


def _ambiguous_symbolic_path(ctx, packet: NativeTablePacket, column: str) -> bool:
    """True when the same symbolic leaf metric occurs under multiple groups."""

    paths = _metric_paths(column)
    if len(paths) != 1:
        return False
    qualifier, dimension = next(iter(paths))
    signature = f"ap{qualifier}"
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(packet.paper_id)
    page = parsed.page(packet.page) if parsed is not None else None
    if page is None:
        return False
    compact_page = _compact(page.text).replace(dimension, "")
    return compact_page.count(signature) > 1


def _agreed_cells(
    grid_row: dict,
    vision_row: dict,
    plan,
    *,
    ctx=None,
    packet: NativeTablePacket | None = None,
) -> dict[str, Any]:
    agreed: dict[str, Any] = {}
    for column in plan.value_cols:
        name = column["name"]
        if ctx is not None and _column_conflicts_with_question(ctx.question, name):
            continue
        if (
            ctx is not None
            and packet is not None
            and _ambiguous_symbolic_path(ctx, packet, name)
        ):
            continue
        key = _value_key(grid_row.get(name))
        if key is not None and key == _value_key(vision_row.get(name)):
            agreed[name] = grid_row[name]
    return agreed


def _unsafe_replacement_column(column: str) -> bool:
    return bool(
        set(_TOKEN_RE.findall(normalize_text(column)))
        & _UNSAFE_REPLACEMENT_TOKENS
    )


def _distinct_key_count(rows: Sequence[dict], columns: Sequence[str]) -> int:
    return len({
        tuple(normalize_text(row.get(column)) for column in columns)
        for row in rows
        if any(normalize_text(row.get(column)) for column in columns)
    })


def _normalized_key(row: dict, columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(normalize_text(row.get(column)) for column in columns)


def _exact_expected_match(
    row_key: Sequence[Any], expected: Sequence[Any], width: int
) -> bool:
    padded = tuple(list(expected)[:width] + [""] * (width - len(expected)))
    normalized_expected = tuple(normalize_text(value) for value in padded)
    # An empty expected component is a routing wildcard, not an exact scorer
    # key.  It must not reserve a row ahead of a fully specified contract.
    return all(normalized_expected) and tuple(
        normalize_text(value) for value in row_key
    ) == normalized_expected


def assign_rows_to_expected(
    rows: Sequence[dict], expected_keys: Sequence[Sequence[Any]], columns
) -> dict[int, int]:
    """Maximum one-to-one tolerant assignment, with scorer-exact pairs fixed.

    The admission relation is deliberately tolerant, so independent ``any``
    checks are not a coverage proof: both ``iCT`` and ``iCT-deep`` can match the
    one requested ``iCT-deep`` slot.  Reserve exact normalized pairs first,
    then run an augmenting-path matching over the remaining tolerant edges.
    The returned mapping is ``row_position -> expected_position``.
    """

    width = len(columns)
    row_keys = [
        tuple(row.get(column) for column in columns)
        for row in rows
    ]
    assignment: dict[int, int] = {}
    used_expected: set[int] = set()
    for row_position, row_key in enumerate(row_keys):
        for expected_position, expected in enumerate(expected_keys):
            if expected_position in used_expected:
                continue
            if _exact_expected_match(row_key, expected, width):
                assignment[row_position] = expected_position
                used_expected.add(expected_position)
                break

    remaining_rows = [
        position for position in range(len(rows)) if position not in assignment
    ]
    remaining_expected = {
        position
        for position in range(len(expected_keys))
        if position not in used_expected
    }
    edges = {
        row_position: [
            expected_position
            for expected_position in sorted(remaining_expected)
            if matches_expected_key(
                row_keys[row_position],
                expected_keys[expected_position],
                columns,
            )
        ]
        for row_position in remaining_rows
    }
    expected_to_row: dict[int, int] = {}

    def augment(row_position: int, seen: set[int]) -> bool:
        for expected_position in edges[row_position]:
            if expected_position in seen:
                continue
            seen.add(expected_position)
            prior = expected_to_row.get(expected_position)
            if prior is None or augment(prior, seen):
                expected_to_row[expected_position] = row_position
                return True
        return False

    # Constrained rows first makes the deterministic solution less likely to
    # spend a rare compatible slot on a broad alias.
    for row_position in sorted(
        remaining_rows, key=lambda pos: (len(edges[pos]), pos)
    ):
        augment(row_position, set())
    assignment.update(
        (row_position, expected_position)
        for expected_position, row_position in expected_to_row.items()
    )
    return assignment


# Backward-compatible alias for the focused tests and any saved experiment
# scripts that imported the old private helper.  New code uses the public name.
_assign_rows_to_expected = assign_rows_to_expected


def _substitution_targets(frozen, plan) -> list[tuple[tuple[Any, ...], int]]:
    """Pair missing requested keys with surplus rows at fixed cardinality."""

    expected = list(plan.expected_keys)
    # Substitution is not a pruning or cardinality-repair policy.  Requiring
    # exact physical and distinct cardinality equality prevents it from
    # touching open-ended, duplicate-key, underfilled, or overfilled tables.
    if (
        not expected
        or len(frozen) != len(expected)
        or _distinct_key_count(frozen, plan.row_key_cols) != len(frozen)
        or len({tuple(normalize_text(v) for v in key) for key in expected})
        != len(expected)
    ):
        return []
    assignment = assign_rows_to_expected(frozen, expected, plan.row_key_cols)
    unmatched_rows = [
        position for position in range(len(frozen)) if position not in assignment
    ]
    matched_expected = set(assignment.values())
    missing_expected = [
        position
        for position in range(len(expected))
        if position not in matched_expected
    ]
    if len(unmatched_rows) != len(missing_expected):
        return []
    return [
        (tuple(expected[expected_position]), row_position)
        for row_position, expected_position in zip(
            unmatched_rows, missing_expected, strict=True
        )
    ]


def _source_row_for_target(
    ctx,
    plan,
    target,
    llm,
    vision_llm,
    *,
    max_pages: int,
    allow_source_label_correction: bool = False,
):
    route = route_expected_keys(ctx, [tuple(target)])[0]
    route_ids = list(route.paper_ids)
    remaining = [pid for pid in ctx.paper_ids if pid not in route_ids]
    paper_ids = route_ids + remaining
    packets = [
        packet
        for paper_id in paper_ids
        for packet in native_packets_for_target(
            ctx, paper_id, plan, target, max_pages=max_pages
        )
    ]
    route_set = set(route.paper_ids) if route.status == "owned" else set()
    packets.sort(key=lambda item: (
        -(item.score + (150 if item.paper_id in route_set else 0)),
        paper_ids.index(item.paper_id),
        item.page,
    ))
    attempts = []
    for packet in packets[:_MAX_PACKETS_PER_TARGET]:
        grid_row = _read_grid_row(
            ctx,
            plan,
            target,
            packet,
            llm,
            allow_source_label_correction=allow_source_label_correction,
        )
        vision_row = (
            _read_vision_row(
                ctx,
                plan,
                target,
                packet,
                vision_llm,
                allow_source_label_correction=allow_source_label_correction,
            )
            if grid_row is not None
            else None
        )
        key_agreement = bool(
            grid_row is not None
            and vision_row is not None
            and _normalized_key(grid_row, plan.row_key_cols)
            == _normalized_key(vision_row, plan.row_key_cols)
        )
        agreed = (
            _agreed_cells(
                grid_row,
                vision_row,
                plan,
                ctx=ctx,
                packet=packet,
            )
            if grid_row is not None and vision_row is not None
            else {}
        )
        attempts.append({
            "paper_id": packet.paper_id,
            "page": packet.page,
            "strategy": packet.strategy,
            "grid_read": grid_row is not None,
            "vision_read": vision_row is not None,
            "agreed_columns": sorted(agreed),
            "source_key_agreement": key_agreement,
        })
        if agreed and (not allow_source_label_correction or key_agreement):
            return grid_row, agreed, attempts
    return None, {}, attempts


def verify_frozen_table(
    ctx,
    plan,
    llm,
    vision_llm,
    *,
    max_pages: int = 2,
    allow_row_additions: bool = False,
    allow_row_substitutions: bool = False,
    allow_cell_updates: bool = True,
    allow_source_key_canonicalization: bool = False,
):
    """Return ``(rows, diagnostics)`` after conservative two-view verification."""

    if (
        isinstance(max_pages, bool)
        or not isinstance(max_pages, int)
        or not 1 <= max_pages <= 5
    ):
        raise ValueError("max_pages must be an integer from 1 to 5")
    if not isinstance(allow_row_additions, bool):
        raise ValueError("allow_row_additions must be a boolean")
    if not isinstance(allow_row_substitutions, bool):
        raise ValueError("allow_row_substitutions must be a boolean")
    if not isinstance(allow_cell_updates, bool):
        raise ValueError("allow_cell_updates must be a boolean")
    if not isinstance(allow_source_key_canonicalization, bool):
        raise ValueError("allow_source_key_canonicalization must be a boolean")
    frozen = [dict(row) for row in (ctx.frozen_table_rows or [])]
    if not frozen or not plan.value_cols:
        return frozen, {
            "status": "ineligible",
            "changed": False,
            "filled_cells": 0,
            "replaced_cells": 0,
            "added_rows": 0,
            "substituted_rows": 0,
            "canonicalized_rows": 0,
            "max_pages": max_pages,
            "allow_row_additions": allow_row_additions,
            "allow_row_substitutions": allow_row_substitutions,
            "allow_cell_updates": allow_cell_updates,
            "allow_source_key_canonicalization": (
                allow_source_key_canonicalization
            ),
            "decisions": [],
        }

    decisions: list[dict[str, Any]] = []
    filled = replaced = added = substituted = canonicalized = 0
    substitutions = (
        _substitution_targets(frozen, plan)
        if allow_row_substitutions
        else []
    )
    substitution_positions = {position for _target, position in substitutions}
    targets: list[tuple[tuple[Any, ...], int | None, str]] = [
        (
            tuple(row.get(column) for column in plan.row_key_cols),
            position,
            "existing",
        )
        for position, row in enumerate(frozen)
        if any(row.get(column) for column in plan.row_key_cols)
        and position not in substitution_positions
    ]
    targets.extend(
        (target, position, "replacement")
        for target, position in substitutions
    )
    addition_budget = 0
    if allow_row_additions and plan.expected_keys:
        missing = [
            tuple(expected)
            for expected in plan.expected_keys
            if not any(
                matches_expected_key(
                    tuple(row.get(column) for column in plan.row_key_cols),
                    expected,
                    plan.row_key_cols,
                )
                for row in frozen
            )
        ]
        addition_budget = max(
            0,
            len(plan.expected_keys) - _distinct_key_count(frozen, plan.row_key_cols),
        )
        targets.extend(
            (target, None, "missing")
            for target in missing[:addition_budget]
        )

    for target, position, target_kind in targets:
        conflicting_columns = [
            column["name"]
            for column in plan.value_cols
            if _column_conflicts_with_question(
                ctx.question, column["name"]
            )
        ]
        if len(conflicting_columns) == len(plan.value_cols):
            decisions.append({
                "target": list(target),
                "kind": target_kind,
                "attempts": [],
                "accepted": False,
                "changes": [],
                "skipped": "question_schema_metric_conflict",
                "conflicting_columns": conflicting_columns,
            })
            continue
        method_key = (
            len(plan.row_key_cols) == 1
            and normalize_text(plan.row_key_cols[0])
            in {"method", "methods", "method name"}
        )
        canonicalization_read = bool(
            allow_source_key_canonicalization
            and target_kind == "existing"
            and method_key
        )
        source_row, agreed, attempts = _source_row_for_target(
            ctx,
            plan,
            target,
            llm,
            vision_llm,
            max_pages=max_pages,
            allow_source_label_correction=canonicalization_read,
        )
        decision = {
            "target": list(target),
            "kind": target_kind,
            "attempts": attempts,
            "accepted": False,
            "changes": [],
        }
        if source_row is None or not agreed:
            decisions.append(decision)
            continue
        if canonicalization_read:
            source_key = tuple(
                source_row.get(column) for column in plan.row_key_cols
            )
            compact_target = _compact(target[0] if target else "")
            compact_source = _compact(source_key[0] if source_key else "")
            if (
                not compact_target
                or not compact_source
                or compact_target == compact_source
                or not _edit_distance_le1(compact_target, compact_source)
            ):
                decision["skipped"] = "source_key_not_unique_one_edit"
                decisions.append(decision)
                continue
            required_columns = {column["name"] for column in plan.value_cols}
            if set(agreed) != required_columns:
                decision["skipped"] = "partial_value_agreement"
                decision["required_columns"] = sorted(required_columns)
                decisions.append(decision)
                continue
            target_row = frozen[position]
            if any(
                target_row.get(column) is not None
                and _value_key(target_row.get(column)) != _value_key(agreed[column])
                for column in required_columns
            ):
                decision["skipped"] = "existing_value_disagrees"
                decisions.append(decision)
                continue
            candidate = dict(target_row)
            candidate[plan.row_key_cols[0]] = source_key[0]
            for column in required_columns:
                if candidate.get(column) is None:
                    candidate[column] = agreed[column]
            normalized_candidate = _normalized_key(candidate, plan.row_key_cols)
            if any(
                other_position != position
                and _normalized_key(row, plan.row_key_cols) == normalized_candidate
                for other_position, row in enumerate(frozen)
            ):
                decision["skipped"] = "duplicate_source_key"
                decisions.append(decision)
                continue
            before = dict(target_row)
            frozen[position] = candidate
            canonicalized += 1
            decision["accepted"] = True
            decision["changes"].append({
                "row_key_canonicalized": {"before": before, "after": candidate}
            })
            decisions.append(decision)
            continue
        if target_kind == "replacement":
            required_columns = {column["name"] for column in plan.value_cols}
            if set(agreed) != required_columns:
                decision["skipped"] = "partial_value_agreement"
                decision["required_columns"] = sorted(required_columns)
                decisions.append(decision)
                continue
            candidate = {
                column: source_row.get(column)
                for column in plan.row_key_cols
            }
            candidate.update(agreed)
            if not all(candidate.get(column) for column in plan.row_key_cols):
                decisions.append(decision)
                continue
            for column in plan.row_key_cols:
                if normalize_text(column) in {"method", "methods", "method name"}:
                    candidate[column] = _strip_method_attribution(candidate[column])
            candidate_key = tuple(
                candidate.get(column) for column in plan.row_key_cols
            )
            if not matches_expected_key(
                candidate_key, target, plan.row_key_cols
            ):
                decision["skipped"] = "source_key_misses_target"
                decisions.append(decision)
                continue
            normalized_candidate = _normalized_key(candidate, plan.row_key_cols)
            if any(
                other_position != position
                and _normalized_key(row, plan.row_key_cols) == normalized_candidate
                for other_position, row in enumerate(frozen)
            ):
                decision["skipped"] = "duplicate_source_key"
                decisions.append(decision)
                continue
            before = dict(frozen[position])
            frozen[position] = candidate
            substituted += 1
            decision["accepted"] = True
            decision["changes"].append({
                "row_substituted": {"before": before, "after": candidate}
            })
            decisions.append(decision)
            continue
        if position is None:
            if added >= addition_budget:
                decisions.append(decision)
                continue
            candidate = {
                column: source_row.get(column)
                for column in plan.row_key_cols
            }
            candidate.update({column["name"]: None for column in plan.value_cols})
            candidate.update(agreed)
            if not all(candidate.get(column) for column in plan.row_key_cols):
                decisions.append(decision)
                continue
            for column in plan.row_key_cols:
                if normalize_text(column) in {"method", "methods", "method name"}:
                    candidate[column] = _strip_method_attribution(candidate[column])
            frozen.append(candidate)
            added += 1
            decision["accepted"] = True
            decision["changes"].append({"row_added": candidate})
            decisions.append(decision)
            continue

        if not allow_cell_updates:
            decision["skipped"] = "cell_updates_disabled"
            decisions.append(decision)
            continue
        target_row = frozen[position]
        for column, value in agreed.items():
            before = target_row.get(column)
            if _value_key(before) == _value_key(value):
                continue
            if before is not None and _unsafe_replacement_column(column):
                continue
            target_row[column] = value
            kind = "filled" if before is None else "replaced"
            filled += int(kind == "filled")
            replaced += int(kind == "replaced")
            decision["changes"].append({
                "column": column,
                "before": before,
                "after": value,
                "kind": kind,
            })
        decision["accepted"] = bool(decision["changes"])
        decisions.append(decision)

    return frozen, {
        "status": "verified",
        "changed": bool(
            filled or replaced or added or substituted or canonicalized
        ),
        "filled_cells": filled,
        "replaced_cells": replaced,
        "added_rows": added,
        "substituted_rows": substituted,
        "canonicalized_rows": canonicalized,
        "max_pages": max_pages,
        "allow_row_additions": allow_row_additions,
        "allow_row_substitutions": allow_row_substitutions,
        "allow_cell_updates": allow_cell_updates,
        "allow_source_key_canonicalization": allow_source_key_canonicalization,
        "decisions": decisions,
    }
