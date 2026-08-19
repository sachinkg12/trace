"""Produce source-attested table facts from PDF geometry plus visual review.

PyMuPDF's table finder misses some borderless scientific tables and sometimes
segments an entire two-column page as one grid.  The word geometry is still
useful: it preserves row alignment and the x-positions of hierarchical
headers.  This opt-in producer sends a bounded coordinate packet to a text
reader, then asks an independent visual reader to verify the same proposed
row, header path, and value.  Only exact two-view agreements enter the fact
ledger; neither reader emits scorer-facing rows directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import re
from typing import Any, Callable, Protocol, runtime_checkable

from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_extract import (
    _coerce_string_cell,
    _strip_method_attribution,
)
from littraceqa.answer.table_fact_ledger import (
    _EXCLUSIVE_HEADER_FAMILIES,
    _OPTIONAL_RESULT_TOKENS,
    _token_signature,
    CellFact,
    RowTarget,
    TableFactLedger,
    complete_header_path,
    make_cell_fact_with_reasons,
    make_row_attestation,
)
from littraceqa.answer.table_value_contract import build_cell_value_contract
from littraceqa.answer.table_verify import nearest_printed_target, target_present
from littraceqa.localize.pymupdf_runtime import serialized_pymupdf
from littraceqa.localize.render import render_page_clip_png


_MAX_PACKET_CHARS = 18_000
_MAX_PACKETS_PER_TARGET = 4
_Y_BEFORE = 520.0
_Y_AFTER = 115.0
_WORD_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_OWNER_ROW_RE = re.compile(
    r"^(?:ours?|our method|proposed(?: method)?)$", re.IGNORECASE
)
_STEP_REQUIREMENT_RE = re.compile(
    r"(?<!\d)(\d+)\s*(?:-|\s)*(?:step|nfe)s?\b", re.IGNORECASE
)
_POSITIONED_WORD_RE = re.compile(r'x=([-+]?\d+(?:\.\d+)?)\s+"([^"\n]*)"')
_POSITIONED_LINE_RE = re.compile(r"^y=([-+]?\d+(?:\.\d+)?):")

_GEOMETRY_SYSTEM = (
    "You verify ONE requested table row using PDF word coordinates. The packet "
    "lists visible words as y-lines with x positions. Follow the complete "
    "hierarchical header path, including dataset, metric, step/NFE, split, "
    "IPC, voting, and model-size qualifiers. A method may occupy several "
    "physical lines for different step counts; select only the line whose "
    "headers and row-level qualifiers answer the requested schema column. "
    "The requested label may differ from the printed method by exactly one "
    "character. A paper can also name its introduced method in the page text "
    "but print its result row as Ours; only in that case, return the explicitly "
    "named printed method with the Ours-row scalar. "
    "Copy the printed row label and scalar exactly. Never infer a value. "
    "Respond with STRICT minified JSON only: "
    '{"printed_row_key":["..."],"cells":[{"column_name":"exact schema '
    'name","header_path":["outer","leaf","row qualifier"],'
    '"raw_value":"printed scalar"}]}. Return an empty cells list on any '
    "ambiguity."
)

_VISION_SYSTEM = (
    "You independently verify proposed cells on ONE rendered academic-paper "
    "page. Check the exact printed row, complete hierarchical header path, "
    "row-level qualifier such as step/NFE, and scalar. Do not repair or infer. "
    "The requested label may have one transcription error. If the page names "
    "that introduced method but its unique result row says Ours, return the "
    "explicitly named printed method with the Ours-row scalar. "
    "Return STRICT minified JSON in exactly the same printed_row_key/cells "
    "shape, containing only proposals that are visibly correct. Return an "
    "empty cells list when a proposal points to an adjacent row or column."
)


@dataclass(frozen=True)
class GeometryPacket:
    paper_id: str
    page: int
    source_type: str
    object_id: str | None
    bbox: tuple[float, float, float, float]
    text: str
    score: int


@dataclass(frozen=True)
class ProposedCell:
    printed_row_key: tuple[str, ...]
    column_name: str
    header_path: tuple[str, ...]
    raw_value: str
    typed_value: object


@dataclass(frozen=True)
class FactProductionResult:
    ledger: TableFactLedger
    diagnostics: tuple[dict[str, Any], ...]


@runtime_checkable
class TableFactProducer(Protocol):
    def produce(
        self,
        ctx,
        ledger: TableFactLedger,
        *,
        text_llm,
        vision_llm,
    ) -> FactProductionResult: ...


_PRODUCERS: dict[str, Callable[..., TableFactProducer]] = {}


def register_table_fact_producer(name: str):
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("table-fact producer name must be non-empty")

    def decorate(producer: Callable[..., TableFactProducer]):
        if clean_name in _PRODUCERS:
            raise ValueError(f"table-fact producer {clean_name!r} already registered")
        _PRODUCERS[clean_name] = producer
        return producer

    return decorate


def build_table_fact_producer(name: str, **kwargs: Any) -> TableFactProducer:
    if name in {
        "native_text",
        "native_table",
        "native_table_rows",
        "native_text_table_rows",
        "native_text_table_consensus",
        "native_targeted_text",
        "native_consensus_targeted",
        "source_dispatch",
    } and name not in _PRODUCERS:
        # Keep the geometry producer independent and import the optional
        # native-source producers only when composition explicitly requests
        # one.
        import littraceqa.answer.table_fact_native  # noqa: F401
    if name not in _PRODUCERS:
        raise KeyError(
            f"no table-fact producer registered as {name!r}; "
            f"have {sorted(_PRODUCERS)}"
        )
    return _PRODUCERS[name](**kwargs)


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _anchor_aliases(target: RowTarget) -> tuple[str, ...]:
    out: list[str] = []
    for alias in target.aliases:
        for value in alias:
            compact = _compact(value)
            if len(compact) >= 3 and compact not in out:
                out.append(compact)
    return tuple(out)


def _geometry_anchor_aliases(target: RowTarget) -> tuple[tuple[str, ...], ...]:
    """Return source-row search aliases without widening row admission.

    Scientific tables often print ``Ground Truth`` on one physical line while
    the question says ``ground-truth prompts``, or split a leading ``w/`` from
    ``Cube RCNN``.  These concise forms are safe for locating a page window,
    but they are deliberately *not* added to ``RowTarget.aliases`` and cannot
    by themselves authorize an output row or value.
    """

    output: list[tuple[str, ...]] = list(target.aliases)
    if len(target.expected_key) != 1:
        return tuple(output)
    for alias in target.aliases:
        if len(alias) != 1:
            continue
        raw = str(alias[0] or "").strip()
        variants = {
            re.sub(r"(?i)^\s*(?:w\s*/|with)\s*", "", raw).strip(),
            re.sub(r"(?i)\s+(?:prompts?|2d\s+detections?)\s*$", "", raw).strip(),
        }
        variants.add(re.sub(
            r"(?i)\s+(?:prompts?|2d\s+detections?)\s*$",
            "",
            re.sub(r"(?i)^\s*(?:w\s*/|with)\s*", "", raw),
        ).strip())
        for variant in sorted(variants):
            if len(_compact(variant)) < 5:
                continue
            candidate = (variant,)
            if candidate not in output:
                output.append(candidate)
    return tuple(output)


def _line_groups(words) -> list[tuple[float, list[tuple]]]:
    groups: list[tuple[float, list[tuple]]] = []
    for word in sorted(words, key=lambda item: (float(item[1]), float(item[0]))):
        y = float(word[1])
        if not groups or abs(groups[-1][0] - y) > 2.0:
            groups.append((y, [word]))
        else:
            old_y, items = groups[-1]
            items.append(word)
            groups[-1] = ((old_y * (len(items) - 1) + y) / len(items), items)
    for _y, items in groups:
        items.sort(key=lambda item: float(item[0]))
    return groups


def _line_text(words: list[tuple]) -> str:
    return " ".join(str(word[4]) for word in words)


def _packet_text(groups: list[tuple[float, list[tuple]]]) -> str:
    lines: list[str] = []
    for y, words in groups:
        positioned = " | ".join(
            f'x={float(word[0]):.1f} "{str(word[4])}"' for word in words
        )
        lines.append(f"y={y:.1f}: {positioned}")
    return "\n".join(lines)


def _page_object(ctx, paper_id: str, page: int) -> tuple[str, str | None]:
    candidates = [
        item
        for item in (getattr(ctx, "evidence", None) or [])
        if item.paper_id == paper_id
        and item.page == page
        and item.source_type in {"table", "figure"}
    ]
    candidates.sort(
        key=lambda item: (
            item.source_type != "table",
            -float(getattr(item, "confidence", 0.0)),
            item.object_id or "",
        )
    )
    if not candidates:
        return "table", None
    return candidates[0].source_type, candidates[0].object_id or None


@serialized_pymupdf
def geometry_packets_for_target(
    ctx,
    target: RowTarget,
    value_columns,
    *,
    max_pages: int,
) -> list[GeometryPacket]:
    """Return bounded word-coordinate windows containing a target alias."""

    if not _anchor_aliases(target):
        return []
    anchor_aliases = _geometry_anchor_aliases(target)
    packets: list[GeometryPacket] = []
    for paper_id in target.owner_papers:
        pdf_bytes = (getattr(ctx, "pdf_bytes_by_id", None) or {}).get(paper_id)
        if not pdf_bytes:
            continue
        try:
            import fitz

            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                for page_index in range(document.page_count):
                    page = document.load_page(page_index)
                    words = page.get_text("words", sort=True)
                    if not words:
                        continue
                    groups = _line_groups(words)
                    anchors = [
                        (y, line_words)
                        for y, line_words in groups
                        if any(
                            target_present(_line_text(line_words), alias)
                            for alias in anchor_aliases
                        )
                    ]
                    if not anchors:
                        continue
                    for anchor_y, anchor_words in anchors:
                        window = [
                            (y, line_words)
                            for y, line_words in groups
                            if anchor_y - _Y_BEFORE <= y <= anchor_y + _Y_AFTER
                        ]
                        text = _packet_text(window)
                        if not text or len(text) > _MAX_PACKET_CHARS:
                            continue
                        normalized_window = normalize_text(
                            " ".join(_line_text(items) for _y, items in window)
                        )
                        column_hits = sum(
                            token in normalized_window
                            for column in value_columns
                            for token in _WORD_TOKEN_RE.findall(
                                normalize_text(column.name)
                            )
                            if len(token) >= 2
                        )
                        numeric_words = sum(
                            bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(word[4])))
                            for _y, line_words in window
                            for word in line_words
                        )
                        anchor_text = normalize_text(_line_text(anchor_words))
                        owner_marker = int(
                            "ours" in anchor_text or "our " in anchor_text
                        )
                        qualifier_hits = sum(
                            _compact(qualifier) in _compact(normalized_window)
                            for qualifier in target.header_requirements
                            if len(_compact(qualifier)) >= 3
                        )
                        table_marker = int("table" in normalized_window)
                        page_number = page_index + 1
                        source_type, object_id = _page_object(
                            ctx, paper_id, page_number
                        )
                        packets.append(GeometryPacket(
                            paper_id=paper_id,
                            page=page_number,
                            source_type=source_type,
                            object_id=object_id,
                            bbox=(
                                0.0,
                                max(0.0, anchor_y - _Y_BEFORE),
                                float(page.rect.width),
                                min(float(page.rect.height), anchor_y + _Y_AFTER),
                            ),
                            text=text,
                            score=(
                                100
                                + 20 * column_hits
                                + 80 * owner_marker
                                + 40 * qualifier_hits
                                + 30 * table_marker
                                + 300 * int(object_id is not None)
                                + min(30, numeric_words)
                            ),
                        ))
        except Exception:  # noqa: BLE001 -- one malformed PDF fails closed
            continue
    # One highest-scoring window per physical page prevents four abstract
    # mentions from exhausting the global packet budget before the result
    # table later in the paper.
    deduped: dict[tuple[Any, ...], GeometryPacket] = {}
    for packet in packets:
        key = (packet.paper_id, packet.page)
        prior = deduped.get(key)
        if prior is None or (packet.score, tuple(-v for v in packet.bbox)) > (
            prior.score,
            tuple(-v for v in prior.bbox),
        ):
            deduped[key] = packet
    ranked = sorted(
        deduped.values(),
        key=lambda item: (-item.score, item.paper_id, item.page, item.bbox),
    )
    # ``max_pages`` bounds distinct pages; the hard packet cap bounds repeated
    # target mentions on a selected page.
    selected_pages: list[tuple[str, int]] = []
    output: list[GeometryPacket] = []
    for packet in ranked:
        page_key = (packet.paper_id, packet.page)
        if page_key not in selected_pages:
            if len(selected_pages) >= max_pages:
                continue
            selected_pages.append(page_key)
        output.append(packet)
        if len(output) >= _MAX_PACKETS_PER_TARGET:
            break
    return output


def _schema_prompt(ledger: TableFactLedger) -> str:
    return json.dumps(
        {
            "row_key_cols": list(ledger.row_key_cols),
            "value_columns": [column.name for column in ledger.value_columns],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _geometry_prompt(
    ctx, ledger: TableFactLedger, target: RowTarget, packet: GeometryPacket
) -> str:
    return (
        f"Question: {ctx.question}\n"
        f"Schema: {_schema_prompt(ledger)}\n"
        f"Requested row identity: {json.dumps(list(target.expected_key))}\n"
        f"Search aliases: {json.dumps([list(alias) for alias in target.aliases])}\n"
        f"Qualifiers: {json.dumps(list(target.qualifiers))}\n"
        f"Atomic claim tuple: {json.dumps(asdict(target.claim))}\n"
        f"Paper: {packet.paper_id}; physical PDF page: {packet.page}\n"
        f"Coordinate packet:\n{packet.text}\n\n"
        "Return the JSON object only."
    )


def _vision_prompt(
    ctx,
    ledger: TableFactLedger,
    target: RowTarget,
    packet: GeometryPacket,
    proposals: list[ProposedCell],
) -> str:
    payload = {
        "printed_row_key": list(proposals[0].printed_row_key),
        "cells": [
            {
                "column_name": item.column_name,
                "header_path": list(item.header_path),
                "raw_value": item.raw_value,
            }
            for item in proposals
        ],
    }
    return (
        f"Question: {ctx.question}\n"
        f"Schema: {_schema_prompt(ledger)}\n"
        f"Requested row identity: {json.dumps(list(target.expected_key))}\n"
        f"This is physical PDF page {packet.page} of {packet.paper_id}.\n"
        f"Proposals to verify: {json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return only the verified subset in the same JSON shape."
    )


def _strip_fences(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_printed_row_key(
    payload: object, ledger: TableFactLedger
) -> tuple[str, ...] | None:
    """Parse only the lossless row identity from a semantic-reader payload.

    Row identity and cell extraction are separate claims.  Keeping this parser
    independent lets a source producer retain an exact, physically visible
    row label even when the reader correctly abstains on every value cell.
    """
    if not isinstance(payload, dict):
        return None
    raw_key = payload.get("printed_row_key")
    # A one-column row key has only one lossless representation.  Strict JSON
    # models occasionally emit that scalar directly instead of wrapping it in
    # a one-element list.  Multi-key schemas remain strict.
    if (
        isinstance(raw_key, str)
        and raw_key.strip()
        and len(ledger.row_key_cols) == 1
    ):
        raw_key = [raw_key]
    if not isinstance(raw_key, list) or len(raw_key) != len(ledger.row_key_cols):
        return None
    printed = tuple(
        _strip_method_attribution(str(value or "").strip())
        if normalize_text(column) in {"method", "methods", "method name"}
        else str(value or "").strip()
        for column, value in zip(ledger.row_key_cols, raw_key, strict=True)
    )
    return None if any(not value for value in printed) else printed


def _parse_proposals(
    response: str, ledger: TableFactLedger
) -> list[ProposedCell]:
    try:
        payload = json.loads(_strip_fences(response))
    except (json.JSONDecodeError, TypeError):
        return []
    printed = _parse_printed_row_key(payload, ledger)
    if printed is None:
        return []
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        return []
    out: list[ProposedCell] = []
    seen: set[str] = set()
    for item in raw_cells:
        if not isinstance(item, dict):
            continue
        column_name = str(item.get("column_name") or "").strip()
        if column_name in seen:
            return []
        try:
            column = ledger.column(column_name)
        except KeyError:
            continue
        raw_path = item.get("header_path")
        raw_value = str(item.get("raw_value") or "").strip()
        if not isinstance(raw_path, list) or not raw_value:
            continue
        path = tuple(str(value or "").strip() for value in raw_path)
        if not path or any(not value for value in path):
            continue
        if column.value_type != "number":
            cleaned = _coerce_string_cell(raw_value, column_name)
            if cleaned is not None:
                # Preserve existing schema-specific string normalization, then
                # let the value contract record source/scorer representations.
                raw_value = _coerce_string_cell(
                    re.sub(r"\\pm", "±", cleaned, flags=re.IGNORECASE),
                    column_name,
                ) or raw_value
        contract = build_cell_value_contract(
            raw_value,
            column_type=column.value_type,
            column_name=column_name,
            header_path=path,
        )
        if contract is None:
            continue
        typed = contract.source_value
        raw_value = contract.source_literal
        seen.add(column_name)
        out.append(ProposedCell(
            printed_row_key=printed,
            column_name=column_name,
            header_path=path,
            raw_value=raw_value,
            typed_value=typed,
        ))
    return out


def _header_key(path: tuple[str, ...]) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in _WORD_TOKEN_RE.findall(" ".join(path).casefold()):
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.append("step" if token == "nfe" else token)
    return tuple(tokens)


def _value_key(value: object) -> tuple[str, object] | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return "number", float(value)
    if isinstance(value, str):
        normalized = normalize_text(value)
        return ("string", normalized) if normalized else None
    return None


def _agreed_proposals(
    geometry: list[ProposedCell], vision: list[ProposedCell]
) -> list[ProposedCell]:
    if not geometry or not vision:
        return []
    geometry_key = tuple(normalize_text(value) for value in geometry[0].printed_row_key)
    vision_key = tuple(normalize_text(value) for value in vision[0].printed_row_key)
    if geometry_key != vision_key:
        return []
    vision_by_column = {item.column_name: item for item in vision}
    agreed: list[ProposedCell] = []
    for item in geometry:
        other = vision_by_column.get(item.column_name)
        if other is None:
            continue
        if (
            _value_key(item.typed_value) == _value_key(other.typed_value)
            and _header_key(item.header_path) == _header_key(other.header_path)
        ):
            agreed.append(item)
    return agreed


def _resolve_transposed_header_target(
    proposal: ProposedCell,
    target: RowTarget,
    packet: GeometryPacket,
) -> ProposedCell:
    """Re-orient a visibly transposed generic-value fact.

    Some source tables place requested entities in columns and metrics in rows
    (``PISA`` column, ``Part Selection (%)`` row), while the answer schema asks
    for one entity per output row.  Both readers first agree on the physical
    source orientation.  Only then may this adapter bind the scorer row to an
    exact target alias visibly returned as a header and retain the physical row
    label in the complete header path.  No fuzzy or question-only label can
    trigger the transformation.
    """

    if (
        len(proposal.printed_row_key) != 1
        or normalize_text(proposal.column_name) not in {"value", "values"}
    ):
        return proposal
    if any(
        len(alias) == 1
        and normalize_text(proposal.printed_row_key[0])
        == normalize_text(alias[0])
        for alias in target.aliases
    ):
        return proposal

    printed_source = " ".join(re.findall(r'"([^"\n]*)"', packet.text))
    visible_aliases: list[str] = []
    for header in proposal.header_path:
        normalized_header = normalize_text(header)
        for alias in target.aliases:
            if (
                len(alias) == 1
                and normalized_header == normalize_text(alias[0])
                and target_present(printed_source, alias)
                and alias[0] not in visible_aliases
            ):
                visible_aliases.append(alias[0])
    if len(visible_aliases) != 1 or not target_present(
        printed_source, proposal.printed_row_key
    ):
        return proposal

    path: list[str] = []
    seen: set[str] = set()
    for value in (*proposal.printed_row_key, *proposal.header_path):
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        path.append(value)
    return replace(
        proposal,
        printed_row_key=(visible_aliases[0],),
        header_path=tuple(path),
    )


def _with_row_local_step_qualifier(
    path: tuple[str, ...],
    proposal: ProposedCell,
    target: RowTarget,
    packet: GeometryPacket,
) -> tuple[str, ...]:
    """Restore a step/NFE label proved on the exact geometry row.

    Readers sometimes return ``CIFAR-10 -> FID`` while the NFE is a separate
    numeric cell on the method row.  A page-wide occurrence of ``1`` is not
    evidence, so restoration is permitted only when the requested NFE, printed
    row key, and proposed scalar are all visible on one coordinate line.
    Dataset labels are deliberately never restored here.
    """

    requirements = tuple(dict.fromkeys(
        match.group(1)
        for value in (proposal.column_name, *target.header_requirements)
        for match in _STEP_REQUIREMENT_RE.finditer(str(value or ""))
    ))
    if not requirements:
        return path
    path_text = " ".join(path)
    path_steps = {
        match.group(1) for match in _STEP_REQUIREMENT_RE.finditer(path_text)
    }
    if set(requirements).issubset(path_steps):
        return path
    row_key = " ".join(proposal.printed_row_key)
    raw_compact = _compact(proposal.raw_value)
    for line in packet.text.splitlines():
        printed_words = re.findall(r'"([^"\n]*)"', line)
        if not printed_words:
            continue
        visible = " ".join(printed_words)
        if (
            not target_present(visible, (row_key,))
            or raw_compact not in _compact(visible)
        ):
            continue
        # A scalar such as ``1.92`` must not be mistaken for an NFE cell ``1``.
        # Require one complete quoted PDF word equal to the requested count.
        visible_words_normalized = {
            normalize_text(word) for word in printed_words
        }
        restored = [
            f"{number}-step"
            for number in requirements
            if number in visible_words_normalized
        ]
        if len(restored) == len(requirements):
            return (*path, *restored)
    return path


def _positioned_lines(text: str) -> list[tuple[float, list[tuple[float, str]]]]:
    lines: list[tuple[float, list[tuple[float, str]]]] = []
    for raw_line in str(text or "").splitlines():
        y_match = _POSITIONED_LINE_RE.match(raw_line)
        if y_match is None:
            continue
        words = [
            (float(match.group(1)), match.group(2))
            for match in _POSITIONED_WORD_RE.finditer(raw_line)
        ]
        if words:
            lines.append((float(y_match.group(1)), words))
    return lines


def _sequence_x_positions(
    words: list[tuple[float, str]], value: str
) -> list[float]:
    wanted = _compact(value)
    if not wanted:
        return []
    output: list[float] = []
    for start in range(len(words)):
        combined = ""
        for _position, (_x, word) in enumerate(words[start:], start=start):
            combined += _compact(word)
            if combined == wanted:
                output.append(words[start][0])
                break
            if len(combined) >= len(wanted):
                break
    return output


def _proposal_value_position(
    packet: GeometryPacket, proposal: ProposedCell
) -> float | None:
    """Locate one scalar nearest its printed row; reject positional ties."""

    lines = _positioned_lines(packet.text)
    row_value = " ".join(proposal.printed_row_key)
    row_positions = [
        (y, x)
        for y, words in lines
        for x in _sequence_x_positions(words, row_value)
    ]
    if not row_positions:
        return None
    candidates: list[tuple[float, float, float, float]] = []
    for y, words in lines:
        for x in _sequence_x_positions(words, proposal.raw_value):
            compatible = [
                (abs(y - row_y), abs(x - row_x))
                for row_y, row_x in row_positions
                if x >= row_x and abs(x - row_x) <= 300.0
            ]
            if not compatible:
                continue
            vertical, horizontal = min(
                compatible, key=lambda item: (4 * item[0] + item[1], *item)
            )
            candidates.append((4 * vertical + horizontal, vertical, horizontal, x))
    if not candidates:
        return None
    nearest_score = min(score for score, _vertical, _horizontal, _x in candidates)
    nearest_candidates = [
        item for item in candidates if item[0] == nearest_score
    ]
    if len(nearest_candidates) != 1:
        return None
    _score, nearest_distance, _horizontal, nearest_x = nearest_candidates[0]
    # A value on a different table row cannot establish branch ownership.
    if nearest_distance > 24.0:
        return None
    return nearest_x


def _caption_segment_tokens(
    packet: GeometryPacket, value_x: float
) -> frozenset[str]:
    """Return tokens from the unique Table-caption branch above ``value_x``."""

    candidates: list[tuple[float, frozenset[str]]] = []
    for _y, words in _positioned_lines(packet.text):
        starts = [
            position for position, (_x, word) in enumerate(words)
            if normalize_text(word) in {"table", "tab"}
        ]
        for offset, start in enumerate(starts):
            left = words[start][0]
            right = (
                words[starts[offset + 1]][0]
                if offset + 1 < len(starts)
                else math.inf
            )
            if not left <= value_x < right:
                continue
            tokens = _token_signature(
                " ".join(word for _x, word in words[start:(
                    starts[offset + 1] if offset + 1 < len(starts) else len(words)
                )])
            )
            if tokens:
                candidates.append((left, tokens))
    distinct = {tokens for _left, tokens in candidates}
    return next(iter(distinct)) if len(distinct) == 1 else frozenset()


def _has_aligned_step_header(
    packet: GeometryPacket, value_x: float, required: frozenset[str]
) -> bool:
    if "step" not in required:
        return False
    for _y, words in _positioned_lines(packet.text):
        for x, word in words:
            if abs(x - value_x) <= 45.0 and normalize_text(word) in {
                "step", "steps", "nfe", "nfes"
            }:
                return True
    return False


def _with_geometry_source_branch(
    path: tuple[str, ...],
    proposal: ProposedCell,
    packet: GeometryPacket,
) -> tuple[str, ...]:
    """Restore a missing dataset parent from one geometrically bound caption.

    This is narrower than page-level token recovery.  The exact scalar must be
    nearest the exact printed row, its x-position must fall under one caption
    branch, and that caption must contain the requested dataset family without
    a conflicting sibling.  All other missing metric tokens still fail closed.
    """

    required = _token_signature(proposal.column_name)
    present = _token_signature(" ".join(path))
    missing = required - present
    selected_dataset: set[str] = set()
    for family in _EXCLUSIVE_HEADER_FAMILIES:
        selected_dataset.update(required.intersection(family))
    if not selected_dataset or not selected_dataset.intersection(missing):
        return path
    value_x = _proposal_value_position(packet, proposal)
    if value_x is None:
        return path
    caption = _caption_segment_tokens(packet, value_x)
    if not selected_dataset.issubset(caption):
        return path
    for family in _EXCLUSIVE_HEADER_FAMILIES:
        selected = required.intersection(family)
        if selected and caption.intersection(family) - selected:
            return path
    numeric_missing = {
        token for token in missing if any(character.isdigit() for character in token)
    }
    if not numeric_missing.issubset(caption):
        return path
    allowed_missing = set(selected_dataset) | numeric_missing | set(
        _OPTIONAL_RESULT_TOKENS
    )
    if "step" in missing and _has_aligned_step_header(packet, value_x, required):
        allowed_missing.add("step")
    if not missing.issubset(allowed_missing):
        return path
    return (*path, proposal.column_name)


def _resolve_owner_row(
    proposals: list[ProposedCell], target: RowTarget, packet: GeometryPacket
) -> list[ProposedCell]:
    """Bind a printed ``Ours`` row to one uniquely named source method."""

    if not proposals or len(proposals[0].printed_row_key) != 1:
        return proposals
    printed = proposals[0].printed_row_key[0].strip()
    if _OWNER_ROW_RE.fullmatch(printed) is None:
        return proposals
    source_label = nearest_printed_target(packet.text, target.expected_key)
    if source_label is None:
        return proposals
    return [
        replace(item, printed_row_key=(source_label,)) for item in proposals
    ]


@register_table_fact_producer("geometry_vision")
class GeometryVisionFactProducer:
    """Create facts only after coordinate-text and rendered-page agreement."""

    def __init__(
        self, *, max_pages: int = 2, empty_vision_retries: int = 0
    ):
        if (
            isinstance(max_pages, bool)
            or not isinstance(max_pages, int)
            or not 1 <= max_pages <= 5
        ):
            raise ValueError("max_pages must be an integer from 1 to 5")
        if (
            isinstance(empty_vision_retries, bool)
            or not isinstance(empty_vision_retries, int)
            or not 0 <= empty_vision_retries <= 2
        ):
            raise ValueError(
                "empty_vision_retries must be an integer from 0 to 2"
            )
        self._max_pages = max_pages
        self._empty_vision_retries = empty_vision_retries

    def produce(
        self,
        ctx,
        ledger: TableFactLedger,
        *,
        text_llm,
        vision_llm,
    ) -> FactProductionResult:
        current = ledger
        diagnostics: list[dict[str, Any]] = []
        for target in ledger.targets:
            resolved_columns: set[str] = set()
            packets = geometry_packets_for_target(
                ctx,
                target,
                ledger.value_columns,
                max_pages=self._max_pages,
            )
            target_report = {
                "target_id": target.target_id,
                "expected_key": list(target.expected_key),
                "packets": len(packets),
                "attempts": [],
                "accepted_facts": 0,
            }
            for packet in packets:
                attempt = {
                    "paper_id": packet.paper_id,
                    "page": packet.page,
                    "geometry_cells": 0,
                    "vision_cells": 0,
                    "agreed_cells": 0,
                }
                try:
                    response = text_llm.complete(
                        _geometry_prompt(ctx, ledger, target, packet),
                        system=_GEOMETRY_SYSTEM,
                        temperature=0.0,
                    )
                except Exception as error:  # noqa: BLE001 -- fail closed
                    attempt["outcome"] = "geometry_reader_error"
                    attempt["error_type"] = type(error).__name__
                    attempt["error"] = str(error)[:300]
                    target_report["attempts"].append(attempt)
                    continue
                attempt["geometry_response"] = _strip_fences(response)[:2000]
                geometry = _resolve_owner_row(
                    _parse_proposals(response, ledger), target, packet
                )
                attempt["geometry_cells"] = len(geometry)
                if not geometry:
                    attempt["outcome"] = "geometry_no_proposal"
                    target_report["attempts"].append(attempt)
                    continue
                pdf_bytes = (getattr(ctx, "pdf_bytes_by_id", None) or {}).get(
                    packet.paper_id
                )
                png = (
                    render_page_clip_png(
                        pdf_bytes, packet.page, packet.bbox, dpi=250
                    )
                    if pdf_bytes
                    else None
                )
                if png is None:
                    attempt["outcome"] = "render_failed"
                    target_report["attempts"].append(attempt)
                    continue
                vision = []
                vision_responses = []
                for _vision_attempt in range(self._empty_vision_retries + 1):
                    try:
                        response = vision_llm.complete(
                            _vision_prompt(
                                ctx, ledger, target, packet, geometry
                            ),
                            system=_VISION_SYSTEM,
                            temperature=0.0,
                            images=[png],
                        )
                    except Exception:  # noqa: BLE001 -- fail this packet closed
                        break
                    stripped_response = _strip_fences(response)[:2000]
                    vision_responses.append(stripped_response)
                    vision = _resolve_owner_row(
                        _parse_proposals(response, ledger), target, packet
                    )
                    if vision:
                        break
                attempt["vision_responses"] = vision_responses
                attempt["vision_response"] = (
                    vision_responses[-1] if vision_responses else ""
                )
                agreed = _agreed_proposals(geometry, vision)
                source_oriented = list(agreed)
                agreed = [
                    _resolve_transposed_header_target(item, target, packet)
                    for item in agreed
                ]
                attempt["vision_cells"] = len(vision)
                attempt["agreed_cells"] = len(agreed)
                attempt["transposed_cells"] = sum(
                    item != source
                    for item, source in zip(
                        agreed, source_oriented, strict=True
                    )
                )
                attempt["agreed_proposals"] = [
                    {
                        "printed_row_key": list(item.printed_row_key),
                        "column_name": item.column_name,
                        "header_path": list(item.header_path),
                        "raw_value": item.raw_value,
                    }
                    for item in agreed
                ]
                attempt["rejected_cells"] = []
                accepted_cells = []
                for proposal in agreed:
                    if proposal.column_name in resolved_columns:
                        continue
                    source_path = _with_row_local_step_qualifier(
                        proposal.header_path, proposal, target, packet
                    )
                    source_path = _with_geometry_source_branch(
                        source_path, proposal, packet
                    )
                    completed_path = complete_header_path(
                        source_path,
                        proposal.column_name,
                        target.header_requirements,
                        packet.text,
                    )
                    if completed_path is None:
                        attempt["rejected_cells"].append({
                            "column_name": proposal.column_name,
                            "raw_value": proposal.raw_value,
                            "reasons": ["header_path_incomplete"],
                        })
                        continue
                    fact, rejection_reasons = make_cell_fact_with_reasons(
                        current,
                        target_id=target.target_id,
                        paper_id=packet.paper_id,
                        printed_row_key=proposal.printed_row_key,
                        column_name=proposal.column_name,
                        header_path=completed_path,
                        raw_value=proposal.raw_value,
                        typed_value=proposal.typed_value,
                        page=packet.page,
                        source_type=packet.source_type,
                        object_id=packet.object_id,
                        quote=packet.text,
                        native_packet_text=packet.text,
                        verifier_families={"layout_text", "vision"},
                        required_header_terms=target.header_requirements,
                        question=ctx.question,
                        bbox=packet.bbox,
                    )
                    if fact is None:
                        attempt["rejected_cells"].append({
                            "column_name": proposal.column_name,
                            "raw_value": proposal.raw_value,
                            "reasons": list(rejection_reasons),
                        })
                        continue
                    current = current.with_fact(fact)
                    attestation = make_row_attestation(
                        current,
                        target_id=target.target_id,
                        paper_id=packet.paper_id,
                        printed_row_key=proposal.printed_row_key,
                        page=packet.page,
                        source_type=packet.source_type,
                        object_id=packet.object_id,
                        quote=packet.text,
                        source_text=packet.text,
                        bbox=packet.bbox,
                    )
                    if attestation is not None:
                        current = current.with_attestation(attestation)
                    target_report["accepted_facts"] += 1
                    accepted_cells.append(proposal.column_name)
                    resolved_columns.add(proposal.column_name)
                attempt["accepted_cells"] = accepted_cells
                if not geometry:
                    attempt["outcome"] = "geometry_no_proposal"
                elif not vision:
                    attempt["outcome"] = "vision_no_proposal"
                elif not agreed:
                    attempt["outcome"] = "cross_view_disagreement"
                elif accepted_cells:
                    attempt["outcome"] = "accepted"
                else:
                    attempt["outcome"] = "source_contract_rejected"
                target_report["attempts"].append(attempt)
                if resolved_columns == {
                    column.name for column in ledger.value_columns
                }:
                    break
            if target_report["accepted_facts"]:
                target_report["outcome"] = "accepted"
            elif not packets:
                target_report["outcome"] = "no_source_packets"
            else:
                target_report["outcome"] = "unresolved"
            diagnostics.append(target_report)
        return FactProductionResult(current, tuple(diagnostics))
