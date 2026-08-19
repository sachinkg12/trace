"""Source-type-dispatched facts from immutable native PDF blocks.

The geometry producer is intentionally specialized for printed tables.  It
should not be coerced into reading prose, equations, or bibliography entries.
The opt-in ``native_text`` producer uses the deterministic
:mod:`source_ledger` to select a small number of physically grounded
non-visual blocks.  The separately registered ``native_table`` producer reads
only deterministic native table objects, so it can be gated without changing
the geometry path.  Both ask one semantic reader to map a block to the
requested schema, then admit a value only when the independent native-source
contract proves the row label, header terms, and scalar verbatim.

``SourceDispatchFactProducer`` composes this producer after the existing
geometry+vision producer.  Conflicting values remain conflicts in the common
``CellFact`` ledger and therefore fail closed during assembly.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import threading
from typing import Any, Iterable

from littraceqa.answer.table_fact_extract import (
    FactProductionResult,
    GeometryVisionFactProducer,
    _parse_printed_row_key,
    _parse_proposals,
    _strip_fences,
    register_table_fact_producer,
)
from littraceqa.answer.table_fact_ledger import (
    CellFact,
    RowTarget,
    TableFactLedger,
    _token_signature,
    complete_header_path,
    make_cell_fact_with_reasons,
    make_row_attestation,
)
from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_assemble import matches_expected_key
from littraceqa.answer.table_verify import target_present
from littraceqa.localize.source_ledger import (
    SourceBlock,
    SourceLedgerBuildError,
    build_source_ledger_builder,
)


_NATIVE_TEXT_SOURCE_TYPES = frozenset({
    "text_span", "equation_algorithm", "citation_context"
})
_NATIVE_TABLE_SOURCE_TYPES = frozenset({"table"})
_NATIVE_TARGET_SOURCE_TYPES = frozenset({
    *_NATIVE_TEXT_SOURCE_TYPES,
    *_NATIVE_TABLE_SOURCE_TYPES,
})
_MAX_NATIVE_PACKET_CHARS = 12_000
_MAX_NATIVE_PACKETS_PER_TARGET = 4
_SOURCE_LEDGER_CACHE_SIZE = 64
_SOURCE_LEDGER_CACHE: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
_SOURCE_LEDGER_CACHE_LOCK = threading.Lock()

_NATIVE_SYSTEM = (
    "You extract source facts from ONE immutable academic-PDF text block. "
    "The requested row identity may be proven in the separate "
    "ROW_IDENTITY_SOURCE; the schema meaning and answer value must be printed "
    "in IMMUTABLE_PDF_BLOCK. Copy the proven row label and raw value "
    "verbatim. header_path must contain only printed phrases from the block "
    "that jointly express the schema column and every requested qualifier. "
    "Never calculate, paraphrase, complete an equation, or use outside "
    "knowledge. Respond with STRICT minified JSON only: "
    '{"printed_row_key":["..."],"cells":[{"column_name":"exact schema '
    'name","header_path":["printed phrase"],"raw_value":"verbatim"}]}. '
    "Return an empty cells list on ambiguity."
)

_TARGETED_NATIVE_SYSTEM = (
    "You extract one value for an already-existing frozen answer row from ONE "
    "immutable academic-PDF source block. In owner_attributed mode, the "
    "pipeline has already assigned REQUESTED_ROW to exactly one selected "
    "paper and exactly one frozen row; copy REQUESTED_ROW as printed_row_key. "
    "The value and complete header_path must still be printed verbatim in "
    "IMMUTABLE_PDF_BLOCK. Never add a row, calculate, paraphrase, or use "
    "outside knowledge. Respond with STRICT minified JSON only: "
    "{\"printed_row_key\":[\"...\"],\"cells\":[{\"column_name\":\"exact schema "
    "name\",\"header_path\":[\"printed phrase\"],\"raw_value\":\"verbatim\"}]}. "
    "Return an empty cells list on ambiguity."
)

_ATTRIBUTION_GENERIC_TOKENS = frozenset({
    "accuracy", "answer", "attribute", "detail", "method", "model",
    "paper", "quantity", "result", "score", "setting", "value",
})


def _cached_source_ledger(builder, paper_id: str, pdf_bytes: bytes):
    """Build one immutable ledger per PDF snapshot and parser configuration.

    Composed fact producers previously reparsed the same PDF two or three
    times in one question. Native table detection is the dominant CPU cost and
    caused otherwise-valid table records to hit the question timeout. The
    exact PDF hash and builder configuration make reuse byte-safe; the bounded
    LRU prevents a long 71-question run from retaining the whole corpus.
    """

    snapshot = bytes(pdf_bytes)
    builder_config = json.dumps({
        "include_native_tables": getattr(builder, "_include_native_tables", None),
    }, sort_keys=True, separators=(",", ":"))
    key = (
        f"{type(builder).__module__}.{type(builder).__qualname__}:{builder_config}",
        paper_id,
        hashlib.sha256(snapshot).hexdigest(),
    )
    with _SOURCE_LEDGER_CACHE_LOCK:
        cached = _SOURCE_LEDGER_CACHE.get(key)
        if cached is not None:
            _SOURCE_LEDGER_CACHE.move_to_end(key)
            return cached
    built = builder.build(paper_id, snapshot)
    with _SOURCE_LEDGER_CACHE_LOCK:
        existing = _SOURCE_LEDGER_CACHE.get(key)
        if existing is not None:
            _SOURCE_LEDGER_CACHE.move_to_end(key)
            return existing
        _SOURCE_LEDGER_CACHE[key] = built
        while len(_SOURCE_LEDGER_CACHE) > _SOURCE_LEDGER_CACHE_SIZE:
            _SOURCE_LEDGER_CACHE.popitem(last=False)
    return built


@dataclass(frozen=True)
class NativeFactPacket:
    paper_id: str
    page: int
    source_type: str
    object_id: str | None
    bbox: tuple[float, float, float, float] | None
    text: str
    identity_text: str
    score: int
    identity_mode: str = "printed"
    merged_blocks: int = 1


def _join_packet_texts(values: Iterable[str], *, limit: int) -> str:
    """Join distinct immutable blocks without exceeding the reader budget.

    PyMuPDF commonly emits several overlapping blocks for one physical page.
    Sending those blocks as separate semantic-reader attempts spends the
    target-wide packet budget on duplicate locators.  Exact duplicates are
    removed and additional blocks are kept whole so a truncation cannot turn
    a printed scalar into a misleading fragment.
    """
    separator = "\n\n--- SAME PHYSICAL PAGE BLOCK ---\n\n"
    output: list[str] = []
    used = 0
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value or value in output:
            continue
        extra = len(value) + (len(separator) if output else 0)
        if output and used + extra > limit:
            continue
        if not output and len(value) > limit:
            value = value[:limit]
            extra = len(value)
        output.append(value)
        used += extra
    return separator.join(output)


def _merge_same_locator_packets(
    ranked: Iterable[NativeFactPacket],
) -> list[NativeFactPacket]:
    """Collapse overlapping blocks that resolve to one physical locator.

    Source type and object ID remain part of the key: blocks from different
    scorer-facing objects are never relabelled as one another.  The highest
    ranked block supplies the locator metadata and score; its same-locator
    siblings only widen the immutable text available to the reader.
    """
    grouped: dict[
        tuple[str, int, str, str | None, str], list[NativeFactPacket]
    ] = {}
    for packet in ranked:
        key = (
            packet.paper_id,
            packet.page,
            packet.source_type,
            packet.object_id,
            packet.identity_mode,
        )
        grouped.setdefault(key, []).append(packet)

    output: list[NativeFactPacket] = []
    for packets in grouped.values():
        lead = packets[0]
        output.append(NativeFactPacket(
            paper_id=lead.paper_id,
            page=lead.page,
            source_type=lead.source_type,
            object_id=lead.object_id,
            # A union of multiple block rectangles is not a trustworthy
            # scorer-facing box.  Keep the exact box only for a singleton.
            bbox=lead.bbox if len(packets) == 1 else None,
            text=_join_packet_texts(
                (packet.text for packet in packets),
                limit=_MAX_NATIVE_PACKET_CHARS,
            ),
            identity_text=_join_packet_texts(
                (packet.identity_text for packet in packets),
                limit=_MAX_NATIVE_PACKET_CHARS,
            ),
            score=lead.score,
            identity_mode=lead.identity_mode,
            merged_blocks=sum(packet.merged_blocks for packet in packets),
        ))
    output.sort(key=lambda item: (
        -item.score,
        item.paper_id,
        item.page,
        item.source_type,
        item.object_id or "",
        item.text,
    ))
    return output


def _visible_alias(target: RowTarget, text: str) -> bool:
    return any(target_present(text, alias) for alias in target.aliases)


def _locator_keys(ctx, paper_id: str) -> set[tuple[int, str]]:
    return {
        (int(item.page), str(item.source_type))
        for item in (getattr(ctx, "evidence", None) or [])
        if item.paper_id == paper_id
        and isinstance(item.page, int)
        and not isinstance(item.page, bool)
    }


def _packet_score(
    ctx,
    block: SourceBlock,
    column_names: Iterable[str],
    *,
    source_types: frozenset[str] = _NATIVE_TEXT_SOURCE_TYPES,
) -> int:
    if block.source_type not in source_types:
        return -1
    source_tokens = _token_signature(block.text)
    column_tokens = _token_signature(" ".join(column_names))
    question_tokens = _token_signature(
        str(getattr(ctx, "question", "") or "")
    )
    column_overlap = len(source_tokens.intersection(column_tokens))
    question_overlap = len(source_tokens.intersection(question_tokens))
    locator_bonus = 1000 if (
        block.physical_page, block.source_type
    ) in _locator_keys(ctx, block.paper_id) else 0
    if not locator_bonus and not column_overlap and not question_overlap:
        return -1
    typed_bonus = {
        "equation_algorithm": 80,
        "citation_context": 60,
        "text_span": 20,
    }.get(block.source_type, 0)
    return (
        locator_bonus
        + typed_bonus
        + 50 * column_overlap
        + 2 * question_overlap
    )


def _target_terms(target: RowTarget) -> frozenset[str]:
    """Return non-generic terms that distinguish one requested row.

    These terms are retrieval features only. They never become scorer-facing
    aliases and cannot authorize a new row.
    """

    values = [
        *target.expected_key,
        *(value for alias in target.aliases for value in alias),
        *target.qualifiers,
        *target.claim.header_requirements,
        *target.claim.source_requirements,
    ]
    return frozenset(
        token
        for token in _token_signature(" ".join(values))
        if token not in _ATTRIBUTION_GENERIC_TOKENS
    )


def _unique_frozen_target_row(ctx, ledger, target: RowTarget) -> bool:
    """Whether this target maps to exactly one pre-existing output row.

    Owner-attributed extraction may repair cells but must never manufacture a
    scorer row. Requiring one existing row also prevents a broad paper-level
    match from smearing one scalar across two hedged spellings.
    """

    rows = [
        row
        for row in (getattr(ctx, "frozen_table_rows", None) or [])
        if isinstance(row, dict)
    ]
    matching = []
    for row in rows:
        key = tuple(
            "" if row.get(column) is None else str(row.get(column))
            for column in ledger.row_key_cols
        )
        if any(
            matches_expected_key(key, alias, ledger.row_key_cols)
            or matches_expected_key(alias, key, ledger.row_key_cols)
            for alias in target.aliases
        ):
            matching.append(key)
    return len(matching) == 1


def _attributed_packet_score(
    ctx,
    block: SourceBlock,
    target: RowTarget,
    column_names: Iterable[str],
    *,
    source_types: frozenset[str],
) -> int:
    """Rank a block for a uniquely owned existing row without label leakage."""

    if block.source_type not in source_types:
        return -1
    target_terms = _target_terms(target)
    source_terms = _token_signature(block.text)
    overlap = source_terms.intersection(target_terms)
    # One generic coincidence such as ``training`` or ``accuracy`` is not an
    # assignment. Two independently printed target terms are the minimum.
    if len(overlap) < 2:
        return -1
    base = _packet_score(
        ctx, block, column_names, source_types=source_types
    )
    if base < 0:
        return -1
    requirement_tokens = _token_signature(" ".join(
        (*target.header_requirements, *target.claim.source_requirements)
    ))
    requirement_overlap = len(source_terms.intersection(requirement_tokens))
    return base + 200 + 60 * len(overlap) + 100 * requirement_overlap


def _native_prompt(ctx, ledger, target, packet: NativeFactPacket) -> str:
    return "\n".join((
        f"QUESTION: {ctx.question}",
        f"ROW_IDENTITY_MODE: {packet.identity_mode}",
        "ROW_KEY_COLUMNS: " + json.dumps(
            list(ledger.row_key_cols), ensure_ascii=False
        ),
        "REQUESTED_ROW: " + json.dumps(
            list(target.expected_key), ensure_ascii=False
        ),
        "SAFE_ROW_ALIASES: " + json.dumps(
            [list(alias) for alias in target.aliases], ensure_ascii=False
        ),
        "ROW_IDENTITY_SOURCE (may differ from the value block):",
        packet.identity_text[:2000],
        "ROW_QUALIFIERS: " + json.dumps(
            list(target.qualifiers), ensure_ascii=False
        ),
        "ATOMIC_CLAIM_TUPLE: " + json.dumps(
            asdict(target.claim), ensure_ascii=False
        ),
        "VALUE_COLUMNS: " + json.dumps([
            {"name": column.name, "type": column.value_type}
            for column in ledger.value_columns
        ], ensure_ascii=False),
        f"SOURCE_TYPE: {packet.source_type}",
        f"OBJECT_ID: {packet.object_id or ''}",
        f"PHYSICAL_PAGE: {packet.page}",
        "IMMUTABLE_PDF_BLOCK:",
        packet.text[:_MAX_NATIVE_PACKET_CHARS],
    ))


def _native_packets(
    ctx,
    target: RowTarget,
    ledger: TableFactLedger,
    source_ledgers,
    *,
    max_pages: int,
    source_types: frozenset[str] = _NATIVE_TEXT_SOURCE_TYPES,
    allow_attributed_identity: bool = False,
) -> list[NativeFactPacket]:
    ranked: list[NativeFactPacket] = []
    column_names = [column.name for column in ledger.value_columns]
    for paper_id in target.owner_papers:
        source_ledger = source_ledgers.get(paper_id)
        if source_ledger is None:
            continue
        identity_blocks = [
            block.text
            for block in source_ledger.blocks
            if _visible_alias(target, block.text)
        ]
        title = str(
            (getattr(ctx, "paper_titles", None) or {}).get(paper_id) or ""
        )
        if title and _visible_alias(target, title):
            identity_blocks.insert(0, title)
        attributed_identity = (
            not identity_blocks
            and allow_attributed_identity
            and len(target.owner_papers) == 1
            and _unique_frozen_target_row(ctx, ledger, target)
            and len(_target_terms(target)) >= 2
        )
        if not identity_blocks and not attributed_identity:
            continue
        identity_text = (
            "\n".join(dict.fromkeys(identity_blocks[:2]))
            if identity_blocks
            else title
        )
        for block in source_ledger.blocks:
            score = (
                _attributed_packet_score(
                    ctx,
                    block,
                    target,
                    column_names,
                    source_types=source_types,
                )
                if attributed_identity
                else _packet_score(
                    ctx, block, column_names, source_types=source_types
                )
            )
            if score < 0:
                continue
            ranked.append(NativeFactPacket(
                paper_id=paper_id,
                page=block.physical_page,
                source_type=block.source_type,
                object_id=block.object_id,
                bbox=block.bbox,
                text=block.text,
                identity_text=identity_text,
                score=score,
                identity_mode=(
                    "owner_attributed" if attributed_identity else "printed"
                ),
            ))
    ranked.sort(key=lambda item: (
        -item.score,
        item.paper_id,
        item.page,
        item.source_type,
        item.object_id or "",
        item.text,
    ))
    # One semantic-reader attempt should correspond to one exact physical
    # locator, not one parser block.  Without this collapse, four overlapping
    # text blocks from the first relevant page exhaust the global target
    # budget and later pages are never inspected.  Distinct printed objects
    # on the same page remain independent packets and retain their provenance.
    ranked = _merge_same_locator_packets(ranked)
    output: list[NativeFactPacket] = []
    pages_by_paper: dict[str, set[int]] = {}
    for packet in ranked:
        pages = pages_by_paper.setdefault(packet.paper_id, set())
        if packet.page not in pages and len(pages) >= max_pages:
            continue
        pages.add(packet.page)
        output.append(packet)
        if len(output) >= _MAX_NATIVE_PACKETS_PER_TARGET:
            break
    return output


@register_table_fact_producer("native_text")
class NativeTextFactProducer:
    """Create prose/equation/reference facts under an exact source contract."""

    _producer_name = "native_text"
    _source_types = _NATIVE_TEXT_SOURCE_TYPES
    _allow_attributed_identity = False
    _system_prompt = _NATIVE_SYSTEM
    # Row-only attestation is deliberately disabled for established producers.
    # A separately registered experiment enables it for native table objects.
    _attest_row_only = False

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
        # The composition root already exposes this bounded retry control.  In
        # the non-visual producer it means semantic-reader retries; no image is
        # sent and the same immutable native block is reused.
        self._reader_retries = empty_vision_retries

    def produce(
        self,
        ctx,
        ledger: TableFactLedger,
        *,
        text_llm,
        vision_llm,
    ) -> FactProductionResult:
        del vision_llm
        current = ledger
        builder = build_source_ledger_builder("pymupdf")
        source_ledgers: dict[str, Any] = {}
        build_errors: dict[str, str] = {}
        for paper_id in dict.fromkeys(
            paper
            for target in ledger.targets
            for paper in target.owner_papers
        ):
            pdf_bytes = (getattr(ctx, "pdf_bytes_by_id", None) or {}).get(
                paper_id
            )
            if not pdf_bytes:
                build_errors[paper_id] = "pdf_unavailable"
                continue
            try:
                source_ledgers[paper_id] = _cached_source_ledger(
                    builder, paper_id, pdf_bytes
                )
            except SourceLedgerBuildError:
                build_errors[paper_id] = "source_ledger_error"

        diagnostics: list[dict[str, Any]] = []
        for target in ledger.targets:
            resolved_columns = {
                fact.column_name
                for fact in current.facts
                if fact.target_id == target.target_id
                and len(fact.verifier_families) >= 2
            }
            report: dict[str, Any] = {
                "producer": self._producer_name,
                "target_id": target.target_id,
                "expected_key": list(target.expected_key),
                "packets": 0,
                "attempts": [],
                "accepted_facts": 0,
                "accepted_attestations": 0,
                "build_errors": {
                    paper: build_errors[paper]
                    for paper in target.owner_papers
                    if paper in build_errors
                },
            }
            if resolved_columns == {
                column.name for column in ledger.value_columns
            }:
                report["outcome"] = "already_resolved"
                diagnostics.append(report)
                continue
            packets = _native_packets(
                ctx,
                target,
                current,
                source_ledgers,
                max_pages=self._max_pages,
                source_types=self._source_types,
                allow_attributed_identity=self._allow_attributed_identity,
            )
            report["packets"] = len(packets)
            for packet in packets:
                attempt: dict[str, Any] = {
                    "paper_id": packet.paper_id,
                    "page": packet.page,
                    "source_type": packet.source_type,
                    "object_id": packet.object_id,
                    "identity_mode": packet.identity_mode,
                    "merged_blocks": packet.merged_blocks,
                    "proposal_cells": 0,
                    "accepted_cells": [],
                    "rejected_cells": [],
                }
                proposals = []
                responses = []
                parsed_row_keys: list[tuple[str, ...]] = []
                for _reader_attempt in range(self._reader_retries + 1):
                    try:
                        response = text_llm.complete(
                            _native_prompt(ctx, current, target, packet),
                            system=self._system_prompt,
                            temperature=0.0,
                        )
                    except Exception as error:  # noqa: BLE001 -- fail closed
                        attempt["outcome"] = "semantic_reader_error"
                        attempt["error_type"] = type(error).__name__
                        attempt["error"] = str(error)[:300]
                        break
                    stripped = _strip_fences(response)[:2000]
                    responses.append(stripped)
                    if self._attest_row_only:
                        try:
                            payload = json.loads(stripped)
                        except (json.JSONDecodeError, TypeError):
                            payload = None
                        parsed_key = _parse_printed_row_key(payload, current)
                        if parsed_key is not None:
                            parsed_row_keys.append(parsed_key)
                    proposals = _parse_proposals(response, current)
                    if packet.identity_mode == "owner_attributed":
                        proposals = [
                            replace(
                                proposal,
                                printed_row_key=target.expected_key,
                            )
                            for proposal in proposals
                        ]
                    if proposals:
                        break
                attempt["responses"] = responses
                attempt["proposal_cells"] = len(proposals)
                if not proposals:
                    # A null-valued row is still scorer-relevant, but only the
                    # opt-in table-row producer may retain it.  Require every
                    # non-empty reader claim to agree and require the complete
                    # printed key in this exact physical table object.  The
                    # separate identity context is intentionally *not* enough.
                    distinct_keys = list(dict.fromkeys(parsed_row_keys))
                    # A budget qualifier can be identity-bearing even when a
                    # physical source row prints only the short method name.
                    # That row proves neither which budget was requested nor
                    # the complete scorer-facing identity, so row-only
                    # additions with budget requirements fail closed.
                    exact_source_key_safe = not any(
                        "budget" in normalize_text(requirement)
                        for requirement in target.header_requirements
                    )
                    owner_match = (
                        not target.owner_papers
                        or packet.paper_id in target.owner_papers
                    )
                    attempt["target_owner_papers"] = list(
                        target.owner_papers
                    )
                    attempt["target_header_requirements"] = list(
                        target.header_requirements
                    )
                    attempt["exact_source_key_safe"] = exact_source_key_safe
                    attempt["owner_match"] = owner_match
                    attestation = (
                        make_row_attestation(
                            current,
                            target_id=target.target_id,
                            paper_id=packet.paper_id,
                            printed_row_key=target.expected_key,
                            page=packet.page,
                            source_type=packet.source_type,
                            object_id=packet.object_id,
                            quote=packet.text,
                            source_text=packet.text,
                            identity_source_text=None,
                            bbox=packet.bbox,
                        )
                        if self._attest_row_only and exact_source_key_safe
                        else None
                    )
                    if (
                        attestation is None
                        and self._attest_row_only
                        and exact_source_key_safe
                        and len(distinct_keys) == 1
                    ):
                        attestation = make_row_attestation(
                            current,
                            target_id=target.target_id,
                            paper_id=packet.paper_id,
                            printed_row_key=distinct_keys[0],
                            page=packet.page,
                            source_type=packet.source_type,
                            object_id=packet.object_id,
                            quote=packet.text,
                            source_text=packet.text,
                            identity_source_text=None,
                            bbox=packet.bbox,
                        )
                    if attestation is not None:
                        current = current.with_attestation(attestation)
                        report["accepted_attestations"] += 1
                        attempt["accepted_row_key"] = list(
                            attestation.printed_row_key
                        )
                        attempt["outcome"] = (
                            "row_attested_exact_source"
                            if attestation.printed_row_key == target.expected_key
                            else "row_attested"
                        )
                    else:
                        attempt.setdefault("outcome", "semantic_no_proposal")
                    report["attempts"].append(attempt)
                    continue
                for proposal in proposals:
                    if proposal.column_name in resolved_columns:
                        continue
                    completed_path = complete_header_path(
                        proposal.header_path,
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
                    fact, reasons = make_cell_fact_with_reasons(
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
                        verifier_families={
                            "semantic_reader",
                            "native_pdf",
                            *(
                                {"owner_target_assignment"}
                                if packet.identity_mode == "owner_attributed"
                                else set()
                            ),
                        },
                        row_identity_source_text=packet.identity_text,
                        allow_attributed_row_identity=(
                            packet.identity_mode == "owner_attributed"
                        ),
                        required_header_terms=target.header_requirements,
                        question=ctx.question,
                        bbox=packet.bbox,
                    )
                    if fact is None:
                        attempt["rejected_cells"].append({
                            "column_name": proposal.column_name,
                            "raw_value": proposal.raw_value,
                            "reasons": list(reasons),
                        })
                        continue
                    current = current.with_fact(fact)
                    attestation = (
                        make_row_attestation(
                            current,
                            target_id=target.target_id,
                            paper_id=packet.paper_id,
                            printed_row_key=proposal.printed_row_key,
                            page=packet.page,
                            source_type=packet.source_type,
                            object_id=packet.object_id,
                            quote=packet.text,
                            source_text=packet.text,
                            identity_source_text=packet.identity_text,
                            bbox=packet.bbox,
                        )
                        if packet.identity_mode == "printed"
                        else None
                    )
                    if attestation is not None:
                        current = current.with_attestation(attestation)
                    resolved_columns.add(proposal.column_name)
                    report["accepted_facts"] += 1
                    attempt["accepted_cells"].append(proposal.column_name)
                attempt["outcome"] = (
                    "accepted"
                    if attempt["accepted_cells"]
                    else "source_contract_rejected"
                )
                report["attempts"].append(attempt)
                if resolved_columns == {
                    column.name for column in ledger.value_columns
                }:
                    break
            if report["accepted_facts"] or report["accepted_attestations"]:
                report["outcome"] = "accepted"
            elif not packets:
                report["outcome"] = "no_source_packets"
            else:
                report["outcome"] = "unresolved"
            diagnostics.append(report)
        return FactProductionResult(current, tuple(diagnostics))


@register_table_fact_producer("native_targeted_text")
class NativeTargetedTextFactProducer(NativeTextFactProducer):
    """Repair an existing frozen row through unique owner/target attribution.

    The ordinary native producer requires the complete scorer-facing row label
    to be printed in the PDF. Scientific papers often print only the setting
    and value because the paper itself supplies the method identity. This
    opt-in producer admits that common case only when one selected paper and
    one existing frozen row are uniquely assigned, at least two target terms
    occur in the immutable block, and every header/value source check passes.
    It never creates a row attestation, so it cannot add an output row.
    """

    _producer_name = "native_targeted_text"
    _source_types = _NATIVE_TARGET_SOURCE_TYPES
    _allow_attributed_identity = True
    _system_prompt = _TARGETED_NATIVE_SYSTEM


@register_table_fact_producer("native_table")
class NativeTableFactProducer(NativeTextFactProducer):
    """Read deterministic native table objects under the CellFact contract.

    This producer is intentionally separate from ``native_text`` and the
    geometry+vision path.  Experiments can therefore measure whether native
    table objects add value without silently changing either established
    producer.
    """

    _producer_name = "native_table"
    _source_types = _NATIVE_TABLE_SOURCE_TYPES


@register_table_fact_producer("native_table_rows")
class NativeTableRowFactProducer(NativeTableFactProducer):
    """Retain exact PDF-table row identities even when all cells abstain.

    The policy is separately named so ``native_table`` remains byte-for-byte
    compatible with its frozen gate.  Assembly still controls whether an
    attested null row may be added.
    """

    _producer_name = "native_table_rows"
    _attest_row_only = True


@register_table_fact_producer("native_text_table_rows")
class NativeTextTableRowFactProducer:
    """Compose proven native-text cells with source-attested table rows."""

    def __init__(
        self, *, max_pages: int = 2, empty_vision_retries: int = 0
    ):
        self._native_text = NativeTextFactProducer(
            max_pages=max_pages,
            empty_vision_retries=empty_vision_retries,
        )
        self._native_rows = NativeTableRowFactProducer(
            max_pages=max_pages,
            empty_vision_retries=empty_vision_retries,
        )
        self._max_pages = max_pages

    def produce(
        self,
        ctx,
        ledger: TableFactLedger,
        *,
        text_llm,
        vision_llm,
    ) -> FactProductionResult:
        native = self._native_text.produce(
            ctx,
            ledger,
            text_llm=text_llm,
            vision_llm=vision_llm,
        )
        rows = self._native_rows.produce(
            ctx,
            native.ledger,
            text_llm=text_llm,
            vision_llm=vision_llm,
        )
        return FactProductionResult(
            rows.ledger,
            (*native.diagnostics, *rows.diagnostics),
        )


def _consensus_value_key(value: Any) -> tuple[str, Any] | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return ("number", numeric) if math.isfinite(numeric) else None
    normalized = normalize_text(value)
    return ("string", normalized) if normalized else None


def _unique_route_fact(
    facts: Iterable[CellFact], target_id: str, column_name: str
) -> tuple[
    tuple[str, Any] | None,
    CellFact | None,
    list[str],
    tuple[CellFact, ...],
]:
    matching = [
        fact
        for fact in facts
        if fact.target_id == target_id and fact.column_name == column_name
    ]
    grouped: dict[tuple[str, Any], list[CellFact]] = {}
    for fact in matching:
        key = _consensus_value_key(fact.typed_value)
        if key is not None:
            grouped.setdefault(key, []).append(fact)
    visible = [repr(key[1]) for key in sorted(grouped, key=repr)]
    if len(grouped) != 1:
        return None, None, visible, ()
    key, agreed = next(iter(grouped.items()))
    agreed.sort(key=lambda item: (
        item.paper_id,
        item.page,
        item.object_id or "",
        item.source_type,
        item.raw_value,
    ))
    return key, agreed[0], visible, tuple(agreed)


@register_table_fact_producer("native_text_table_consensus")
class NativeTextTableConsensusFactProducer:
    """Require independent prose and table-object agreement for cell facts.

    The older composition passed the text producer's populated ledger into the
    table producer.  That made a text fact terminal: the table route reported
    ``already_resolved`` and never corroborated it.  Here both routes start
    from the same immutable input ledger.  Only a unique normalized value
    emitted by *both* routes is admitted; disagreements and one-sided facts
    fail closed.  Exact table-row attestations remain available for separately
    controlled null-row retention.
    """

    def __init__(
        self,
        *,
        max_pages: int = 2,
        empty_vision_retries: int = 0,
    ):
        self._native_text = NativeTextFactProducer(
            max_pages=max_pages,
            empty_vision_retries=empty_vision_retries,
        )
        self._native_rows = NativeTableRowFactProducer(
            max_pages=max_pages,
            empty_vision_retries=empty_vision_retries,
        )
        self._max_pages = max_pages

    def produce(
        self,
        ctx,
        ledger: TableFactLedger,
        *,
        text_llm,
        vision_llm,
    ) -> FactProductionResult:
        text = self._native_text.produce(
            ctx,
            ledger,
            text_llm=text_llm,
            vision_llm=vision_llm,
        )
        table = self._native_rows.produce(
            ctx,
            ledger,
            text_llm=text_llm,
            vision_llm=vision_llm,
        )
        combined = ledger
        for attestation in table.ledger.attestations:
            combined = combined.with_attestation(attestation)

        consensus_diagnostics: list[dict[str, Any]] = []
        for target in ledger.targets:
            agreements: list[dict[str, Any]] = []
            disagreements: list[dict[str, Any]] = []
            for column in ledger.value_columns:
                text_key, text_fact, text_values, text_facts = _unique_route_fact(
                    text.ledger.facts, target.target_id, column.name
                )
                table_key, table_fact, table_values, table_facts = _unique_route_fact(
                    table.ledger.facts, target.target_id, column.name
                )
                strict_pair = (
                    len(text_facts) == 1
                    and len(table_facts) == 1
                    and text_facts[0].paper_id == table_facts[0].paper_id
                )
                if (
                    text_key is None
                    or table_key is None
                    or text_fact is None
                    or table_fact is None
                    or text_key != table_key
                    or not strict_pair
                ):
                    disagreements.append({
                        "column_name": column.name,
                        "text_values": text_values,
                        "table_values": table_values,
                        "reason": (
                            "ambiguous_or_cross_paper_routes"
                            if (
                                text_key is not None
                                and table_key is not None
                                and text_key == table_key
                                and not strict_pair
                            )
                            else
                            "cross_route_disagreement"
                            if text_key is not None and table_key is not None
                            else "missing_independent_route"
                        ),
                    })
                    continue
                consensus_fact = replace(
                    table_fact,
                    verifier_families=frozenset({
                        *text_fact.verifier_families,
                        *table_fact.verifier_families,
                        "native_text_route",
                        "native_table_route",
                    }),
                )
                combined = combined.with_fact(consensus_fact)
                agreements.append({
                    "column_name": column.name,
                    "value_key": list(text_key),
                    "policy": "cross_route_value_agreement",
                    "text_source": [
                        text_fact.paper_id, text_fact.page,
                        text_fact.object_id,
                    ],
                    "table_source": [
                        table_fact.paper_id, table_fact.page,
                        table_fact.object_id,
                    ],
                    "text_sources": [
                        [item.paper_id, item.page, item.object_id]
                        for item in text_facts
                    ],
                    "table_sources": [
                        [item.paper_id, item.page, item.object_id]
                        for item in table_facts
                    ],
                })
            attestation_count = sum(
                item.target_id == target.target_id
                for item in table.ledger.attestations
            )
            consensus_diagnostics.append({
                "producer": "native_text_table_consensus",
                "strict_cross_route_same_paper": True,
                "target_id": target.target_id,
                "expected_key": list(target.expected_key),
                "packets": 0,
                "attempts": [],
                "accepted_facts": len(agreements),
                "accepted_attestations": attestation_count,
                "agreements": agreements,
                "disagreements": disagreements,
                "build_errors": {},
                "outcome": (
                    "accepted"
                    if agreements
                    else "row_attested"
                    if attestation_count
                    else "unresolved"
                ),
            })
        return FactProductionResult(
            combined,
            (*text.diagnostics, *table.diagnostics, *consensus_diagnostics),
        )


@register_table_fact_producer("native_consensus_targeted")
class NativeConsensusTargetedFactProducer:
    """Run strict cross-route consensus, then target unresolved frozen cells."""

    def __init__(
        self, *, max_pages: int = 2, empty_vision_retries: int = 0
    ):
        self._consensus = NativeTextTableConsensusFactProducer(
            max_pages=max_pages,
            empty_vision_retries=empty_vision_retries,
        )
        self._targeted = NativeTargetedTextFactProducer(
            max_pages=max_pages,
            empty_vision_retries=empty_vision_retries,
        )
        self._max_pages = max_pages

    def produce(
        self,
        ctx,
        ledger: TableFactLedger,
        *,
        text_llm,
        vision_llm,
    ) -> FactProductionResult:
        consensus = self._consensus.produce(
            ctx,
            ledger,
            text_llm=text_llm,
            vision_llm=vision_llm,
        )
        targeted = self._targeted.produce(
            ctx,
            consensus.ledger,
            text_llm=text_llm,
            vision_llm=vision_llm,
        )
        return FactProductionResult(
            targeted.ledger,
            (*consensus.diagnostics, *targeted.diagnostics),
        )


@register_table_fact_producer("source_dispatch")
class SourceDispatchFactProducer:
    """Run visual table extraction, then non-visual native extraction."""

    def __init__(
        self, *, max_pages: int = 2, empty_vision_retries: int = 0
    ):
        self._geometry = GeometryVisionFactProducer(
            max_pages=max_pages,
            empty_vision_retries=empty_vision_retries,
        )
        self._native = NativeTextFactProducer(
            max_pages=max_pages,
            empty_vision_retries=empty_vision_retries,
        )
        self._max_pages = max_pages

    def produce(
        self,
        ctx,
        ledger: TableFactLedger,
        *,
        text_llm,
        vision_llm,
    ) -> FactProductionResult:
        visual = self._geometry.produce(
            ctx,
            ledger,
            text_llm=text_llm,
            vision_llm=vision_llm,
        )
        native = self._native.produce(
            ctx,
            visual.ledger,
            text_llm=text_llm,
            vision_llm=vision_llm,
        )
        visual_diagnostics = tuple({
            **item, "producer": "geometry_vision"
        } for item in visual.diagnostics)
        return FactProductionResult(
            native.ledger,
            (*visual_diagnostics, *native.diagnostics),
        )
