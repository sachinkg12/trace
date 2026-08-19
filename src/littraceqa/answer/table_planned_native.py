"""Conservative two-stage planned-table answerer.

The first stage is the ordinary schema-planned answerer.  The second stage
uses strict native-text/native-table consensus only to fill null value cells
in those exact rows.  It is deliberately not a general replay strategy:
external frozen rows, row additions, replacements, tolerant row assignment,
and key canonicalization cannot authorize output changes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from littraceqa.answer.interfaces import (
    AnswerContext,
    StrategyOutput,
    register_strategy,
)
from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_fact_assign import (
    FactAssemblyResult,
    assemble_table_facts,
)
from littraceqa.answer.table_fact_ledger import (
    TableFactLedger,
    build_table_fact_ledger,
)
from littraceqa.answer.table_fact_native import (
    NativeTextTableConsensusFactProducer,
)
from littraceqa.answer.table_plan import TablePlan
from littraceqa.answer.table_planned import SchemaPlannedTableAnswerer


_EXACT_ASSIGNMENT_KINDS = frozenset({"schema_exact", "alias_exact"})


@dataclass(frozen=True)
class _NativeProposal:
    rows: tuple[dict, ...]
    assembly: FactAssemblyResult
    target_rows: Mapping[str, int]
    diagnostics: tuple[dict[str, Any], ...]


def _normalized_key(row: Mapping[str, Any], columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(normalize_text(row.get(column)) for column in columns)


def _target_keys(target) -> frozenset[tuple[str, ...]]:
    return frozenset(
        tuple(normalize_text(value) for value in key)
        for key in (target.expected_key, *target.aliases)
    )


def _strict_null_ledger(
    plan: TablePlan,
    ctx: AnswerContext,
    rows: list[dict],
) -> tuple[TableFactLedger, dict[str, int]] | None:
    """Bind each null-bearing base row to one exact ledger target.

    The scorer-normalized row keys must be complete and globally unique.
    Every null-bearing row must then have exactly one normalized exact
    expected-key or alias match, and a target may be used at most once.
    """

    row_key_cols = tuple(plan.row_key_cols)
    value_cols = tuple(str(column["name"]) for column in plan.value_cols)
    if not row_key_cols or not value_cols or not rows:
        return None
    normalized_rows: list[tuple[str, ...]] = []
    null_rows: list[int] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        if any(column not in row for column in row_key_cols):
            return None
        key = _normalized_key(row, row_key_cols)
        if any(not value for value in key):
            return None
        normalized_rows.append(key)
        if any(column not in row or row[column] is None for column in value_cols):
            null_rows.append(position)
    if len(normalized_rows) != len(set(normalized_rows)) or not null_rows:
        return None

    ledger = build_table_fact_ledger(plan, ctx)
    selected = []
    target_rows: dict[str, int] = {}
    used_target_ids: set[str] = set()
    for row_position in null_rows:
        key = normalized_rows[row_position]
        matches = [
            target
            for target in ledger.targets
            if key in _target_keys(target)
        ]
        if len(matches) != 1 or matches[0].target_id in used_target_ids:
            return None
        target = matches[0]
        used_target_ids.add(target.target_id)
        selected.append(target)
        target_rows[target.target_id] = row_position
    return replace(ledger, targets=tuple(selected)), target_rows


def _assignment_proves_changes(
    assembly: FactAssemblyResult,
    target_rows: Mapping[str, int],
) -> bool:
    """Require independent exact source and frozen assignment per change."""

    if assembly.added_target_ids or assembly.replaced_cells:
        return False
    changed = tuple(assembly.changed_cells)
    if not changed or len(changed) != len(set(changed)):
        return False
    changed_targets = {target_id for target_id, _column in changed}
    if not changed_targets.issubset(target_rows):
        return False
    facts: dict[tuple[str, str], list[Any]] = {}
    for fact in assembly.selected_facts:
        facts.setdefault((fact.target_id, fact.column_name), []).append(fact)
    for cell in changed:
        selected = facts.get(cell, [])
        if len(selected) != 1 or not {
            "native_text_route", "native_table_route"
        }.issubset(selected[0].verifier_families):
            return False
    if set(facts) != set(changed):
        return False
    assignments: dict[tuple[str, str], list[str]] = {}
    for route, target_id, kind in assembly.assignment_kinds:
        assignments.setdefault((route, target_id), []).append(kind)
    for target_id in changed_targets:
        for route in ("source", "frozen"):
            kinds = assignments.get((route, target_id), [])
            if len(kinds) != 1 or kinds[0] not in _EXACT_ASSIGNMENT_KINDS:
                return False
    return True


def _same_paper_routes_prove_changes(
    assembly: FactAssemblyResult,
    diagnostics: Iterable[Mapping[str, Any]],
) -> bool:
    """Bind each changed fact to one unambiguous same-paper route pair."""

    facts: dict[tuple[str, str], list[Any]] = {}
    for fact in assembly.selected_facts:
        facts.setdefault((fact.target_id, fact.column_name), []).append(fact)
    consensus = [
        item
        for item in diagnostics
        if isinstance(item, Mapping)
        and item.get("producer") == "native_text_table_consensus"
        and item.get("strict_cross_route_same_paper") is True
    ]
    for cell in assembly.changed_cells:
        target_id, column = cell
        selected = facts.get(cell, [])
        if len(selected) != 1:
            return False
        records = [item for item in consensus if item.get("target_id") == target_id]
        if len(records) != 1:
            return False
        agreements = [
            item
            for item in (records[0].get("agreements") or [])
            if item.get("column_name") == column
        ]
        if len(agreements) != 1:
            return False
        agreement = agreements[0]
        text_sources = agreement.get("text_sources")
        table_sources = agreement.get("table_sources")
        if (
            agreement.get("policy") != "cross_route_value_agreement"
            or not isinstance(text_sources, list)
            or not isinstance(table_sources, list)
            or len(text_sources) != 1
            or len(table_sources) != 1
            or text_sources[0] != agreement.get("text_source")
            or table_sources[0] != agreement.get("table_source")
            or not isinstance(text_sources[0], list)
            or not isinstance(table_sources[0], list)
            or len(text_sources[0]) != 3
            or len(table_sources[0]) != 3
            or not str(text_sources[0][0] or "").strip()
            or text_sources[0][0] != table_sources[0][0]
            or selected[0].paper_id != text_sources[0][0]
        ):
            return False
    return True


def _accepted_null_fills(
    base_rows: list[dict],
    proposal: _NativeProposal,
    plan: TablePlan,
) -> tuple[list[dict], list[dict[str, Any]]] | None:
    """Apply a positional output firewall around the common assembler."""

    if (
        not _assignment_proves_changes(proposal.assembly, proposal.target_rows)
        or not _same_paper_routes_prove_changes(
            proposal.assembly, proposal.diagnostics
        )
    ):
        return None
    candidate_rows = list(proposal.rows)
    if len(candidate_rows) != len(base_rows):
        return None
    value_cols = {str(column["name"]) for column in plan.value_cols}
    changed_by_cell: dict[tuple[int, str], str] = {}
    for target_id, column in proposal.assembly.changed_cells:
        if column not in value_cols:
            return None
        cell = (proposal.target_rows[target_id], column)
        if cell in changed_by_cell:
            return None
        changed_by_cell[cell] = target_id

    actual_fills: list[dict[str, Any]] = []
    output: list[dict] = []
    for row_position, (base_row, candidate_row) in enumerate(
        zip(base_rows, candidate_rows, strict=True)
    ):
        if not isinstance(candidate_row, dict):
            return None
        base_keys = list(base_row)
        candidate_keys = list(candidate_row)
        if candidate_keys[:len(base_keys)] != base_keys:
            return None
        added_keys = candidate_keys[len(base_keys):]
        if any(column not in value_cols for column in added_keys):
            return None
        for key, base_value in base_row.items():
            if key not in candidate_row:
                return None
            candidate_value = candidate_row[key]
            if base_value is not None and (
                type(candidate_value) is not type(base_value)
                or candidate_value != base_value
            ):
                return None
            if base_value is None and key not in value_cols and candidate_value is not None:
                return None
        for column in value_cols:
            base_value = base_row.get(column)
            candidate_value = candidate_row.get(column)
            cell = (row_position, column)
            if base_value is None and candidate_value is not None:
                if cell not in changed_by_cell:
                    return None
                actual_fills.append({"row_index": row_position, "column": column})
            elif cell in changed_by_cell:
                return None
        output.append(candidate_row)
    if len(actual_fills) != len(changed_by_cell) or not actual_fills:
        return None
    return output, actual_fills


@register_strategy("table_planned_native")
class PlannedNativeNullFillTableAnswerer:
    """Ordinary planned answer followed by strict internal consensus fills."""

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
        visual_consensus_repeats: int = 1,
        extraction_sources: str = "both",
        text_context_mode: str = "evidence",
        text_page_k: int = 3,
        fill_nulls_from_scalar_evidence: bool = False,
        trace_reassembly_inputs: bool = False,
        native_max_pages: int = 2,
        native_empty_vision_retries: int = 0,
    ):
        self._base = SchemaPlannedTableAnswerer(
            retain_concise_missing_expected_rows=retain_concise_missing_expected_rows,
            route_expected_rows_to_papers=route_expected_rows_to_papers,
            open_ended_one_row_per_paper=open_ended_one_row_per_paper,
            prefer_owned_cells=prefer_owned_cells,
            retain_source_attested_expected_rows=retain_source_attested_expected_rows,
            visual_extraction_mode=visual_extraction_mode,
            visual_retry_owned_rows=visual_retry_owned_rows,
            visual_consensus_repeats=visual_consensus_repeats,
            extraction_sources=extraction_sources,
            text_context_mode=text_context_mode,
            text_page_k=text_page_k,
            fill_nulls_from_scalar_evidence=fill_nulls_from_scalar_evidence,
            trace_reassembly_inputs=trace_reassembly_inputs,
        )
        self._producer = NativeTextTableConsensusFactProducer(
            max_pages=native_max_pages,
            empty_vision_retries=native_empty_vision_retries,
        )

    def _run_native(
        self,
        ctx: AnswerContext,
        plan: TablePlan,
        rows: list[dict],
    ) -> _NativeProposal | None:
        strict = _strict_null_ledger(plan, ctx, rows)
        if strict is None:
            return None
        ledger, target_rows = strict
        production = self._producer.produce(
            ctx,
            ledger,
            text_llm=ctx.llm,
            vision_llm=ctx.vision_llm or ctx.llm,
        )
        assembly = assemble_table_facts(
            production.ledger,
            rows,
            allow_row_additions=False,
            preserve_unmatched_frozen=True,
            allow_cell_replacements=False,
            canonicalize_attestation_only_rows=False,
            minimum_verifier_families=2,
            cell_value_policy="source",
        )
        return _NativeProposal(
            rows=assembly.rows,
            assembly=assembly,
            target_rows=target_rows,
            diagnostics=production.diagnostics,
        )

    def answer(self, ctx: AnswerContext) -> StrategyOutput:
        # External replay rows never enter either stage of this strategy.
        base_ctx = replace(ctx, frozen_table_rows=None)
        base, plan = self._base.answer_with_plan(base_ctx)
        if ctx.frozen_table_rows is not None or plan is None:
            return base
        if not isinstance(base.value, list) or not all(
            isinstance(row, dict) for row in base.value
        ):
            return base
        base_rows = base.value
        value_cols = tuple(str(column["name"]) for column in plan.value_cols)
        if not value_cols or not any(
            column not in row or row[column] is None
            for row in base_rows
            for column in value_cols
        ):
            return base
        try:
            # The producer receives neither the first-stage list nor the
            # pristine firewall snapshot.  A buggy producer can mutate its
            # private working copy and raise without corrupting the base
            # StrategyOutput returned by this fail-safe wrapper.
            pristine_rows = deepcopy(base_rows)
            working_rows = deepcopy(pristine_rows)
            internal_ctx = replace(base_ctx, frozen_table_rows=working_rows)
            proposal = self._run_native(internal_ctx, plan, working_rows)
            if proposal is None:
                return base
            accepted = _accepted_null_fills(pristine_rows, proposal, plan)
            if accepted is None:
                return base
            rows, filled_cells = accepted
        except Exception:  # noqa: BLE001 -- exact first-stage answer is the floor
            return base
        return StrategyOutput(
            value=rows,
            confidence=base.confidence,
            attested_evidence=base.attested_evidence,
            diagnostics={
                "status": "planned_native_null_filled",
                "base": base.diagnostics,
                "filled_cells": filled_cells,
                "producer": "native_text_table_consensus",
                "fact_production": list(proposal.diagnostics),
                "assignment_kinds": [
                    list(item) for item in proposal.assembly.assignment_kinds
                ],
            },
        )
