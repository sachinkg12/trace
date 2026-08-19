"""One-to-one assignment and conservative assembly for table fact ledgers.

This module is reachable only through the explicit table-fact producer option.
It converts immutable facts into scorer-shaped rows while preserving the
current answer as an explicit floor. Source rows and requested targets are
matched one-to-one, and conflicts cause abstention. Existing non-null cells
remain immutable unless the separate dual-verified replacement policy is
explicitly enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Iterable

from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_assemble import matches_expected_key
from littraceqa.answer.table_fact_ledger import (
    CellFact,
    RowAttestation,
    TableFactLedger,
)
from littraceqa.answer.table_verify import assign_rows_to_expected
from littraceqa.answer.table_value_contract import replacement_preserves_information


@dataclass(frozen=True)
class FactAssemblyResult:
    rows: tuple[dict, ...]
    selected_facts: tuple[CellFact, ...]
    added_target_ids: tuple[str, ...]
    retry_target_ids: tuple[str, ...]
    changed_cells: tuple[tuple[str, str], ...]
    replaced_cells: tuple[tuple[str, str], ...]
    unresolved: tuple[str, ...]
    assignment_kinds: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class _SourceRow:
    source_key: tuple[Any, ...]
    printed_row_key: tuple[str, ...]
    target_ids: frozenset[str]
    attestations: tuple[RowAttestation, ...]
    facts: tuple[CellFact, ...]

    @property
    def strength(self) -> tuple[int, int]:
        return (
            max((len(fact.verifier_families) for fact in self.facts), default=0),
            len(self.facts),
        )


def _item_source_key(item: RowAttestation | CellFact) -> tuple[Any, ...]:
    # A visible object id already identifies the table/figure; bbox is useful
    # only when that id is missing.  This keeps an object-level row attestation
    # and its cell-level bbox fact in the same physical source-row group.
    bbox = (
        tuple(round(value, 3) for value in item.bbox)
        if item.bbox and not item.object_id
        else ()
    )
    return (*item.source_row_key, bbox)


def _source_rows(ledger: TableFactLedger) -> list[_SourceRow]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in [*ledger.attestations, *ledger.facts]:
        key = _item_source_key(item)
        slot = grouped.setdefault(key, {
            "printed_row_key": item.printed_row_key,
            "target_ids": set(),
            "attestations": [],
            "facts": [],
        })
        slot["target_ids"].add(item.target_id)
        if isinstance(item, CellFact):
            slot["facts"].append(item)
        else:
            slot["attestations"].append(item)
    rows = [
        _SourceRow(
            source_key=key,
            printed_row_key=slot["printed_row_key"],
            target_ids=frozenset(slot["target_ids"]),
            attestations=tuple(slot["attestations"]),
            facts=tuple(slot["facts"]),
        )
        for key, slot in grouped.items()
    ]
    # Stronger independently verified rows reserve exact target slots first.
    # The lexical suffix makes the result independent of insertion order.
    rows.sort(key=lambda item: (-item.strength[0], -item.strength[1], item.source_key))
    return rows


def _as_row(key: tuple[str, ...], columns: tuple[str, ...]) -> dict:
    return {column: key[position] for position, column in enumerate(columns)}


def _value_key(value: Any) -> tuple[str, Any] | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return ("number", float(value))
    if isinstance(value, str):
        normalized = normalize_text(value)
        return ("string", normalized) if normalized else None
    return None


def _fact_sort_key(fact: CellFact) -> tuple[Any, ...]:
    return (
        -len(fact.verifier_families),
        fact.paper_id,
        fact.page,
        fact.object_id or "",
        tuple(normalize_text(value) for value in fact.printed_row_key),
        fact.header_path,
        repr(fact.typed_value),
    )


def _select_fact(
    facts: Iterable[CellFact],
    *,
    minimum_verifier_families: int,
    value_policy: str,
) -> tuple[CellFact | None, object | None, bool]:
    eligible = sorted(
        (
            fact for fact in facts
            if len(fact.verifier_families) >= minimum_verifier_families
        ),
        key=_fact_sort_key,
    )
    by_value: dict[tuple[str, Any], list[CellFact]] = {}
    for fact in eligible:
        value = _fact_value(fact, value_policy)
        key = _value_key(value)
        if key is not None:
            by_value.setdefault(key, []).append(fact)
    if len(by_value) != 1:
        return None, None, len(by_value) > 1
    only_value = next(iter(by_value.values()))
    fact = only_value[0]
    return fact, _fact_value(fact, value_policy), False


def _fact_value(fact: CellFact, policy: str) -> object:
    if policy == "source":
        return fact.typed_value
    if policy not in {"schema_canonical", "header_unit_explicit"}:
        raise ValueError(
            "cell value policy must be 'source', 'schema_canonical', or "
            "'header_unit_explicit'"
        )
    if policy == "header_unit_explicit":
        if fact.unit == "%" and fact.unit_origin in {"header", "column"}:
            for candidate in fact.scorer_candidates:
                if isinstance(candidate, str) and candidate.strip().endswith("%"):
                    return candidate
        return fact.typed_value
    if fact.unit == "%" and fact.unit_origin in {"header", "column"}:
        for candidate in fact.scorer_candidates:
            if isinstance(candidate, str) and "%" not in candidate:
                return candidate
    return fact.typed_value


def _normalized_key(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(normalize_text(value) for value in values)


def _match_quality(
    row_key: tuple[str, ...], target, columns: tuple[str, ...]
) -> tuple[int, str] | None:
    """Rank schema identity above exact aliases above tolerant matches."""

    if _normalized_key(row_key) == _normalized_key(target.expected_key):
        return 10_000_000, "schema_exact"
    if any(
        _normalized_key(row_key) == _normalized_key(alias)
        for alias in target.aliases
    ):
        return 100_000, "alias_exact"
    if matches_expected_key(row_key, target.expected_key, columns):
        return 1_000, "schema_tolerant"
    if any(
        matches_expected_key(row_key, alias, columns)
        for alias in target.aliases
    ):
        return 10, "alias_tolerant"
    return None


def _assign_rows_to_targets(
    rows: list[dict],
    ledger: TableFactLedger,
    *,
    allowed_target_ids: list[frozenset[str]] | None = None,
) -> tuple[dict[int, int], dict[int, str]]:
    """Maximum-quality, identity-preserving one-to-one target assignment.

    Typical benchmark tables have fewer than ten requested rows, so an exact
    bitmask dynamic program is both simpler and safer than a new dependency.
    A deterministic legacy fallback handles pathological schemas with more
    than sixteen target slots.
    """

    targets = ledger.targets
    row_keys = [
        tuple(
            "" if row.get(column) is None else str(row.get(column))
            for column in ledger.row_key_cols
        )
        for row in rows
    ]
    if len(targets) > 16:
        legacy = assign_rows_to_expected(
            rows,
            [target.expected_key for target in targets],
            ledger.row_key_cols,
        )
        if allowed_target_ids is not None:
            legacy = {
                row: target
                for row, target in legacy.items()
                if targets[target].target_id in allowed_target_ids[row]
            }
        kinds = {
            row: (_match_quality(row_keys[row], targets[target], ledger.row_key_cols)
                  or (0, "legacy"))[1]
            for row, target in legacy.items()
        }
        return legacy, kinds

    edges: list[list[tuple[int, int, str]]] = []
    for row_position, row_key in enumerate(row_keys):
        row_edges = []
        for target_position, target in enumerate(targets):
            if (
                allowed_target_ids is not None
                and target.target_id not in allowed_target_ids[row_position]
            ):
                continue
            quality = _match_quality(row_key, target, ledger.row_key_cols)
            if quality is not None:
                row_edges.append((target_position, *quality))
        edges.append(row_edges)

    @lru_cache(maxsize=None)
    def solve(row_position: int, used_mask: int):
        if row_position >= len(rows):
            return 0, 0, ()
        best = solve(row_position + 1, used_mask)
        for target_position, score, kind in edges[row_position]:
            bit = 1 << target_position
            if used_mask & bit:
                continue
            count, total, pairs = solve(row_position + 1, used_mask | bit)
            candidate = (
                count + 1,
                total + score,
                ((row_position, target_position, kind), *pairs),
            )
            candidate_objective = (candidate[1], candidate[0])
            best_objective = (best[1], best[0])
            if (
                candidate_objective > best_objective
                or (
                    candidate_objective == best_objective
                    and candidate[2] < best[2]
                )
            ):
                best = candidate
        return best

    _count, _score, pairs = solve(0, 0)
    return (
        {row: target for row, target, _kind in pairs},
        {row: kind for row, _target, kind in pairs},
    )


def _assign_source_rows(
    ledger: TableFactLedger, source_rows: list[_SourceRow]
) -> tuple[dict[int, int], dict[int, str]]:
    rows = [_as_row(item.printed_row_key, ledger.row_key_cols) for item in source_rows]
    return _assign_rows_to_targets(
        rows,
        ledger,
        allowed_target_ids=[item.target_ids for item in source_rows],
    )


def assemble_table_facts(
    ledger: TableFactLedger,
    frozen_rows: Iterable[dict],
    *,
    allow_row_additions: bool = False,
    preserve_unmatched_frozen: bool = True,
    allow_cell_replacements: bool = False,
    canonicalize_attestation_only_rows: bool = False,
    minimum_verifier_families: int = 2,
    cell_value_policy: str = "source",
) -> FactAssemblyResult:
    """Assemble agreed facts under explicit row and replacement policies.

    ``allow_row_additions`` and ``preserve_unmatched_frozen`` are explicit
    experiment controls.  Their conservative defaults leave row cardinality
    unchanged. ``allow_cell_replacements`` is also off by default and still
    requires the same unique value plus at least two verifier families.  The
    opt-in canonicalization applies only to a row supported by an attestation
    but no accepted cell fact; source rows carrying values retain their
    physical printed identity.
    """

    frozen = [dict(row) for row in frozen_rows]
    if minimum_verifier_families < 2:
        raise ValueError("table facts require at least two verifier families")
    if cell_value_policy not in {
        "source", "schema_canonical", "header_unit_explicit"
    }:
        raise ValueError(
            "cell value policy must be 'source', 'schema_canonical', or "
            "'header_unit_explicit'"
        )
    if not ledger.targets:
        return FactAssemblyResult(
            rows=tuple(frozen), selected_facts=(), added_target_ids=(),
            retry_target_ids=(), changed_cells=(), replaced_cells=(),
            unresolved=(), assignment_kinds=(),
        )
    source_rows = _source_rows(ledger)
    source_assignment, source_assignment_kinds = _assign_source_rows(
        ledger, source_rows
    )
    target_to_source = {
        target_position: source_position
        for source_position, target_position in source_assignment.items()
    }

    frozen_assignment, frozen_assignment_kinds = _assign_rows_to_targets(
        frozen, ledger
    )
    if preserve_unmatched_frozen:
        rows = frozen
        old_to_new = {position: position for position in range(len(frozen))}
    else:
        kept = sorted(frozen_assignment)
        rows = [frozen[position] for position in kept]
        old_to_new = {old: new for new, old in enumerate(kept)}
    target_to_row = {
        target_position: old_to_new[row_position]
        for row_position, target_position in frozen_assignment.items()
        if row_position in old_to_new
    }

    selected: list[CellFact] = []
    added: list[str] = []
    retry: list[str] = []
    changed: list[tuple[str, str]] = []
    replaced: list[tuple[str, str]] = []
    unresolved: list[str] = []

    for target_position, target in enumerate(ledger.targets):
        source_position = target_to_source.get(target_position)
        if source_position is None:
            continue
        source_row = source_rows[source_position]
        target_attested = any(
            item.target_id == target.target_id for item in source_row.attestations
        ) or any(item.target_id == target.target_id for item in source_row.facts)
        row_addition_attested = any(
            item.target_id == target.target_id
            for item in source_row.attestations
        ) or any(
            item.target_id == target.target_id
            and "owner_target_assignment" not in item.verifier_families
            for item in source_row.facts
        )
        source_target_facts = [
            fact for fact in source_row.facts
            if fact.target_id == target.target_id
        ]
        row_position = target_to_row.get(target_position)
        if row_position is None and allow_row_additions and row_addition_attested:
            emitted_key = (
                target.expected_key
                if canonicalize_attestation_only_rows
                and not source_target_facts
                else source_row.printed_row_key
            )
            row = _as_row(emitted_key, ledger.row_key_cols)
            row.update({column.name: None for column in ledger.value_columns})
            rows.append(row)
            row_position = len(rows) - 1
            target_to_row[target_position] = row_position
            added.append(target.target_id)

        has_missing_or_conflict = False
        all_target_facts = [
            fact for fact in ledger.facts if fact.target_id == target.target_id
        ]
        for column in ledger.value_columns:
            _, _, conflict = _select_fact(
                (
                    fact
                    for fact in all_target_facts
                    if fact.column_name == column.name
                ),
                minimum_verifier_families=minimum_verifier_families,
                value_policy=cell_value_policy,
            )
            if conflict:
                unresolved.append(f"{target.target_id}:{column.name}:conflict")
                has_missing_or_conflict = True
                continue
            fact, emitted_value, _ = _select_fact(
                (
                    fact
                    for fact in source_target_facts
                    if fact.column_name == column.name
                ),
                minimum_verifier_families=minimum_verifier_families,
                value_policy=cell_value_policy,
            )
            if fact is None:
                has_missing_or_conflict = True
                continue
            if row_position is None:
                # A cell cannot create a scorer row unless the separately
                # controlled row-addition policy admitted its source identity.
                has_missing_or_conflict = True
                continue
            if rows[row_position].get(column.name) is None:
                rows[row_position][column.name] = emitted_value
                selected.append(fact)
                changed.append((target.target_id, column.name))
            elif _value_key(rows[row_position].get(column.name)) != _value_key(
                emitted_value
            ):
                if allow_cell_replacements:
                    incumbent = rows[row_position].get(column.name)
                    if replacement_preserves_information(
                        incumbent,
                        emitted_value,
                        column_type=column.value_type,
                    ):
                        rows[row_position][column.name] = emitted_value
                        selected.append(fact)
                        changed.append((target.target_id, column.name))
                        replaced.append((target.target_id, column.name))
                    else:
                        unresolved.append(
                            f"{target.target_id}:{column.name}:"
                            "incumbent_information_loss"
                        )
                        has_missing_or_conflict = True
                else:
                    unresolved.append(
                        f"{target.target_id}:{column.name}:incumbent"
                    )
                    has_missing_or_conflict = True
        if target_attested and has_missing_or_conflict:
            retry.append(target.target_id)

    # A fact can claim a target yet lose the one-to-one source-row assignment.
    # Surface that as a diagnostic rather than silently treating it as absent.
    assigned_targets = {
        ledger.targets[position].target_id for position in target_to_source
    }
    claimed_targets = {
        item.target_id for item in [*ledger.attestations, *ledger.facts]
    }
    for target_id in sorted(claimed_targets - assigned_targets):
        unresolved.append(f"{target_id}:row_assignment")

    return FactAssemblyResult(
        rows=tuple(rows),
        selected_facts=tuple(sorted(selected, key=_fact_sort_key)),
        added_target_ids=tuple(added),
        retry_target_ids=tuple(dict.fromkeys(retry)),
        changed_cells=tuple(changed),
        replaced_cells=tuple(replaced),
        unresolved=tuple(sorted(set(unresolved))),
        assignment_kinds=tuple(sorted([
            *(
                (
                    "source",
                    ledger.targets[target_position].target_id,
                    source_assignment_kinds[source_position],
                )
                for source_position, target_position in source_assignment.items()
            ),
            *(
                (
                    "frozen",
                    ledger.targets[target_position].target_id,
                    frozen_assignment_kinds[row_position],
                )
                for row_position, target_position in frozen_assignment.items()
            ),
        ])),
    )
