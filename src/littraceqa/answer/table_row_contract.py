"""Conservative row-key de-hedging for frozen table answers.

Some table generators hedge an uncertain row identity by emitting several
aliases with exactly the same value cells.  The official scorer turns those
aliases into distinct predicted rows, so even a correct value is accompanied
by avoidable false positives.  This module collapses such rows only when the
question plan and the frozen value structure define the same cardinality and
a unique one-to-one assignment.

The operation is deliberately row-only: it never invents, deletes, or changes
a value-cell payload.  Any ambiguity returns the input byte-for-byte.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any, Iterable, Sequence

from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_assemble import matches_expected_key


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on",
    "paper", "the", "to", "used", "using", "with",
})
_GENERIC_WRAPPER_RE = re.compile(
    r"\s*\((?:our|ours|proposed)\s*(?:approach|method|model)?\)\s*$",
    re.IGNORECASE,
)
_OPERATIONAL_QUALIFIER_RE = re.compile(
    r"\([^)]*(?:\d|epoch|iter|nfe|parameter|shot|step|train)[^)]*\)",
    re.IGNORECASE,
)
_QUESTION_IDENTITY_COLUMNS = frozenset({
    "attribute", "attributes", "metric", "metrics", "quantity", "quantities",
})


@dataclass(frozen=True)
class RowContractResult:
    rows: tuple[dict[str, Any], ...]
    changed: bool
    reason: str
    input_rows: int
    output_rows: int
    value_groups: int
    assignments: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()


def _value_key(value: Any) -> tuple[str, Any]:
    if value is None:
        return ("none", "")
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return ("number", float(value))
    return ("text", normalize_text(value))


def _value_signature(
    row: dict[str, Any], value_columns: Sequence[str]
) -> tuple[tuple[str, Any], ...]:
    return tuple(_value_key(row.get(column)) for column in value_columns)


def _row_key(
    row: dict[str, Any], row_key_columns: Sequence[str]
) -> tuple[str, ...]:
    return tuple(str(row.get(column) or "").strip() for column in row_key_columns)


def _tokens(values: Iterable[Any]) -> frozenset[str]:
    return frozenset(
        token
        for value in values
        for token in _TOKEN_RE.findall(str(value or "").casefold())
        if token not in _STOPWORDS
    )


def _match_score(
    aliases: Sequence[tuple[str, ...]],
    expected: tuple[str, ...],
    row_key_columns: Sequence[str],
) -> int:
    expected_normalized = tuple(normalize_text(value) for value in expected)
    best = 0
    expected_tokens = _tokens(expected)
    for alias in aliases:
        normalized = tuple(normalize_text(value) for value in alias)
        if normalized == expected_normalized:
            best = max(best, 10_000)
            continue
        if matches_expected_key(alias, expected, list(row_key_columns)):
            best = max(best, 1_000)
        alias_tokens = _tokens(alias)
        if expected_tokens and alias_tokens:
            overlap = len(expected_tokens.intersection(alias_tokens))
            union = len(expected_tokens.union(alias_tokens))
            if overlap:
                best = max(best, 100 + int(100 * overlap / union))
    return best


def _unique_assignment(
    groups: Sequence[Sequence[tuple[str, ...]]],
    expected_keys: Sequence[tuple[str, ...]],
    row_key_columns: Sequence[str],
) -> tuple[int, ...] | None:
    """Return target->group positions only for one strict best assignment."""

    scores = [
        [
            _match_score(group, expected, row_key_columns)
            for group in groups
        ]
        for expected in expected_keys
    ]
    if any(not row or max(row) < 100 for row in scores):
        return None

    @lru_cache(maxsize=None)
    def solve(target: int, used: int) -> tuple[int, int, tuple[int, ...]]:
        if target == len(expected_keys):
            return (0, 1, ())
        best_score = -1
        best_count = 0
        best_assignment: tuple[int, ...] = ()
        for group in range(len(groups)):
            if used & (1 << group) or scores[target][group] < 100:
                continue
            suffix_score, suffix_count, suffix = solve(
                target + 1, used | (1 << group)
            )
            if suffix_count == 0:
                continue
            total = scores[target][group] + suffix_score
            candidate = (group, *suffix)
            if total > best_score:
                best_score = total
                best_count = suffix_count
                best_assignment = candidate
            elif total == best_score:
                best_count += suffix_count
                if not best_assignment or candidate < best_assignment:
                    best_assignment = candidate
        return (best_score, best_count, best_assignment)

    _score, count, assignment = solve(0, 0)
    return assignment if count == 1 and len(assignment) == len(expected_keys) else None


def _emitted_key(
    aliases: Sequence[tuple[str, ...]],
    expected: tuple[str, ...],
    row_key_columns: Sequence[str],
) -> tuple[str, ...]:
    """Choose question identity unless the printed label carries identity.

    Metric/quantity tables use the question phrase as the scorer row. Method
    tables normally retain the printed/source label, including a one-character
    correction or an identity-bearing qualifier. The one exception is a
    generic ``(Ours)`` wrapper: removing it recovers a concise method explicitly
    named in the question (for example ``Method-X``).
    """

    expected_normalized = tuple(normalize_text(value) for value in expected)
    for alias in aliases:
        if tuple(normalize_text(value) for value in alias) == expected_normalized:
            return alias
    column = normalize_text(row_key_columns[0]) if row_key_columns else ""
    if column in _QUESTION_IDENTITY_COLUMNS:
        return expected
    if len(expected) == 1 and not _OPERATIONAL_QUALIFIER_RE.search(expected[0]):
        for alias in aliases:
            base = _GENERIC_WRAPPER_RE.sub("", alias[0]).strip()
            if normalize_text(base) == normalize_text(expected[0]):
                return expected
    tolerant = [
        alias
        for alias in aliases
        if matches_expected_key(alias, expected, list(row_key_columns))
    ]
    if tolerant:
        return min(
            tolerant,
            key=lambda alias: (sum(len(value) for value in alias), alias),
        )
    return expected


def collapse_hedged_rows(
    frozen_rows: Iterable[dict[str, Any]],
    *,
    row_key_columns: Sequence[str],
    value_columns: Sequence[str],
    expected_keys: Sequence[Sequence[Any]],
) -> RowContractResult:
    """Collapse duplicate-value aliases under a strict cardinality contract.

    The transformation activates only for a single row-key column, at least
    two expected rows, more frozen rows than expected rows, and exactly one
    non-null value signature per expected row.  Each expected row must have a
    unique one-to-one match to a signature group.  Otherwise the frozen rows
    are returned unchanged.
    """

    frozen = tuple(dict(row) for row in frozen_rows)

    def unchanged(reason: str, groups: int = 0) -> RowContractResult:
        return RowContractResult(
            rows=frozen,
            changed=False,
            reason=reason,
            input_rows=len(frozen),
            output_rows=len(frozen),
            value_groups=groups,
        )

    row_columns = tuple(str(column) for column in row_key_columns if column)
    values = tuple(str(column) for column in value_columns if column)
    expected = tuple(
        tuple(str(value or "").strip() for value in key)
        for key in expected_keys
    )
    if len(row_columns) != 1:
        return unchanged("requires_single_row_key")
    if not values:
        return unchanged("requires_value_columns")
    if len(expected) < 2 or any(len(key) != 1 or not key[0] for key in expected):
        return unchanged("requires_explicit_expected_keys")
    if len(frozen) <= len(expected):
        return unchanged("not_overcomplete")

    grouped: dict[tuple[tuple[str, Any], ...], list[dict[str, Any]]] = {}
    for row in frozen:
        signature = _value_signature(row, values)
        if not any(kind != "none" for kind, _value in signature):
            return unchanged("all_null_value_group")
        grouped.setdefault(signature, []).append(row)
    if len(grouped) != len(expected):
        return unchanged("cardinality_mismatch", len(grouped))
    groups = tuple(grouped.values())
    alias_groups = tuple(
        tuple(_row_key(row, row_columns) for row in group) for group in groups
    )
    assignment = _unique_assignment(alias_groups, expected, row_columns)
    if assignment is None:
        return unchanged("ambiguous_assignment", len(groups))

    out: list[dict[str, Any]] = []
    assignments: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for expected_key, group_position in zip(expected, assignment, strict=True):
        representative = dict(groups[group_position][0])
        # Only row-key fields change. Every value and auxiliary field comes
        # from an existing frozen row in the selected identical-value group.
        emitted_key = _emitted_key(
            alias_groups[group_position], expected_key, row_columns
        )
        representative[row_columns[0]] = emitted_key[0]
        out.append(representative)
        assignments.append((expected_key, alias_groups[group_position][0]))
    return RowContractResult(
        rows=tuple(out),
        changed=tuple(out) != frozen,
        reason="collapsed" if tuple(out) != frozen else "already_canonical",
        input_rows=len(frozen),
        output_rows=len(out),
        value_groups=len(groups),
        assignments=tuple(assignments),
    )
