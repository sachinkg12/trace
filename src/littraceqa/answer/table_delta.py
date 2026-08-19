"""Source-grounded, add-only review of a frozen table draft.

The reviewer is deliberately asymmetric: it may identify a row absent from a
validator-approved frozen answer, but it cannot replace or delete any frozen
row or cell. A deterministic alias veto removes shortened/paraphrased versions
of rows already present before assembly pays a row-precision cost.
"""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_extract import (
    _evidence_line,
    _focused_paper_lines,
    _parse_rows,
)
from littraceqa.answer.table_assemble import matches_expected_key


_SYSTEM = (
    "You audit a CURRENT table answer against source pages from its already-"
    "selected academic papers. Return ONLY genuinely missing rows. Never "
    "repeat, rename, shorten, paraphrase, split, or merge a current row. Never "
    "rewrite a current cell. A new row is allowed only when the source pages "
    "explicitly state both its row identity and at least one requested value. "
    "Respond with STRICT minified JSON only: {\"rows\":[{...}]}. Use exactly "
    "the supplied schema column names and copy source wording/precision. Return "
    "{\"rows\":[]} when the current answer already covers the source claims."
)


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value)))


def _key_alias(candidate: dict, current: dict, row_key_cols: Sequence[str]) -> bool:
    for column in row_key_cols:
        if _compact(candidate.get(column)) == _compact(current.get(column)):
            continue
        left = _tokens(candidate.get(column))
        right = _tokens(current.get(column))
        if not left or not right:
            return False
        overlap = len(left & right) / len(left | right)
        subset = min(len(left), len(right)) >= 2 and (left <= right or right <= left)
        if overlap < 0.72 and not subset:
            return False
    return True


def _distinctive_value_alias(
    candidate: dict,
    current: dict,
    value_cols: Sequence[str],
) -> bool:
    for column in value_cols:
        left = _compact(candidate.get(column))
        right = _compact(current.get(column))
        # Long text/equation/prompt strings are identity-bearing. Numeric-like
        # scores are intentionally excluded because two real methods can tie.
        if len(left) >= 8 and left == right and not left.isdigit():
            return True
    return False


def _is_alias(
    candidate: dict,
    current_rows: Sequence[dict],
    row_key_cols: Sequence[str],
    value_cols: Sequence[str],
) -> bool:
    candidate_key = tuple(normalize_text(candidate.get(c)) for c in row_key_cols)
    for current in current_rows:
        current_key = tuple(normalize_text(current.get(c)) for c in row_key_cols)
        if candidate_key == current_key:
            return True
        if _key_alias(candidate, current, row_key_cols):
            return True
        if _distinctive_value_alias(candidate, current, value_cols):
            return True
    return False


def _distinct_row_key_count(rows: Sequence[dict], row_key_cols: Sequence[str]) -> int:
    """Count scorer-distinct, non-empty row keys in a frozen table."""
    keys = {
        tuple(normalize_text(row.get(column)) for column in row_key_cols)
        for row in rows
    }
    return sum(any(value for value in key) for key in keys)


def review_frozen_table_additions(ctx, plan, llm, *, page_k: int = 3):
    """Return ``(additions, diagnostics)`` without mutating frozen rows."""
    current_rows = [dict(row) for row in (ctx.frozen_table_rows or [])]
    if not current_rows:
        return [], {"status": "no_frozen_rows", "proposed": 0, "accepted": 0}

    missing_expected: list[tuple[str, ...]] = []
    addition_budget: int | None = None
    if plan.expected_keys:
        for expected in plan.expected_keys:
            covered = any(
                matches_expected_key(
                    tuple(row.get(column) for column in plan.row_key_cols),
                    expected,
                    plan.row_key_cols,
                )
                for row in current_rows
            )
            if not covered:
                missing_expected.append(tuple(expected))
        if not missing_expected:
            return [], {
                "status": "all_expected_covered",
                "proposed": 0,
                "accepted": 0,
                "missing_expected": [],
            }
        # Expected-key matching is intentionally tolerant, but a planner can
        # still phrase the same semantic row differently from the frozen/source
        # label. In an add-only reviewer that false mismatch must not increase
        # cardinality beyond the number of rows the question explicitly asks
        # for. Otherwise a condition (for example a difficulty endpoint) can be
        # appended as a new row beside the already-present requested quantity.
        # Category/open-ended questions have no expected keys and are unaffected.
        addition_budget = max(
            0,
            len(plan.expected_keys)
            - _distinct_row_key_count(current_rows, plan.row_key_cols),
        )
        if addition_budget == 0:
            return [], {
                "status": "expected_cardinality_saturated",
                "proposed": 0,
                "accepted": 0,
                "missing_expected": [list(key) for key in missing_expected],
                "expected_cardinality": len(plan.expected_keys),
                "frozen_cardinality": _distinct_row_key_count(
                    current_rows, plan.row_key_cols
                ),
            }

    source_lines: list[str] = []
    for paper_id in ctx.paper_ids:
        source_lines.extend(
            f"[paper {paper_id}] {_evidence_line(item)}"
            for item in ctx.evidence
            if item.paper_id == paper_id and (item.quote or item.object_id)
        )
        source_lines.extend(
            f"[paper {paper_id}] {line}"
            for line in _focused_paper_lines(
                ctx, paper_id, plan, page_k=page_k
            )
        )
    source_lines = list(dict.fromkeys(source_lines))
    if not source_lines:
        return [], {"status": "no_source_pages", "proposed": 0, "accepted": 0}

    schema = [
        *plan.row_key_cols,
        *(column["name"] for column in plan.value_cols),
    ]
    prompt = (
        f"Question: {ctx.question}\n\n"
        f"Schema columns: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Current rows: {json.dumps(current_rows, ensure_ascii=False)}\n\n"
        f"Missing requested row keys: "
        f"{json.dumps(missing_expected, ensure_ascii=False)}\n\n"
        "Source pages:\n" + "\n".join(source_lines) +
        "\n\nReturn missing rows only."
    )
    try:
        response = llm.complete(prompt, system=_SYSTEM, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 -- fail closed to frozen table
        return [], {
            "status": "model_error",
            "error_type": type(exc).__name__,
            "proposed": 0,
            "accepted": 0,
        }
    proposed = _parse_rows(response, plan)
    value_cols = [column["name"] for column in plan.value_cols]
    accepted: list[dict] = []
    for row in proposed:
        if not any(row.get(column) is not None for column in value_cols):
            continue
        if _is_alias(row, [*current_rows, *accepted], plan.row_key_cols, value_cols):
            continue
        if missing_expected and not any(
            matches_expected_key(
                tuple(row.get(column) for column in plan.row_key_cols),
                expected,
                plan.row_key_cols,
            )
            for expected in missing_expected
        ):
            continue
        accepted.append(row)
        if addition_budget is not None and len(accepted) >= addition_budget:
            break
    return accepted, {
        "status": "reviewed",
        "proposed": len(proposed),
        "accepted": len(accepted),
        "alias_vetoed": len(proposed) - len(accepted),
        "page_k": page_k,
        "missing_expected": [list(key) for key in missing_expected],
    }
