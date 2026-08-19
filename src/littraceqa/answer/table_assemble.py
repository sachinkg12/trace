"""Deterministically assemble extracted per-paper rows into the final table.

The scorer matches rows by the tuple `(normalize_text(row[c]) for c in
row_key_cols)` (correction #2), so this module groups by that exact tuple
(via the `scorer_contract` seam, never a private normalizer). It then:

  * MERGES value cells across papers on the same row-key (first non-null wins) --
    a row whose columns come from several papers is legal (cross-paper rows).
  * CONSTRAINS to `plan.expected_keys` when the question enumerated them, but
    ALIAS/QUALIFIER-TOLERANTLY (correction, proven on public gold): the question
    naming often differs from the printed/gold key -- `ECM-XL (100k iterations)`
    vs `ECM-XL` (q_028), `AP-BPTT` vs printed `AT-BPTT` (q_056), an UNKNOWN second
    row-key component (q_025 `Base Model`) that must behave as a wildcard. So a
    row is kept if it matches an expected key up to parenthetical qualifiers,
    prefix, or a single-character typo, with empty expected components matching
    anything. The output row-key is ALWAYS the EXTRACTED PRINTED key (gold follows
    the printed table, not the question's wording).
  * DROPS rows whose value cells are all None (when the schema has value cols).
  * DEDUPES by normalized printed key, preserving first-seen order.
"""
from __future__ import annotations

import re

from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_plan import RowAxis

_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_KEY_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_AMBIGUOUS_EXPECTED_RE = re.compile(
    r"[()]|\b(?:for|given|over|under|using|with)\b", re.IGNORECASE
)
_SETTING_WRAPPER_TOKENS = frozenset({
    "w", "with", "prompt", "prompts", "detection", "detections",
    "setting", "settings", "mode", "modes", "2d",
})


def _strip_qual(text: str) -> str:
    """Drop parenthetical qualifiers (`ECM-XL (100k iterations)` -> `ECM-XL`)."""
    return _PAREN_RE.sub("", text).strip()


def _edit_distance_le1(a: str, b: str) -> bool:
    """True iff `a` and `b` are within one insert/delete/substitute. Used only
    for length>=5 tokens so short unrelated names don't collapse (AP-BPTT vs
    AT-BPTT differ by one substitution)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    i = 0
    while i < la and i < lb and a[i] == b[i]:
        i += 1
    if la == lb:                       # substitution
        return a[i + 1:] == b[i + 1:]
    if la < lb:                        # insertion into a
        return a[i:] == b[i + 1:]
    return a[i + 1:] == b[i:]          # deletion from a


def _setting_signature(value: str) -> tuple[str, bool]:
    """Canonical identity after removing only presentation/setting wrappers.

    This lets a question-side contract such as ``ground-truth prompts`` admit
    a printed row ``w/ Ground Truth`` and normalizes ``R-CNN``/``RCNN`` by
    joining tokens. It is intentionally equality-only and is enabled only when
    at least one wrapper token was removed, avoiding a new broad substring
    relation between arbitrary method names.
    """
    tokens = _KEY_TOKEN_RE.findall(normalize_text(value))
    kept = [token for token in tokens if token not in _SETTING_WRAPPER_TOKENS]
    return "".join(kept), len(kept) != len(tokens)


def _tol_eq(extracted, expected) -> bool:
    """Tolerant single-cell match: empty expected -> wildcard; else equal after
    normalization, equal after stripping parenthetical qualifiers, one a prefix
    of the other (len>=3), or a single-char typo (len>=5)."""
    ne = normalize_text(expected)
    if not ne:
        return True                    # unknown expected component = wildcard
    nx = normalize_text(extracted)
    if not nx:
        return False
    if nx == ne:
        return True
    sx, se = _strip_qual(nx), _strip_qual(ne)
    if sx == se:
        return True
    shorter = min(len(sx), len(se))
    if shorter >= 3 and (sx.startswith(se) or se.startswith(sx)):
        return True
    if len(sx) >= 5 and len(se) >= 5 and _edit_distance_le1(sx, se):
        return True
    signature_x, stripped_x = _setting_signature(sx)
    signature_e, stripped_e = _setting_signature(se)
    if (
        (stripped_x or stripped_e)
        and len(signature_x) >= 6
        and signature_x == signature_e
    ):
        return True
    return False


def _matches_any(key_vals, expected_tuples, row_key_cols) -> bool:
    n = len(row_key_cols)
    for exp in expected_tuples:
        exp = tuple(list(exp)[:n] + [""] * (n - len(exp)))  # pad/truncate
        if all(_tol_eq(key_vals[i], exp[i]) for i in range(n)):
            return True
    return False


def matches_expected_key(key_vals, expected, row_key_cols) -> bool:
    """Public single-key form of the scorer-aligned admission relation."""

    return _matches_any(key_vals, [expected], row_key_cols)


def _preferred_papers_for_key(
    key_vals,
    preferred_papers_by_expected_key,
    row_key_cols,
) -> set[str]:
    """Papers with a uniquely-routed claim on this extracted row key.

    Expected keys are question-side search hints while ``key_vals`` are copied
    from the source, so use the same qualifier/typo-tolerant relation as the
    existing admission filter. Multiple matching routes fail open by unioning
    their owners; the merge never erases a value merely because ownership is
    ambiguous.
    """

    out: set[str] = set()
    for expected_key, paper_ids in (preferred_papers_by_expected_key or {}).items():
        if _matches_any(key_vals, [expected_key], row_key_cols):
            out.update(str(paper_id) for paper_id in paper_ids if paper_id)
    return out


def concise_missing_expected_rows(plan, rows) -> list[dict]:
    """Return conservative null-cell rows for explicitly requested keys.

    The official scorer awards row-key F1 independently from value cells. A
    missing row and a present row with null values receive the same cell score,
    so dropping an unextracted but explicit key throws away free row credit.
    Restrict recovery to single-key ENTITY tables, short unqualified labels,
    and at least two missing keys. The last condition avoids turning a single
    question/source spelling disagreement (public q_056 AP-BPTT vs AT-BPTT)
    into a speculative extra row.
    """

    if (
        plan.row_axis is not RowAxis.ENTITY
        or len(plan.row_key_cols) != 1
        or not plan.value_cols
    ):
        return []
    key_column = plan.row_key_cols[0]
    existing = {normalize_text(row.get(key_column)) for row in rows}
    missing: list[str] = []
    for values in plan.expected_keys:
        if len(values) != 1 or not isinstance(values[0], str):
            continue
        value = values[0].strip()
        token_count = len(_KEY_TOKEN_RE.findall(value))
        if (
            not value
            or token_count > 3
            or _AMBIGUOUS_EXPECTED_RE.search(value)
            or normalize_text(value) in existing
        ):
            continue
        missing.append(value)
        existing.add(normalize_text(value))
    if len(missing) < 2:
        return []
    return [
        {
            key_column: value,
            **{column["name"]: None for column in plan.value_cols},
        }
        for value in missing
    ]


def source_attested_missing_expected_rows(
    plan,
    rows,
    source_attested_expected_keys,
) -> list[dict]:
    """Return null-cell rows only for complete, source-printed expected keys."""

    if plan.row_axis is not RowAxis.ENTITY:
        return []
    existing = [
        tuple(row.get(column) for column in plan.row_key_cols)
        for row in rows
    ]
    recovered: list[dict] = []
    for values in source_attested_expected_keys or []:
        if len(values) != len(plan.row_key_cols):
            continue
        normalized = tuple(normalize_text(value) for value in values)
        if (
            any(not value for value in normalized)
            or any(
                _matches_any(key_values, [values], plan.row_key_cols)
                for key_values in existing
            )
        ):
            continue
        recovered.append({
            **{
                column: values[position]
                for position, column in enumerate(plan.row_key_cols)
            },
            **{column["name"]: None for column in plan.value_cols},
        })
        existing.append(tuple(values))
    return recovered


def assemble_rows(
    plan,
    per_paper_rows,
    *,
    retain_concise_missing_expected_rows: bool = False,
    preferred_papers_by_expected_key=None,
    source_attested_expected_keys=None,
) -> list[dict]:
    # A value-bearing PAPER-axis table canonicalizes every extracted row key to
    # the selected paper's released metadata title before assembly. Question
    # enumeration, however, often produces a descriptive shorthand ("GRAB
    # benchmark", "CMAD paper"). Filtering the authoritative title against that
    # shorthand is self-contradictory and used to erase otherwise-valid rows.
    # Selected paper IDs already provide the precision boundary for this axis;
    # expected-key filtering remains necessary only for content/entity rows.
    expected = [] if plan.row_axis is RowAxis.PAPER else plan.expected_keys
    order: list[tuple] = []
    by_key: dict[tuple, dict] = {}
    cell_priorities: dict[tuple, dict[str, int]] = {}

    for paper_id, rows in per_paper_rows:
        for row in rows:
            key_vals = tuple(row.get(c) for c in plan.row_key_cols)
            if not any(key_vals):
                continue
            if expected and not _matches_any(key_vals, expected, plan.row_key_cols):
                continue  # not a requested row (tolerant) -> drop
            nkey = tuple(normalize_text(v) for v in key_vals)
            if nkey not in by_key:
                # PRESERVE the extracted printed key (gold follows the table).
                merged = {c: (key_vals[i] if isinstance(key_vals[i], str) else "")
                          for i, c in enumerate(plan.row_key_cols)}
                for col in plan.value_cols:
                    merged[col["name"]] = None
                by_key[nkey] = merged
                cell_priorities[nkey] = {
                    col["name"]: 1 for col in plan.value_cols
                }
                order.append(nkey)
            merged = by_key[nkey]
            preferred_papers = _preferred_papers_for_key(
                key_vals,
                preferred_papers_by_expected_key,
                plan.row_key_cols,
            )
            source_priority = 0 if paper_id in preferred_papers else 1
            for col in plan.value_cols:
                column = col["name"]
                candidate = row.get(column)
                if candidate is None:
                    continue
                if (
                    merged[column] is None
                    or source_priority < cell_priorities[nkey][column]
                ):
                    merged[column] = candidate
                    cell_priorities[nkey][column] = source_priority

    out: list[dict] = []
    for nkey in order:
        merged = by_key[nkey]
        if plan.value_cols and not any(
                merged[col["name"]] is not None for col in plan.value_cols):
            continue
        out.append(merged)
    if retain_concise_missing_expected_rows:
        out.extend(concise_missing_expected_rows(plan, out))
    out.extend(
        source_attested_missing_expected_rows(
            plan, out, source_attested_expected_keys
        )
    )
    return out
