"""Schema-driven table PLAN: decide the row axis, type the columns, and
enumerate the row-keys the QUESTION requests -- BEFORE any extraction.

Why plan first (reviewer correction #1): asking a vision/text model for "every
printed row" of each paper's table floods the prediction with rows the question
never asked for, destroying row PRECISION (the scorer is P/R/F1 over the row-key
set). When the question names the rows it wants ("compare DDPM, EDM and PFGM++"),
we enumerate them and constrain extraction to exactly those; when it does not
(open-ended "all methods in Table 3 of paper X"), extraction is bounded to a
single paper's table instead.

Row axis (correction #4): a row is a PAPER only when the single row-key column
genuinely denotes the paper itself (`"paper" in name`, or bare `"title"`).
`Author` / `Citation` / `Reference` / `Cited Work` and any multi-key schema are
ENTITY axes whose rows are extracted from paper CONTENT, never faked from a
selected paper's metadata title. Axis is independent of value-column presence.
"""
from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass

_DEFAULT_ROW_KEY_COL = "Paper Title"
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_ROW_KEY_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_MAX_SINGLE_LITERAL_KEY_TOKENS = 8


class RowAxis(enum.Enum):
    PAPER = "paper"
    ENTITY = "entity"


@dataclass(frozen=True)
class TablePlan:
    """The extraction contract for one table question.

    `row_key_cols` is EVERY `is_row_key` column, in schema order -- the scorer
    matches rows by the tuple `(normalize_text(row[c]) for c in row_key_cols)`,
    so a multi-key schema (e.g. Method + Base Model) is matched as a pair.
    `expected_keys` is the question-enumerated row-key tuples (possibly empty);
    when non-empty, extraction/assembly keep ONLY these rows."""
    row_key_cols: list[str]
    row_axis: RowAxis
    value_cols: list[dict]
    expected_keys: list[tuple[str, ...]]


def row_key_columns(schema: list[dict] | None) -> list[str]:
    if not schema:
        return [_DEFAULT_ROW_KEY_COL]
    keys = [col["name"] for col in schema if col.get("is_row_key")]
    return keys or [_DEFAULT_ROW_KEY_COL]


def value_columns(schema: list[dict] | None) -> list[dict]:
    return [col for col in schema if not col.get("is_row_key")] if schema else []


def classify_row_axis(row_key_cols: list[str], question: str) -> RowAxis:
    """PAPER only for a SINGLE row-key column that denotes the paper itself.
    Multi-key, or any entity-named key (method/benchmark/author/citation/...),
    is ENTITY -- its rows come from paper content, never a metadata title."""
    if len(row_key_cols) != 1:
        return RowAxis.ENTITY
    name = (row_key_cols[0] or "").strip().lower()
    if "paper" in name or name == "title":
        return RowAxis.PAPER
    return RowAxis.ENTITY


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", (text or "").strip()).strip()


_ENUM_SYSTEM = (
    "You read a research question and the row-key columns of a comparison "
    "table, and list ONLY the specific answer rows the question explicitly "
    "asks to compare (e.g. the named methods/models/datasets/settings). First "
    "identify the fixed subject, the measured output, and the varying "
    "conditions. The row identity is the requested output or varying condition; "
    "a fixed method/model/paper shared by every answer is NOT itself a row. "
    "For example, if one fixed method is evaluated with ground-truth prompts "
    "and Cube R-CNN detections, emit the two prompt/detection settings, not the "
    "fixed method name. When the row-key column is Quantity/Metric/Attribute, "
    "keep the measured quantity in every row label: a request for a ratio at "
    "the lowest and highest difficulty has rows like 'ratio for lowest "
    "difficulty' and 'ratio for highest difficulty', NOT the difficulty input "
    "values themselves. Preserve explicit identity-bearing qualifiers such as "
    "training budgets, step counts, prompt modes, and model sizes. Do NOT invent "
    "rows, do NOT include rows the question does not name. If the question does "
    "not enumerate specific rows (it asks open-endedly for 'all' rows of some "
    "table), return an empty list. A condition described in prose is NOT a "
    "literal printed row label: if you cannot name the table's row label from "
    "the question alone, return an empty list rather than copying a long phrase "
    "from the question. Prefer the shortest explicit entity label: use the "
    "named dataset or method rather than copying surrounding evaluation prose. "
    "Respond with STRICT minified JSON only: "
    '{"rows": [ {<row-key column>: <value>, ...}, ... ]} -- one object per '
    "requested row, with exactly the row-key columns as keys. When one named "
    "entity contributes multiple separate values that describe different "
    "components or roles, emit one specific row key per component instead of "
    "collapsing them into a generic entity row. Every separately requested "
    "scalar, setting, attribute, metric, component, or condition must have its "
    "own row unless the schema provides separate value columns for those facts. "
    "Preserve the question's concise identity-bearing words so the resulting "
    "keys remain distinct."
)


def enumerate_expected_keys(question: str, row_key_cols: list[str],
                            value_cols: list[dict], llm) -> list[tuple[str, ...]]:
    """Ask the LLM to list the row-keys the QUESTION requests, as tuples in
    `row_key_cols` order. Returns [] when the question is open-ended or on any
    parse/LLM failure (extraction then bounds itself per-table instead)."""
    cols = ", ".join(row_key_cols)
    prompt = (f"Question: {question}\n\nRow-key columns: {cols}\n"
              f"Value columns: {', '.join(c['name'] for c in value_cols) or '(none)'}\n\n"
              "List the specific rows the question asks to compare. Respond with "
              "the JSON object only.")
    try:
        resp = llm.complete(prompt, system=_ENUM_SYSTEM, temperature=0.0)
    except Exception:  # noqa: BLE001 -- enumeration is best-effort; failure => open-ended
        return []
    cleaned = _strip_fences(resp)
    try:
        obj = json.loads(cleaned) if cleaned else None
    except json.JSONDecodeError:
        obj = None
    rows = obj.get("rows") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        vals = tuple(str(raw.get(c)).strip() if raw.get(c) is not None else ""
                     for c in row_key_cols)
        if not any(vals):
            continue
        if vals in seen:
            continue
        seen.add(vals)
        out.append(vals)
    # A single long value is usually the model copying a prose condition rather
    # than enumerating a literal printed row label.  Keeping it is destructive:
    # assembly then drops every correctly extracted printed row.  Treat that
    # uncertain case as open-ended and let extraction preserve printed labels.
    # Multiple expected rows are not filtered here because a legitimate method
    # may carry a long qualifier (for example a training-budget suffix).
    if len(out) == 1 and len(row_key_cols) == 1:
        token_count = len(_ROW_KEY_TOKEN_RE.findall(out[0][0]))
        if token_count > _MAX_SINGLE_LITERAL_KEY_TOKENS:
            return []
    return _canonicalize_single_method_groups(out, row_key_cols)


def _canonicalize_single_method_groups(expected, row_key_cols):
    """Use the named method alone when it contributes exactly one row.

    Enumeration models sometimes copy the question's prose role (``FAST
    pruning design``). A concise schema generally expects the method identifier
    (``FAST``). A method contributing several component rows must retain those
    qualifiers, so only singleton first-token groups are shortened, and only
    for identifier-like tokens (acronyms, hyphens, or internal capitals).
    """
    if len(row_key_cols) != 1 or row_key_cols[0].strip().casefold() not in {
        "method", "methods",
    }:
        return expected
    first_tokens = [values[0].split()[0] if values and values[0].split() else ""
                    for values in expected]
    counts = {
        token.casefold(): sum(
            other.casefold() == token.casefold() for other in first_tokens
        )
        for token in first_tokens if token
    }
    out = []
    for values, first in zip(expected, first_tokens):
        identifier_like = (
            "-" in first
            or first.isupper()
            or any(character.isupper() for character in first[1:])
        )
        remainder_tokens = set(
            _ROW_KEY_TOKEN_RE.findall(values[0].casefold())
        ) - {first.casefold()}
        has_role_noun = bool(remainder_tokens.intersection({
            "approach", "benchmark", "design", "framework", "method",
            "model", "paper", "pipeline", "system", "taxonomy", "work",
        }))
        if (
            first
            and counts.get(first.casefold()) == 1
            and identifier_like
            and has_role_noun
        ):
            out.append((first,))
        else:
            out.append(values)
    return out


def plan_table(question: str, table_schema: list[dict] | None,
               paper_titles: dict[str, str], llm) -> TablePlan:
    """Build the `TablePlan`: row-key columns, axis, typed value columns, and
    (for ENTITY axes) the question-enumerated expected row-keys."""
    row_key_cols = row_key_columns(table_schema)
    val_cols = value_columns(table_schema)
    axis = classify_row_axis(row_key_cols, question)
    expected: list[tuple[str, ...]] = []
    # Enumerate requested rows for every ENTITY table AND every table with value
    # columns. A paper-axis table with value columns must still be extracted
    # and assembled rather than sent to the metadata-title-only path.
    run_enum = axis is RowAxis.ENTITY or bool(val_cols)
    if run_enum and llm is not None:
        expected = enumerate_expected_keys(question, row_key_cols, val_cols, llm)
    return TablePlan(row_key_cols=row_key_cols, row_axis=axis,
                     value_cols=val_cols, expected_keys=expected)
