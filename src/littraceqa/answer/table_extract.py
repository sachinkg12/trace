"""Extract a paper's table rows from BOTH its visual and non-visual evidence.

Correction #3: 5 of 11 dev table-answer records are primary figure / equation /
citation / text -- their rows live in localized QUOTES / parsed text, not in a
renderable table image. So extraction runs two paths per paper and merges:

  * VISUAL -- every unique `table`/`figure` locator is rendered to PNG and read
    by the vision model (PyMuPDF scrambles dense metric tables column-major, so
    values are only recoverable from pixels).
  * NON-VISUAL -- the paper's `text_span` / `equation_algorithm` /
    `citation_context` evidence quotes are read by the text LLM.

Both paths are CONSTRAINED to the plan's `expected_keys` when the question
enumerated them (correction #1): the model is told to return ONLY those rows,
so a paper's table does not flood the prediction with unrequested rows. When
`expected_keys` is empty (open-ended question), each path returns the rows it
finds in that single image / quote-set -- still bounded to one paper.

Rows are parsed defensively (correction #2 -- every `is_row_key` column is a
verbatim string, never number-coerced; value columns typed via `_coerce_number`;
missing cells are `None`, never `0`/`""`). Nothing here raises.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
import math
import re
import unicodedata

from littraceqa.answer.table import _clean_cell, _coerce_number, _strip_code_fences
from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_route import route_expected_keys
from littraceqa.answer.vision import render_evidence_png

_VISUAL_SOURCE_TYPES = ("table", "figure")
_METHOD_LEAF_RE = re.compile(r"\bw/\s*", re.IGNORECASE)
_METHOD_ROW_KEY_NAMES = {"method", "methods", "method name"}
_TRAILING_NUMERIC_CITATION_RE = re.compile(r"\s*\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]\s*$")
_TRAILING_PARENTHETICAL_RE = re.compile(r"\s*\(([^()]*)\)\s*$")
# The indexed localizer historically returned at most ``object_k=3`` visual
# locators per paper. Passage->object grounding must substitute a better locator
# into that budget, not grow the number of paid vision calls without bound.
_MAX_VISUAL_LOCATORS_PER_PAPER = 3
_MAX_RETRY_PAGE_CHARS = 6000
_MAX_TABLE_TEXT_PAGE_CHARS = 6000
_RETRY_STOPWORDS = {
    "a", "an", "and", "average", "component", "data", "dataset", "design",
    "for", "from", "in", "learning", "method", "of", "on", "paper", "the",
    "to", "used", "value", "with",
}

_VISION_SYSTEM = (
    "You are reading ONE image of a comparison table (or figure) from an "
    "academic paper. Read the row and column headers and the printed data rows. "
    "Report values ONLY from what is visibly printed -- never guess or compute. "
    "Respond with STRICT minified JSON only: "
    '{"rows": [ {<row-key column>: <name>, <value column>: <value>}, ... ]}. '
    "Copy each row-key and string-column value exactly as printed. Return a "
    "numeric column as a bare JSON number and a string column as a JSON string, "
    "even when the printed string looks numeric. In a hierarchical Method table, "
    "use the specific leaf row label (for example 'w/ Ground Truth'), not the "
    "parent method name concatenated with that leaf. A printed missing-value "
    "marker such as '-' is still a visible string value: return '-' rather than "
    "null. Use null only when the requested cell is not present in the image. "
    "If the requested value-column header appears more than once under different "
    "grouped headers and the schema does not name a group, use its leftmost "
    "occurrence."
)

_TARGETED_VISION_SYSTEM = (
    "You are verifying ONE requested row in ONE academic-paper table image. "
    "Find the requested row and then follow the COMPLETE hierarchical header "
    "path for every requested value, including dataset, metric, step/NFE, IPC, "
    "split, and with/without-voting qualifiers. Do not take a nearby value from "
    "the same row or the requested value from a nearby row. Respond with STRICT "
    "minified JSON only: "
    '{"rows":[{<row-key column>:<printed row label>,'
    ' <value column>:<printed value>}]}.'
    " Return at most one row. Copy the concise source row label, removing only "
    "a trailing citation or '(ours)' marker; preserve meaningful numeric/model "
    "qualifiers. If the exact row and qualified value are not both visibly "
    "present, return {\"rows\":[]}. Never infer a value from prose or another "
    "table. Preserve printed precision and missing markers such as '-'."
)

_GRID_VISION_SYSTEM = (
    "You transcribe ONE complete rectangular data grid from an academic-paper "
    "table or figure. Do not answer the question and do not select only the "
    "requested rows or columns. Transcribe every visible header, data column, "
    "and data row from the relevant grid. Respond with STRICT minified JSON "
    "only in this shape: "
    '{"columns":[{"headers":["group","leaf"],"role":"data"}],'
    '"rows":[["cell",1.2,null]]}. '
    "Columns and every row array must use the same left-to-right order. The "
    "headers list is the complete top-to-bottom hierarchical header path for "
    "that column, omitting blank levels. Mark columns that identify a row as "
    "role=row_header and measured/value columns as role=data. For hierarchical "
    "row labels, emit one row_header column per visible level and repeat merged "
    "parent labels on every child row; name the deepest leaf column after the "
    "entity it identifies (for example Method). Copy strings exactly as "
    "printed, use bare JSON numbers for numeric cells, preserve printed missing "
    "markers such as '-' as strings, and use null only for an absent cell. "
    "Never infer, compute, rename, or paraphrase a cell."
)

_TEXT_SYSTEM = (
    "You extract comparison-table rows from academic-paper evidence quotes "
    "(text, equations, or citation contexts). Use ONLY the quotes provided -- "
    "do not guess or compute values not stated. Respond with STRICT minified "
    'JSON only: {"rows": [ {<row-key column>: <name>, <value column>: <value>}, '
    "... ]}. Copy row-key and string-column values verbatim; numeric columns as "
    "bare JSON numbers. In a hierarchical Method table, use the specific leaf "
    "row label (for example 'w/ Ground Truth'), not the parent method name "
    "concatenated with that leaf. A printed missing-value marker such as '-' is "
    "a string value, not null. Use null only for a value not present in the "
    "quotes. Match semantically equivalent wording: for example, a stated "
    "'average increase of X IoU over real data' directly answers an 'average "
    "IoU gain over real data' request, and a component's stated complexity "
    "answers the question's named complexity role. Do not require the quote to "
    "repeat the question verbatim. When the question asks which SYMBOL denotes "
    "an object, return only the symbol (for example 'B'), not its definition "
    "or set contents (for example '{b1, ..., bK}'). If a requested "
    "headline contains multiple coupled values for one method, keep every "
    "requested value in that method's single row and copy the claim's wording "
    "from the evidence rather than paraphrasing it. If a question asks for "
    "several independent facts but the schema has one generic value column, "
    "emit one distinct row per requested fact and retain the shortest question-"
    "stated identity plus the fact's differentiating role in the row key. If a requested "
    "value-column header appears "
    "more than once under different grouped headers and no group is named, use "
    "its leftmost occurrence."
)


def unique_visual_locators(evidence, paper_id):
    """All `table`/`figure` evidence items for `paper_id`, deduped by
    canonical `(page, object_id)`, highest-confidence first, within the fixed
    per-paper vision budget. The budget preserves the pre-promotion production
    ceiling while allowing a stronger passage-grounded object to replace a weak
    caption hit instead of adding an unbounded paid call."""
    seen: set[tuple] = set()
    out = []
    candidates = [
        (position, ev)
        for position, ev in enumerate(evidence or [])
        if ev.paper_id == paper_id and ev.source_type in _VISUAL_SOURCE_TYPES
    ]
    candidates.sort(
        key=lambda item: (-float(getattr(item[1], "confidence", 0.0)), item[0])
    )
    for _position, ev in candidates:
        object_id = re.sub(r"\s+", " ", ev.object_id or "").strip().casefold()
        key = (ev.page, object_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
        if len(out) >= _MAX_VISUAL_LOCATORS_PER_PAPER:
            break
    return out


def _nonvisual_evidence(evidence, paper_id):
    """This paper's non-visual evidence (text/equation/citation) -- the quotes
    the text path reads."""
    return [ev for ev in (evidence or [])
            if ev.paper_id == paper_id and ev.source_type not in _VISUAL_SOURCE_TYPES]


def _requested_rows_clause(
    plan,
    *,
    exact_output_keys: bool = False,
    own_paper_only: bool = False,
    paper_title: str = "",
) -> str:
    if not plan.expected_keys:
        if own_paper_only:
            return (
                "This is one paper in a multi-paper comparison. Return exactly "
                "ONE row: the method/model/work introduced by THIS paper that "
                "satisfies the question. The paper title is: "
                f"{paper_title or '(unknown)'}. Do not return baselines, prior "
                "work, compared methods, ablations, component variants, or "
                "every row visible in a comparison table. If this paper does "
                "not introduce a qualifying answer, return an empty rows list."
            )
        return ("Return every data row of this table that is relevant to the "
                "question.")
    labels = []
    for keytuple in plan.expected_keys:
        labels.append(" / ".join(f"{c}={v}" for c, v in zip(plan.row_key_cols, keytuple) if v))
    if exact_output_keys:
        return (
            "Use these as the EXACT output row-key values, one output row per "
            "listed key:\n- "
            + "\n- ".join(labels)
            + "\nDo not shorten, expand, or paraphrase these row keys. Use the "
              "evidence only to fill the requested value columns. Omit a row "
              "when the evidence does not support any requested value."
        )
    return (
        "Use these as SEARCH HINTS for the requested rows (one object each), "
        "and return no unrelated rows:\n- "
        + "\n- ".join(labels)
        + "\nThe hint may be a prose description rather than the source's row "
          "label. For every output row, copy the concise row-key value exactly "
          "as printed in the source. Never copy or expand the hint when the "
          "source prints a shorter label."
    )


def _columns_clause(plan) -> str:
    rk = ", ".join(plan.row_key_cols)
    vc = ", ".join(
        f"{column['name']} ({column.get('type') or 'string'})"
        for column in plan.value_cols
    ) or "(none)"
    return f"Row-key columns: {rk}\nValue columns: {vc}"


def _build_vision_prompt(
    question,
    plan,
    ev,
    *,
    own_paper_only: bool = False,
    paper_title: str = "",
) -> str:
    where = ev.object_id or "the table"
    requested = _requested_rows_clause(
        plan,
        own_paper_only=own_paper_only,
        paper_title=paper_title,
    )
    return (f"Question: {question}\n\nThis image is {where} on page {ev.page} of "
            f"one paper.\n{_columns_clause(plan)}\n\n"
            f"{requested}\n\n"
            "Respond with the JSON object only.")


def _build_grid_vision_prompt(question, ev) -> str:
    where = ev.object_id or "the table"
    return (
        f"Navigation question (do not answer it): {question}\n\n"
        f"This image is {where} on page {ev.page} of one paper. Transcribe the "
        "complete visible data grid that contains information relevant to the "
        "navigation question. Do not filter rows or columns using the question. "
        "Respond with the JSON object only."
    )


def _build_targeted_vision_prompt(question, plan, ev) -> str:
    """A one-row verification prompt used only by the opt-in repair path."""

    where = ev.object_id or "the table"
    requested = _requested_rows_clause(plan)
    return (
        f"Question: {question}\n\nThis image is {where} on page {ev.page} of "
        f"one paper.\n{_columns_clause(plan)}\n\n{requested}\n\n"
        "Verify this one requested row against the image. Treat every qualifier "
        "in the value-column name and question as mandatory. Return the JSON "
        "object only."
    )


def _build_text_prompt(
    question,
    plan,
    lines,
    *,
    own_paper_only: bool = False,
    paper_title: str = "",
) -> str:
    body = "\n".join(lines)
    requested = _requested_rows_clause(
        plan,
        exact_output_keys=True,
        own_paper_only=own_paper_only,
        paper_title=paper_title,
    )
    return (f"Question: {question}\n\n{_columns_clause(plan)}\n\n"
            f"Evidence (one paper; each item tagged with its source type, page "
            f"and object id):\n{body}\n\n"
            f"{requested}\n\n"
            "Respond with the JSON object only.")


def _evidence_line(ev) -> str:
    """One non-visual evidence item rendered with its source type, page and
    object id -- the object id can itself be the requested answer (for example,
    equation IDs that live only in `object_id`), so it must reach the
    extractor."""
    tag = f"{ev.source_type} p.{ev.page}"
    if ev.object_id:
        tag += f" ({ev.object_id})"
    return f"[{tag}] {ev.quote or ''}".rstrip()


def _parse_rows(response: str, plan) -> list[dict]:
    """Parse `{"rows":[...]}` into typed row dicts. Every row-key column is a
    verbatim cleaned string (NEVER number-coerced -- the scorer matches row-keys
    by `normalize_text`); value columns typed via `_coerce_number` for numbers,
    cleaned strings otherwise; missing -> None. A row with no non-empty row-key
    is dropped."""
    cleaned = _strip_code_fences(response or "")
    try:
        obj = json.loads(cleaned) if cleaned else None
    except json.JSONDecodeError:
        obj = None
    raw_rows = obj.get("rows") if isinstance(obj, dict) else None
    if not isinstance(raw_rows, list):
        return []
    out: list[dict] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        row: dict = {}
        for col in plan.row_key_cols:
            v = raw.get(col)
            row[col] = _clean_row_key(v, col)
        if not any(row[c] for c in plan.row_key_cols):
            continue
        for col in plan.value_cols:
            v = raw.get(col["name"])
            if col.get("type") == "number":
                row[col["name"]] = _coerce_number(v)
            else:
                row[col["name"]] = _coerce_string_cell(v, col["name"])
        out.append(row)
    return out


def _consensus_rows(reads: list[list[dict]], plan) -> list[dict]:
    """Keep only row/cell readings supported by a strict call majority.

    Vision errors in dense tables are usually *confident adjacent-cell reads*;
    accepting the first non-null value therefore preserves the error.  This
    helper aligns rows by a uniquely matched requested key (or by their scorer
    normalized printed key for open-ended tables), then votes independently on
    every printed key and value cell.  A disagreement becomes ``None`` rather
    than an invented tie-break.  Downstream assembly already drops an all-null
    row, so uncertainty cannot silently overwrite a stronger extraction.

    One read is an exact pass-through, preserving the production default.
    Within a read, a group contributes at most one vote; duplicate rows from
    one model response cannot manufacture consensus.
    """

    if len(reads) <= 1:
        return list(reads[0]) if reads else []

    from littraceqa.answer.table_assemble import matches_expected_key

    quorum = len(reads) // 2 + 1
    columns = [*plan.row_key_cols, *(column["name"] for column in plan.value_cols)]

    def group_for(row: dict):
        key_values = tuple(row.get(column) for column in plan.row_key_cols)
        matches = [
            position
            for position, expected in enumerate(plan.expected_keys)
            if matches_expected_key(key_values, expected, plan.row_key_cols)
        ]
        if len(matches) == 1:
            return ("expected", matches[0])
        normalized = tuple(normalize_text(value) for value in key_values)
        return ("printed", *normalized) if any(normalized) else None

    def vote_key(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return ("bool", value)
        if isinstance(value, (int, float)):
            return ("number", float(value))
        if isinstance(value, str):
            normalized = normalize_text(value)
            return ("string", normalized) if normalized else None
        return None

    grouped: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for read in reads:
        seen_in_read: set[tuple] = set()
        for row in read:
            group = group_for(row)
            if group is None or group in seen_in_read:
                continue
            seen_in_read.add(group)
            if group not in grouped:
                grouped[group] = []
                order.append(group)
            grouped[group].append(row)

    out: list[dict] = []
    for group in order:
        voters = grouped[group]
        if len(voters) < quorum:
            continue
        consensus: dict = {}
        for column in columns:
            counts: dict[tuple, int] = {}
            representative: dict[tuple, object] = {}
            for row in voters:
                value = row.get(column)
                key = vote_key(value)
                if key is None:
                    continue
                counts[key] = counts.get(key, 0) + 1
                representative.setdefault(key, value)
            winners = [key for key, count in counts.items() if count >= quorum]
            if len(winners) == 1:
                consensus[column] = representative[winners[0]]
            else:
                consensus[column] = None
        if all(consensus.get(column) is not None for column in plan.row_key_cols):
            out.append(consensus)
    return out


def _header_tokens(value: str) -> list[str]:
    """Comparable tokens for schema names and hierarchical table headers."""

    aliases = {
        "benchmarks": "benchmark",
        "datasets": "dataset",
        "methods": "method",
        "models": "model",
        "steps": "step",
        "nfe": "step",
    }
    return [
        aliases.get(token, token)
        for token in re.findall(r"[a-z0-9]+", normalize_text(value))
    ]


def _grid_column_score(
    schema_name: str,
    headers: list[str],
    *,
    question: str,
    row_key: bool,
    role: str,
) -> tuple[int, int]:
    """Deterministically score one grid column for one scorer column.

    Numeric qualifiers are hard constraints: ``2-step FID`` cannot bind to a
    printed ``NFE 1 / FID`` column. Question tokens only break ties between
    otherwise-equivalent grouped headers; they cannot create a match by
    themselves.
    """

    schema_tokens = _header_tokens(schema_name)
    path = " ".join(headers)
    path_tokens = _header_tokens(path)
    if not schema_tokens or not path_tokens:
        return (0, 0)
    schema_numbers = {token for token in schema_tokens if token.isdigit()}
    path_numbers = {token for token in path_tokens if token.isdigit()}
    if schema_numbers and not schema_numbers.issubset(path_numbers):
        return (-1, 0)
    schema_set = set(schema_tokens)
    path_set = set(path_tokens)
    overlap = len(schema_set & path_set)
    if not overlap:
        return (0, 0)
    leaf = normalize_text(headers[-1]) if headers else ""
    normalized_schema = normalize_text(schema_name)
    coverage = (100 * overlap) // len(schema_set)
    exact_bonus = 80 if normalized_schema in {normalize_text(path), leaf} else 0
    role_bonus = 30 if (row_key and role == "row_header") else 0
    if not row_key and role == "data":
        role_bonus = 30
    if row_key != (role == "row_header"):
        role_bonus -= 30
    question_tokens = set(_header_tokens(question))
    tie_break = len((path_set - schema_set) & question_tokens)
    return (coverage + exact_bonus + role_bonus, tie_break)


def _assign_grid_columns(columns: list[dict], plan, question: str) -> dict[str, int]:
    """Return the unique maximum-weight schema-to-source column assignment.

    Local greedy matching can cross-swap a generic header and a qualified
    sibling: the first schema field consumes the sibling because it has more
    lexical overlap, leaving the exact owner with the generic column.  The
    scorer schema is small, so solve the complete one-to-one assignment with a
    bitmask dynamic program.  An equal-scoring global alternative is source
    ambiguity, not a reason to choose the leftmost physical column.
    """

    schema_columns = [
        (name, True) for name in plan.row_key_cols
    ] + [
        (column["name"], False) for column in plan.value_cols
    ]
    if len({name for name, _row_key in schema_columns}) != len(schema_columns):
        return {}

    # When a scorer field has its own distinguishing token, a source token
    # owned only by a sibling field is a hard conflicting qualifier. Shared
    # tokens (for example ``mean`` and ``score``) remain generic evidence.
    token_owners: dict[tuple[bool, str], set[int]] = {}
    for schema_position, (schema_name, row_key) in enumerate(schema_columns):
        for token in set(_header_tokens(schema_name)):
            token_owners.setdefault((row_key, token), set()).add(schema_position)

    candidates: list[list[tuple[int | None, tuple[int, int]]]] = []
    for schema_position, (schema_name, row_key) in enumerate(schema_columns):
        matches: list[tuple[int | None, tuple[int, int]]] = []
        own_exclusive_tokens = {
            token
            for token in set(_header_tokens(schema_name))
            if token_owners.get((row_key, token)) == {schema_position}
        }
        for source_position, column in enumerate(columns):
            role = column["role"]
            if role and row_key != (role == "row_header"):
                continue
            path_tokens = set(_header_tokens(" ".join(column["headers"])))
            if own_exclusive_tokens and any(
                owners and schema_position not in owners
                for token in path_tokens
                if (owners := token_owners.get((row_key, token))) is not None
            ):
                continue
            quality = _grid_column_score(
                schema_name,
                column["headers"],
                question=question,
                row_key=row_key,
                role=role,
            )
            if quality[0] > 0:
                matches.append((source_position, quality))

        if row_key and not matches:
            # Preserve the prior blank-header fallback only when the complete
            # assignment makes its physical position unique.  Explicit data
            # roles can never stand in for a row identity.
            fallback = [
                source_position
                for source_position, column in enumerate(columns)
                if column["role"] == "row_header"
            ]
            if not fallback:
                fallback = [
                    source_position
                    for source_position, column in enumerate(columns)
                    if not column["role"]
                ]
            matches.extend(
                (source_position, (0, 0)) for source_position in fallback
            )
        elif not row_key:
            # A value field may remain unmapped and become null.  Row keys are
            # mandatory and therefore never receive this abstention edge.
            matches.append((None, (0, 0)))
        candidates.append(matches)

    @lru_cache(maxsize=None)
    def solve(
        schema_position: int, used_mask: int
    ) -> tuple[tuple[int, int], tuple[int | None, ...], int] | None:
        if schema_position == len(schema_columns):
            return (0, 0), (), 1
        best: tuple[tuple[int, int], tuple[int | None, ...], int] | None = None
        for source_position, quality in candidates[schema_position]:
            if (
                source_position is not None
                and used_mask & (1 << source_position)
            ):
                continue
            child = solve(
                schema_position + 1,
                used_mask | (
                    0 if source_position is None else 1 << source_position
                ),
            )
            if child is None:
                continue
            total = (
                quality[0] + child[0][0],
                quality[1] + child[0][1],
            )
            assignment = (source_position, *child[1])
            if best is None or total > best[0]:
                best = total, assignment, child[2]
            elif total == best[0]:
                best = best[0], best[1], min(2, best[2] + child[2])
        return best

    solution = solve(0, 0)
    if solution is None or solution[2] != 1:
        return {}
    return {
        schema_name: source_position
        for (schema_name, _row_key), source_position in zip(
            schema_columns, solution[1], strict=True
        )
        if source_position is not None
    }


def _parse_grid_rows(response: str, plan, question: str) -> list[dict]:
    """Project a schema-independent visual grid into the scorer schema."""

    cleaned = _strip_code_fences(response or "")
    try:
        payload = json.loads(cleaned) if cleaned else None
    except json.JSONDecodeError:
        return []
    raw_columns = payload.get("columns") if isinstance(payload, dict) else None
    raw_rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(raw_columns, list) or not isinstance(raw_rows, list):
        return []

    columns: list[dict] = []
    for raw in raw_columns:
        if not isinstance(raw, dict):
            return []
        headers = raw.get("headers")
        if not isinstance(headers, list) or not all(
            isinstance(header, str) for header in headers
        ):
            return []
        cleaned_headers = [
            _clean_cell(header) for header in headers if _clean_cell(header)
        ]
        role = str(raw.get("role") or "").strip().casefold()
        if role not in {"row_header", "data"}:
            role = ""
        columns.append({"headers": cleaned_headers, "role": role})
    if not columns:
        return []

    assigned = _assign_grid_columns(columns, plan, question)

    # Every scorer row needs all row-key columns mapped. Value columns may stay
    # unmapped and become null; assembly then applies its existing null policy.
    if any(name not in assigned for name in plan.row_key_cols):
        return []
    projected: list[dict] = []
    for raw_row in raw_rows:
        # ``columns`` is the deterministic source scaffold.  A short row can
        # mean either a missing trailing cell or an omitted interior cell, and
        # a long row can mean an untranscribed header.  Both shapes make every
        # subsequent value position ambiguous.  Require an explicit JSON null
        # for a visibly blank cell and reject non-rectangular rows wholesale.
        if not isinstance(raw_row, list) or len(raw_row) != len(columns):
            continue
        item = {
            schema_name: raw_row[position]
            for schema_name, position in assigned.items()
        }
        projected.append(item)
    return _parse_rows(json.dumps({"rows": projected}), plan)


def _clean_row_key(value: object, column: str) -> str | None:
    """Clean a printed row key and detach hierarchical Method leaf labels.

    Vision commonly reads a parent cell plus its indented child as one string,
    e.g. ``DetAny3D (ours) w/ Ground Truth``. The scorer's row key is the leaf
    printed in the child row (``w/ Ground Truth``), so keep that suffix for a
    Method column. Other row-key columns remain verbatim-cleaned.
    """
    if not isinstance(value, str):
        return None
    cleaned = unicodedata.normalize("NFKC", _clean_cell(value))
    if normalize_text(column) in _METHOD_ROW_KEY_NAMES:
        cleaned = _strip_method_attribution(cleaned)
    if column.strip().casefold() == "method":
        match = _METHOD_LEAF_RE.search(cleaned)
        if match is not None and match.start() > 0:
            cleaned = cleaned[match.start():].strip()
    if normalize_text(column) == "base model":
        cleaned = _normalize_base_model_value(cleaned)
    return cleaned or None


def _strip_method_attribution(value: str) -> str:
    """Remove terminal bibliography attribution, never semantic qualifiers.

    Comparison tables frequently print row labels such as ``ATT [22]`` or
    ``iCT-deep (Song & Dhariwal, 2023)``.  The official row-key annotations use
    the method identifier, so retaining the attribution creates a false extra
    row and can let a later duplicate overwrite the owning paper's value.

    Only two source-explicit suffixes are removed: a numeric square-bracket
    citation, and a terminal parenthetical containing ``ours`` or a four-digit
    publication year.  Qualifiers such as ``(102.4M)``, ``(large)``, or
    ``(without voting)`` remain part of the identifier.
    """

    cleaned = _TRAILING_NUMERIC_CITATION_RE.sub("", value).strip()
    match = _TRAILING_PARENTHETICAL_RE.search(cleaned)
    if match is None:
        return cleaned
    attribution = normalize_text(match.group(1))
    if attribution == "ours" or re.search(r"\b(?:19|20)\d{2}\b", attribution):
        return cleaned[:match.start()].strip()
    return cleaned


def _normalize_base_model_value(value: str) -> str:
    """Canonicalize two source-equivalent base-model spellings.

    The official composite row key is exact after whitespace normalization.
    Papers commonly print ``Infinity-2B and Infinity-8B`` for the compact
    schema value ``Infinity-2B/8B`` and append the generic noun ``model`` to a
    version name. These transformations preserve every identifier and size;
    they do not infer an absent model.
    """
    cleaned = re.sub(r"\s+model\s*$", "", value, flags=re.IGNORECASE).strip()
    paired = re.fullmatch(
        r"(.+?)-(\d+(?:\.\d+)?[BM])\s+and\s+\1-(\d+(?:\.\d+)?[BM])",
        cleaned,
        flags=re.IGNORECASE,
    )
    if paired:
        return f"{paired.group(1)}-{paired.group(2)}/{paired.group(3)}"
    compact_size = re.fullmatch(
        r"([A-Za-z][A-Za-z0-9._-]*?[A-Za-z])"
        r"(\d+(?:\.\d+)?[BM])",
        cleaned,
        flags=re.IGNORECASE,
    )
    if compact_size:
        return f"{compact_size.group(1)}-{compact_size.group(2)}"
    return cleaned


def _coerce_string_cell(value: object, column: str = "") -> str | None:
    """Preserve scalar values for scorer-declared string columns.

    Organizer schemas intentionally type some metric columns as ``string`` so
    they can contain both numbers and markers such as ``-``. LLMs still emit a
    printed numeric value as a JSON number. Dropping that number turns every
    extracted row into an all-null row, so stringify finite numeric scalars
    deterministically while retaining textual markers verbatim.
    """
    if isinstance(value, str):
        cleaned = unicodedata.normalize("NFKC", _clean_cell(value))
    elif isinstance(value, bool):
        return None
    elif isinstance(value, int):
        cleaned = str(value)
    elif isinstance(value, float) and math.isfinite(value):
        cleaned = format(value, ".15g")
    else:
        return None

    column_name = normalize_text(column)
    if re.fullmatch(r"\d+[a-z]?", cleaned, flags=re.IGNORECASE):
        if "equation" in column_name and "id" in column_name:
            cleaned = f"Equation {cleaned}"
        elif "algorithm" in column_name and "id" in column_name:
            cleaned = f"Algorithm {cleaned}"
    if column_name == "base model":
        cleaned = _normalize_base_model_value(cleaned)
    return cleaned or None


def _visual_rows(
    ctx,
    paper_id,
    plan,
    vision_llm,
    *,
    own_paper_only: bool = False,
    extraction_mode: str = "direct",
    retry_expected_keys: list[tuple[str, ...]] | None = None,
    consensus_repeats: int = 1,
) -> list[dict]:
    rows: list[dict] = []
    pdf_bytes = (getattr(ctx, "pdf_bytes_by_id", None) or {}).get(paper_id)
    paper_title = str(
        (getattr(ctx, "paper_titles", None) or {}).get(paper_id) or ""
    )
    rendered: list[tuple[object, bytes]] = []
    for ev in unique_visual_locators(ctx.evidence, paper_id):
        render_kwargs = {"pdf_bytes": pdf_bytes}
        if extraction_mode == "full_grid_crop":
            render_kwargs["focus_table"] = True
        png = render_evidence_png(ev, **render_kwargs)
        if png is None:
            continue
        rendered.append((ev, png))
        if extraction_mode in {"full_grid", "full_grid_crop"}:
            prompt = _build_grid_vision_prompt(ctx.question, ev)
            system = _GRID_VISION_SYSTEM
        else:
            prompt = _build_vision_prompt(
                ctx.question,
                plan,
                ev,
                own_paper_only=own_paper_only,
                paper_title=paper_title,
            )
            system = _VISION_SYSTEM
        reads: list[list[dict]] = []
        for _repeat in range(consensus_repeats):
            try:
                response = vision_llm.complete(
                    prompt,
                    system=system,
                    temperature=0.0,
                    images=[png],
                )
            except Exception:  # noqa: BLE001 -- one failed vote is abstention
                reads.append([])
                continue
            if extraction_mode in {"full_grid", "full_grid_crop"}:
                reads.append(_parse_grid_rows(response, plan, ctx.question))
            else:
                reads.append(_parse_rows(response, plan))
        rows.extend(_consensus_rows(reads, plan))

    # Broad table transcription is efficient, but dense tables routinely yield
    # a correct row with the adjacent grouped-header value.  The opt-in repair
    # repeats only uniquely-owned (or single-paper) requested rows with a
    # one-row contract.  Successful repairs replace broad candidates for the
    # same expected key so assembly cannot keep an earlier, less specific cell.
    # This never changes evidence emission and is bounded by
    # len(retry_expected_keys) * the existing three-locator ceiling.
    if extraction_mode == "direct" and retry_expected_keys and rendered:
        from littraceqa.answer.table_assemble import matches_expected_key

        repaired: list[dict] = []
        repaired_keys: list[tuple[str, ...]] = []
        for expected in retry_expected_keys:
            one_row_plan = replace(plan, expected_keys=[tuple(expected)])
            candidates: list[dict] = []
            for ev, png in rendered:
                try:
                    response = vision_llm.complete(
                        _build_targeted_vision_prompt(
                            ctx.question, one_row_plan, ev
                        ),
                        system=_TARGETED_VISION_SYSTEM,
                        temperature=0.0,
                        images=[png],
                    )
                except Exception:  # noqa: BLE001 -- one repair call may fail
                    continue
                for row in _parse_rows(response, one_row_plan):
                    key_values = tuple(
                        row.get(column) for column in plan.row_key_cols
                    )
                    if not matches_expected_key(
                        key_values, expected, plan.row_key_cols
                    ):
                        continue
                    if not any(
                        row.get(column["name"]) is not None
                        for column in plan.value_cols
                    ):
                        continue
                    candidates.append(row)
            if not candidates:
                continue
            # Locator ordering is deterministic and already confidence-ranked.
            # Keep one candidate per expected key; later A/B gates decide
            # whether this repair policy is safe enough to promote.
            repaired.append(candidates[0])
            repaired_keys.append(tuple(expected))

        if repaired:
            rows = [
                row
                for row in rows
                if not any(
                    matches_expected_key(
                        tuple(row.get(column) for column in plan.row_key_cols),
                        expected,
                        plan.row_key_cols,
                    )
                    for expected in repaired_keys
                )
            ]
            rows = repaired + rows
    return rows


def _focused_paper_lines(ctx, paper_id, plan, *, page_k: int) -> list[str]:
    """Expose a few query-relevant pages from one already-selected paper.

    The answerer historically saw only the short emitted evidence quote even
    though table replay had already parsed the complete selected PDF. BM25 page
    shortlisting expands context without changing paper selection or flooding
    the model with the whole document. Original page numbers are preserved.
    """
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(paper_id)
    if parsed is None or not getattr(parsed, "pages", None):
        return []
    from littraceqa.localize.focused import shortlist_parsed_pdf

    expected = " ".join(
        str(value)
        for key in plan.expected_keys
        for value in key
        if value is not None
    )
    columns = " ".join(
        [*plan.row_key_cols, *(column["name"] for column in plan.value_cols)]
    )
    query = " ".join((ctx.question, expected, columns))
    focused = shortlist_parsed_pdf(
        query, parsed, top_k=page_k, max_pages=page_k
    )
    return [
        f"[paper_text p.{page.page}] "
        f"{str(page.text or '')[:_MAX_TABLE_TEXT_PAGE_CHARS]}".rstrip()
        for page in focused.pages
        if str(page.text or "").strip()
    ]


def _text_rows(
    ctx,
    paper_id,
    plan,
    llm,
    *,
    own_paper_only: bool = False,
    context_mode: str = "evidence",
    page_k: int = 3,
) -> list[dict]:
    # Include any item with a quote OR an object_id: an equation/citation item
    # can carry its answer entirely in `object_id` with an empty quote.
    items = [ev for ev in _nonvisual_evidence(ctx.evidence, paper_id)
             if (ev.quote or ev.object_id)]
    if context_mode not in {"evidence", "focused_pages"}:
        raise ValueError("context_mode must be 'evidence' or 'focused_pages'")
    if isinstance(page_k, bool) or not isinstance(page_k, int) or not 1 <= page_k <= 8:
        raise ValueError("page_k must be an integer from 1 to 8")
    if not items and context_mode == "evidence":
        return []
    lines = [_evidence_line(ev) for ev in items]
    if context_mode == "focused_pages":
        lines.extend(_focused_paper_lines(ctx, paper_id, plan, page_k=page_k))
    if not lines:
        return []
    paper_title = str(
        (getattr(ctx, "paper_titles", None) or {}).get(paper_id) or ""
    )
    try:
        prompt = _build_text_prompt(
            ctx.question,
            plan,
            lines,
            own_paper_only=own_paper_only,
            paper_title=paper_title,
        )
        resp = llm.complete(prompt, system=_TEXT_SYSTEM, temperature=0.0)
    except Exception:  # noqa: BLE001 -- text failure yields no rows, never raises
        return []
    rows = _parse_rows(resp, plan)

    # One broad call can silently omit a requested component even when the
    # planner and evidence both contain it. Retry each omitted, paper-relevant
    # planned key once with a single-row contract. This is bounded by the
    # question's explicit row cardinality and never asks an unrelated paper for
    # another paper's named method.
    if plan.expected_keys:
        seen = {
            tuple(normalize_text(row.get(column)) for column in plan.row_key_cols)
            for row in rows
        }
        haystack = " ".join(
            [str((getattr(ctx, "paper_titles", None) or {}).get(paper_id) or "")]
            + lines
        )
        relevant = [
            expected
            for expected in plan.expected_keys
            if _expected_key_relevant(expected, haystack)
        ]
        for expected in plan.expected_keys:
            normalized = tuple(normalize_text(value) for value in expected)
            if normalized in seen or not _retry_key_for_paper(expected, relevant):
                continue
            one_row_plan = replace(plan, expected_keys=[expected])
            retry_lines = _retry_evidence_lines(
                ctx, paper_id, expected, lines
            )
            try:
                retry = llm.complete(
                    _build_text_prompt(
                        ctx.question, one_row_plan, retry_lines
                    ),
                    system=_TEXT_SYSTEM,
                    temperature=0.0,
                )
            except Exception:  # noqa: BLE001 -- one omitted row stays omitted
                continue
            retry_rows = _parse_rows(retry, one_row_plan)
            for row in retry_rows:
                key = tuple(
                    normalize_text(row.get(column))
                    for column in plan.row_key_cols
                )
                if key != normalized or key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows = _repair_complexity_ocr(rows, ctx.question, plan)
    rows = _repair_symbol_cells(rows, ctx, paper_id, plan)
    return rows


def _repair_complexity_ocr(rows, question, plan):
    """Undo two deterministic superscript artifacts in complexity prose.

    PyMuPDF flattens ``K²`` to ``K2`` and can splice ``(p²)`` into the phrase
    ``16 times smaller``. Preserve arbitrary strings everywhere else; these
    repairs apply only to an explicitly requested complexity/relative-size
    answer and only to the narrow OCR shapes observed in the source PDFs.
    """
    normalized_question = normalize_text(question)
    if "complexity" not in normalized_question and "relative" not in normalized_question:
        return rows
    value_names = [column["name"] for column in plan.value_cols]
    out = []
    for row in rows:
        repaired = dict(row)
        for name in value_names:
            value = repaired.get(name)
            if not isinstance(value, str):
                continue
            if "complexity" in normalized_question:
                value = re.sub(
                    r"\bO\(\s*([A-Za-z])\s*\^?\s*2\s*\)",
                    r"O(\1^2)",
                    value,
                )
            value = re.sub(
                r"^(\d+(?:\.\d+)?)\s*\([^)]*\)\s*(times smaller)$",
                r"\1 \2",
                value,
                flags=re.IGNORECASE,
            )
            repaired[name] = value
        out.append(repaired)
    return out


def _repair_symbol_cells(rows, ctx, paper_id, plan):
    """Return a requested notation symbol, not the collection it names.

    A model can copy ``{b1, ..., bK}`` when the question asks what *symbol*
    denotes that vocabulary. Search the already-selected paper for the explicit
    assignment ``B = {b1, ...}`` and replace the set contents only when exactly
    one left-hand symbol is supported. This is deterministic and source-bound;
    ambiguous assignments are left untouched.
    """
    if "symbol" not in normalize_text(ctx.question):
        return rows
    symbol_columns = [
        column["name"]
        for column in plan.value_cols
        if "symbol" in normalize_text(column.get("name"))
    ]
    if not symbol_columns:
        return rows
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(paper_id)
    source = "\n".join(
        getattr(page, "text", "") or ""
        for page in (getattr(parsed, "pages", None) or [])
    )
    labeled_candidates = set(re.findall(
        r"\b(?:candidate\s+arms?|vocab(?:ulary)?)\s+"
        r"(?:[^\w\s]\s*){0,3}([A-Za-z])"
        r"\s*(?::=|=)\s*\{",
        source,
        flags=re.IGNORECASE,
    ))
    if (
        not rows
        and len(labeled_candidates) == 1
        and getattr(getattr(plan, "row_axis", None), "value", None) == "paper"
        and len(plan.row_key_cols) == 1
    ):
        title = (getattr(ctx, "paper_titles", None) or {}).get(paper_id)
        if title:
            rows = [{
                plan.row_key_cols[0]: title,
                **{column["name"]: None for column in plan.value_cols},
            }]
    out = []
    for row in rows:
        repaired = dict(row)
        for name in symbol_columns:
            value = repaired.get(name)
            if value is None and len(labeled_candidates) == 1:
                repaired[name] = next(iter(labeled_candidates))
                continue
            if not isinstance(value, str) or "{" not in value:
                continue
            first_item = re.search(r"\{\s*([A-Za-z])\s*[_^]?\s*1\b", value)
            if first_item is None:
                continue
            item = re.escape(first_item.group(1))
            candidates = set(re.findall(
                rf"\b([A-Za-z])\s*(?::=|=)\s*\{{\s*{item}\s*[_^]?\s*1\b",
                source,
            ))
            if len(candidates) == 1:
                repaired[name] = next(iter(candidates))
        out.append(repaired)
    return out


def _expected_key_relevant(expected, haystack: str) -> bool:
    source = normalize_text(haystack)
    source_tokens = set(re.findall(r"[a-z0-9]+", source))
    key_tokens = _expected_key_tokens(expected)
    if any(token in source_tokens for token in key_tokens):
        return True

    # PDF extraction occasionally inserts spaces inside model names. Token
    # equality can never match that
    # representation, even though the paper identity is unambiguous.
    compact_source = re.sub(r"[^a-z0-9]+", "", source)
    return any(token in compact_source for token in key_tokens)


def _expected_key_tokens(expected) -> list[str]:
    return [
        token
        for value in expected
        for token in re.findall(r"[a-z0-9]+", normalize_text(value))
        if len(token) >= 4 and token not in _RETRY_STOPWORDS
    ]


def _retry_key_for_paper(expected, relevant) -> bool:
    """Retry a missing key only when its requested group belongs here.

    A question can request several components of one method. Only one component
    name may occur in the short localized quote. Once one key anchors that
    group to this paper, its sibling keys are safe to retry; unrelated methods
    remain excluded.
    """
    expected_tokens = set(_expected_key_tokens(expected))
    return any(
        expected_tokens.intersection(_expected_key_tokens(anchor))
        for anchor in relevant
    )


def _retry_evidence_lines(ctx, paper_id, expected, localized_lines):
    """Augment a targeted retry with the best page from the selected paper.

    Evidence locators are optimized for the official evidence metric and can be
    near, rather than on, an appendix setting. We already parsed the selected
    PDF, so answering should not pretend the rest of
    that known paper is unavailable.  Rank pages deterministically by requested
    key tokens and adjacent phrases, then add one bounded excerpt.  Submitted
    evidence is untouched.
    """
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(paper_id)
    pages = getattr(parsed, "pages", None) or []
    if not pages:
        return localized_lines

    words = [
        token
        for value in expected
        for token in re.findall(r"[a-z0-9]+", normalize_text(value))
        if len(token) >= 3 and token not in {"and", "for", "from", "the", "with"}
    ]
    unique_words = list(dict.fromkeys(words))
    phrases = [
        f"{words[i]} {words[i + 1]}"
        for i in range(len(words) - 1)
        if words[i] != words[i + 1]
    ]
    ranked = []
    for position, page in enumerate(pages):
        text = getattr(page, "text", "") or ""
        normalized = normalize_text(text)
        token_set = set(re.findall(r"[a-z0-9]+", normalized))
        token_hits = sum(token in token_set for token in unique_words)
        phrase_hits = sum(phrase in normalized for phrase in phrases)
        score = 10 * token_hits + 25 * phrase_hits
        if score:
            ranked.append((-score, position, page, normalized))
    if not ranked:
        return localized_lines
    _score, _position, page, normalized = min(ranked)
    excerpt = _focused_page_excerpt(
        getattr(page, "text", "") or "", normalized, unique_words, phrases
    )
    if not excerpt:
        return localized_lines
    page_number = getattr(page, "page", "?")
    return list(localized_lines) + [
        f"[selected-paper full-text search p.{page_number}] {excerpt}"
    ]


def _focused_page_excerpt(text, normalized, words, phrases):
    """Return a bounded region around the first/last matching search terms."""
    if len(text) <= _MAX_RETRY_PAGE_CHARS:
        return text
    lowered = text.casefold()
    needles = [phrase for phrase in phrases if phrase in normalized]
    needles.extend(word for word in words if word in normalized)
    positions = [lowered.find(needle) for needle in needles]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return text[:_MAX_RETRY_PAGE_CHARS]
    center = sum(positions) // len(positions)
    start = max(0, center - _MAX_RETRY_PAGE_CHARS // 2)
    end = min(len(text), start + _MAX_RETRY_PAGE_CHARS)
    start = max(0, end - _MAX_RETRY_PAGE_CHARS)
    return text[start:end]


def _select_open_ended_owner_row(ctx, paper_id, plan, rows) -> list[dict]:
    """Keep one source-owned row for an open-ended multi-paper table.

    Prompts carry the semantic contract; this deterministic cap is the safety
    net when a model still copies an entire comparison table. Prefer a row whose
    key identifies this paper by title/early-page routing, then an explicit
    ``ours`` row, then the model's first row. The fallback is intentionally
    recall-preserving: a paper that has no machine-detectable method alias still
    contributes one candidate rather than none.
    """
    if not rows:
        return []
    scored = []
    for position, row in enumerate(rows):
        key = tuple(
            str(row.get(column) or "").strip() for column in plan.row_key_cols
        )
        route = route_expected_keys(ctx, [key])[0]
        source_score = int(route.scores.get(paper_id, 0))
        normalized_key = " ".join(normalize_text(value) for value in key)
        own_marker = bool(
            re.search(r"\b(?:ours?|our method|proposed)\b", normalized_key)
        )
        score = source_score + (8_000 if own_marker else 0)
        scored.append((-score, position, row))
    selected = dict(min(scored)[2])
    _canonicalize_owner_identity(selected, ctx, paper_id, plan)
    return [selected]


def _canonicalize_owner_identity(row, ctx, paper_id, plan) -> None:
    """Use a concise identifier at the start of the owning paper's title.

    Open-ended questions ask for the method introduced by each paper. Vision
    tables often label that row ``Ours`` or an implementation variant instead
    of the paper's method name. A short identifier before ``:``/``---`` is
    authoritative metadata, so use it for a method-like first row-key column.
    Descriptive titles without a separator are left untouched.
    """
    if not plan.row_key_cols:
        return
    key = plan.row_key_cols[0]
    if normalize_text(key) not in {"method", "methods", "method name", "model", "system"}:
        return
    title = str(
        (getattr(ctx, "paper_titles", None) or {}).get(paper_id) or ""
    ).strip()
    head = re.split(r"\s*(?::|---)\s*", title, maxsplit=1)[0].strip()
    if not head or head == title or len(re.findall(r"[A-Za-z0-9]+", head)) > 4:
        return
    identifier_like = (
        "-" in head
        or any(character.isdigit() for character in head)
        or any(character.isupper() for character in head[1:])
    )
    if not identifier_like:
        return
    head = unicodedata.normalize("NFKC", head)
    version = re.fullmatch(r"([A-Z][A-Z0-9-]+)\s+(\d+(?:\.\d+)+)", head)
    if version:
        head = f"{version.group(1)}-{version.group(2)}"
    row[key] = head


def _canonicalize_owner_base_model(row, ctx, paper_id, plan) -> None:
    """Prefer a benchmark-coupled base model stated by the owning paper.

    Comparison tables can offer several model sizes and the selected visual row
    is not necessarily the one evaluated by the method named in the question.
    Override only for a ``Base Model`` row-key question and only when the
    selected paper contains one narrow source construction tied to the
    requested benchmark, such as ``inference scaling with <model> ... GenEval``
    or ``VAR model <model> ... GenEval``. This is stricter than asking the LLM
    to choose among all model names in the paper.
    """
    if "base model" not in normalize_text(ctx.question):
        return
    columns = [
        column
        for column in plan.row_key_cols
        if normalize_text(column) == "base model"
    ]
    if len(columns) != 1:
        return
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(paper_id)
    source = re.sub(
        r"\s+",
        " ",
        "\n".join(
            getattr(page, "text", "") or ""
            for page in (getattr(parsed, "pages", None) or [])
        ),
    )
    if not source:
        return

    patterns = (
        r"inference[- ]time\s+scal(?:ing|ed).{0,120}?\busing\s+"
        r"([A-Za-z][A-Za-z0-9._-]*(?:\s+\d+(?:\.\d+)?[BM])?"
        r"(?:\s+v\d+)?)\s+as\s+(?:the\s+)?base\s+model.{0,120}?\bGenEval\b",
        r"inference\s+scaling\s+with\s+(?:the\s+)?"
        r"([A-Za-z][A-Za-z0-9._-]*(?:\s+\d+(?:\.\d+)?[BM])?"
        r"(?:\s+v\d+)?)\s+model\b.{0,120}?\bGenEval\b",
        r"(?:powerful\s+)?(?:VAR\s+)?model\s+"
        r"([A-Za-z][A-Za-z0-9._-]*\d+(?:\.\d+)?[BM])\s+"
        r"show(?:s|ed)?\b.{0,120}?\bGenEval\b",
    )
    candidates = {
        _normalize_base_model_value(match.group(1))
        for pattern in patterns
        for match in re.finditer(pattern, source, flags=re.IGNORECASE)
    }
    if len(candidates) == 1:
        row[columns[0]] = next(iter(candidates))


def _synthesize_owner_equation_row(ctx, paper_id, plan) -> list[dict]:
    """Recover an omitted method/objective row from explicit paper notation.

    Equation-heavy papers mention many baselines before their own objective, so
    a broad extraction call can return no row. For an open-ended training-
    objective table, identify exactly one acronym introduced by the paper and
    then take the first numbered equation whose loss symbol is ``L<acronym>``.
    Ambiguous acronyms or absent notation leave the output unchanged.
    """
    if "training objective" not in normalize_text(ctx.question):
        return []
    method_columns = [
        column
        for column in plan.row_key_cols
        if normalize_text(column) in {"method", "method name"}
    ]
    equation_columns = [
        column["name"]
        for column in plan.value_cols
        if "equation" in normalize_text(column.get("name"))
        and "id" in normalize_text(column.get("name"))
    ]
    if len(method_columns) != 1 or len(equation_columns) != 1:
        return []
    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(paper_id)
    source = re.sub(
        r"\s+",
        " ",
        "\n".join(
            getattr(page, "text", "") or ""
            for page in (getattr(parsed, "pages", None) or [])
        ),
    )
    if not source:
        return []

    introduced = set(re.findall(
        r"\b(?:we|this\s+paper,?\s+we)\s+"
        r"(?:propose|introduce)\s+[^.]{2,140}?"
        r"\(([A-Z][A-Z0-9-]{1,12})\)",
        source,
        flags=re.IGNORECASE,
    ))
    introduced.update(re.findall(
        r"\b(?:we|this\s+paper,?\s+we)\s+"
        r"(?:propose|introduce)\s+([A-Z][A-Z0-9-]{1,12})\b",
        source,
        flags=re.IGNORECASE,
    ))
    introduced = {
        method.upper(): method
        for method in introduced
        if method == method.upper()
    }
    if len(introduced) != 1:
        return []
    method = next(iter(introduced.values()))
    equation = re.search(
        rf"\bL\s*{re.escape(method)}\s*\(.{{0,900}}?"
        r"\(\s*(\d+[a-z]?)\s*\)",
        source,
        flags=re.IGNORECASE,
    )
    if equation is None:
        return []
    return [{
        method_columns[0]: method,
        equation_columns[0]: f"Equation {equation.group(1)}",
    }]


def extract_rows_from_paper(
    ctx,
    paper_id,
    plan,
    vision_llm,
    llm,
    *,
    own_paper_only: bool = False,
    visual_extraction_mode: str = "direct",
    visual_retry_expected_keys: list[tuple[str, ...]] | None = None,
    visual_consensus_repeats: int = 1,
    extraction_sources: str = "both",
    text_context_mode: str = "evidence",
    text_page_k: int = 3,
) -> list[dict]:
    """This paper's contributed rows, from its VISUAL locators (vision) AND its
    non-visual evidence quotes (text), merged. Constrained to `plan.expected_keys`
    via the prompts. Never raises."""
    if extraction_sources not in {"both", "text_only", "visual_only"}:
        raise ValueError(
            "extraction_sources must be 'both', 'text_only', or 'visual_only'"
        )
    rows: list[dict] = []
    if extraction_sources != "text_only":
        rows.extend(_visual_rows(
            ctx,
            paper_id,
            plan,
            vision_llm,
            own_paper_only=own_paper_only,
            extraction_mode=visual_extraction_mode,
            retry_expected_keys=visual_retry_expected_keys,
            consensus_repeats=visual_consensus_repeats,
        ))
    if extraction_sources != "visual_only":
        rows.extend(_text_rows(
            ctx,
            paper_id,
            plan,
            llm,
            own_paper_only=own_paper_only,
            context_mode=text_context_mode,
            page_k=text_page_k,
        ))
    if own_paper_only:
        if not rows:
            rows = _synthesize_owner_equation_row(ctx, paper_id, plan)
        selected = _select_open_ended_owner_row(ctx, paper_id, plan, rows)
        for row in selected:
            _canonicalize_owner_base_model(row, ctx, paper_id, plan)
        return selected
    return rows
