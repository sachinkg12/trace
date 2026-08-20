"""Table answering: reconstructs `rows` (`list[dict]` keyed by the GOLD
schema's column names -- the scorer matches rows by the schema's
`is_row_key` columns via `normalize_text` and compares numeric cells with
`math.isclose`), covering two distinct table shapes.

1. Paper-LIST table (no non-row-key columns): one row per `ctx.paper_ids`,
   `{row_key_col: ctx.paper_titles[pid]}` using the VERBATIM metadata
   title. Deterministic -- no LLM call, confidence 1.0, `attested_evidence`
   is `[]` (a paper-set-derived row has no evidence cell to attest).
2. Metric table (schema has value columns): one LLM call per paper/row,
   defensively parsed exactly like `seed/anchor.py` / `answer/freeform.py`
   / `answer/multiple_choice.py` (strip code fences, `json.loads`, bail to
   an all-None row on any failure -- never raise). Numeric columns are
   coerced via `_coerce_number` (int/float or None, NEVER a string and
   NEVER 0/"" for a missing value); string columns are kept verbatim.
   Confidence is the fraction of rows with >=1 grounded cell.

Rows are de-duplicated by row-key (last write wins) so a paper that
appears twice in `ctx.paper_ids` doesn't double a row.
"""

from __future__ import annotations

import json
import re

from littraceqa.answer.grounding import ground_value
from littraceqa.answer.interfaces import AnswerContext, StrategyOutput, register_strategy
from littraceqa.answer.vision import render_evidence_png, _trim_answer

_DEFAULT_ROW_KEY_COL = "Paper Title"

# Source types whose row lives in the printed table pixels, not the
# column-major-scrambled text PyMuPDF extracts. The presence of such an
# evidence item for a paper is what triggers the VISION row path.
_VISUAL_SOURCE_TYPES = ("table", "figure")

_VISION_SYSTEM = (
    "You are reading an image of a table (or figure) from ONE academic paper "
    "to fill a single row of a comparison table. Read the row and column "
    "headers and the printed numbers carefully. Report values ONLY from what "
    "is visibly printed in the image -- never guess, compute, or use outside "
    "knowledge. Copy the number for the requested cell EXACTLY as printed "
    "(keep the printed precision, e.g. 2.05 or 96.0). Respond with STRICT "
    "minified JSON only -- no prose, no markdown, no code fences -- with "
    "exactly one key per requested column, keyed by the column name as given."
)

_SYSTEM_PROMPT = (
    "You extract one table row about a single paper from academic-paper "
    "evidence quotes. Use ONLY the evidence provided -- do not guess or "
    "compute values that are not stated. Copy string values VERBATIM; for "
    "numeric values, respond with a bare JSON number (never a quoted "
    'string). If a value is not present in the evidence, use JSON null. '
    "Respond with STRICT minified JSON only -- no prose, no markdown, no "
    "code fences. The JSON object must have exactly one key per requested "
    "column, keyed by the column name exactly as given."
)

# Matches an optional ```json / ``` opening fence and a trailing ``` closing
# fence, so a fenced response (however the model chooses to wrap it) is
# stripped down to the bare JSON body before parsing. Mirrors
# seed/anchor.py / answer/freeform.py / answer/multiple_choice.py.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


# Uncertainty notation: papers print "44.5 ± 0.6" (spaces around the sign) but
# the gold cell is "44.5±0.6"; the scorer's `normalize_text` keeps those
# spaces, so a value the model reads CORRECTLY is zeroed by the spacing alone.
# Collapse whitespace around ± so a right value matches the gold's shape.
# (For example, normalize DEDA "44.5 ± 0.6" to "44.5±0.6".)
_PM_SPACING_RE = re.compile(r"\s*±\s*")


def _clean_cell(value: str) -> str:
    """Trim wrapping quotes/whitespace (`_trim_answer`) then collapse the
    spaces papers put around the ± uncertainty sign, so a correctly-read
    metric cell matches the gold cell's exact shape."""
    return _PM_SPACING_RE.sub("±", _trim_answer(value))


def _row_key_columns(schema: list[dict] | None) -> list[str]:
    """Column name(s) marked `is_row_key: True`. Best-effort single key
    `"Paper Title"` when the schema is absent entirely."""
    if not schema:
        return [_DEFAULT_ROW_KEY_COL]
    keys = [col["name"] for col in schema if col.get("is_row_key")]
    return keys or [_DEFAULT_ROW_KEY_COL]


def _value_columns(schema: list[dict] | None) -> list[dict]:
    """The non-row-key columns (the rest of the schema)."""
    if not schema:
        return []
    return [col for col in schema if not col.get("is_row_key")]


def _coerce_number(x: object) -> int | float | None:
    """Coerce a JSON-parsed cell into a number, defensively -- int/float
    or None, NEVER raises. `bool` is an `int` subclass in Python, but a
    JSON `true`/`false` in a numeric column is not a usable number, so it
    maps to None rather than 1/0."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _build_row_prompt(ctx: AnswerContext, paper_id: str, row_cols: list[dict]) -> str:
    col_names = ", ".join(col["name"] for col in row_cols)
    lines = [f"Question: {ctx.question}", "", f"Columns to extract: {col_names}", "",
             "Evidence (for this paper only):"]
    for ev in ctx.evidence:
        if ev.paper_id != paper_id:
            continue
        where = f"{ev.paper_id} p.{ev.page}"
        if ev.object_id:
            where += f" ({ev.object_id})"
        lines.append(f"- [{where}] {ev.quote}")
    lines.append("")
    lines.append("Respond with the JSON object only.")
    return "\n".join(lines)


def _build_vision_prompt(ctx: AnswerContext, ev, row_cols: list[dict],
                         row_key_col: str) -> str:
    """Prompt the vision model to read THIS paper's table image and return the
    one row for the method the paper contributes to the question. The row-key
    is requested as the short name the QUESTION uses (so it matches the gold
    row key), while every value cell is copied verbatim from the printed
    table."""
    where = ev.object_id or "the table"
    lines = [f"Question: {ctx.question}", "",
             f"This image is {where} on page {ev.page} of one paper. Read it "
             "and extract the single row for the method/model this paper "
             "contributes to the question.", "",
             "Return STRICT minified JSON with exactly these keys:",
             f'- "{row_key_col}": the method/model/dataset name. Use the short '
             "name as it is referred to in the question above (drop qualifiers "
             'the table itself adds such as "(ours)" or iteration counts).']
    for col in row_cols:
        if col["name"] == row_key_col:
            continue
        lines.append(
            f'- "{col["name"]}": the value for that method exactly as printed '
            "in the table (keep the printed precision), or null if it is not "
            "present in the table.")
    lines.append("")
    lines.append("Respond with the JSON object only.")
    return "\n".join(lines)


def _parse_row(response: str, row_cols: list[dict], row_key_names: set[str]) -> dict:
    """Defensively parse the LLM's row JSON into a dict of cells keyed by
    column name -- every requested column is present, missing/unparseable
    values become None (never 0/""), numeric VALUE columns are coerced via
    `_coerce_number`, and string columns are kept verbatim.

    A row-key column is NEVER number-coerced even if its schema types it
    "number": the scorer matches rows by `normalize_text(row.get(key))` (a
    string compare), so coercing a row-key `"14.70"` to float `14.70` would
    normalize to `"14.7"` and silently mismatch the gold key (row_f1 = 0).
    Row-keys therefore always go through the verbatim-string branch."""
    cleaned = _strip_code_fences(response or "")
    parsed = None
    if cleaned:
        try:
            candidate = json.loads(cleaned)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            parsed = candidate

    row: dict = {}
    for col in row_cols:
        name = col["name"]
        raw = parsed.get(name) if parsed else None
        if col.get("type") == "number" and name not in row_key_names:
            row[name] = _coerce_number(raw)
        else:
            row[name] = raw if isinstance(raw, str) else None
    return row


def _visual_evidence_for(evidence: list, paper_id: str):
    """First table/figure evidence item for `paper_id` (the page to render),
    or None."""
    for ev in evidence or []:
        if ev.paper_id == paper_id and ev.source_type in _VISUAL_SOURCE_TYPES:
            return ev
    return None


def _vision_row(ctx: AnswerContext, ev, png: bytes, row_cols: list[dict],
                row_key_col: str, vision_llm) -> dict | None:
    """Ask the vision model to read the rendered table image and return this
    paper's row cells. Numeric value cells are coerced via `_parse_row`; string
    cells and the row-key are `_trim_answer`-cleaned. Returns None (caller falls
    back to the text path) on any error, an empty reply, or an all-null row --
    never raises."""
    prompt = _build_vision_prompt(ctx, ev, row_cols, row_key_col)
    try:
        response = vision_llm.complete(prompt, system=_VISION_SYSTEM,
                                       temperature=0.0, images=[png])
    except Exception:  # noqa: BLE001 -- any vision failure degrades to text path
        return None
    if not (response and response.strip()):
        return None
    cells = _parse_row(response, row_cols, {row_key_col})
    for col in row_cols:
        value = cells.get(col["name"])
        if isinstance(value, str):
            cells[col["name"]] = _clean_cell(value) or None
    if not any(cells.get(col["name"]) is not None for col in row_cols):
        return None
    return cells


def _text_row(ctx: AnswerContext, paper_id: str, row_cols: list[dict],
              row_key_col: str) -> dict:
    """Legacy text path: one LLM call over the paper's evidence quotes."""
    prompt = _build_row_prompt(ctx, paper_id, row_cols)
    try:
        response = ctx.llm.complete(prompt, system=_SYSTEM_PROMPT, temperature=0.0)
    except Exception:  # noqa: BLE001 -- any LLM failure degrades, never crashes
        response = ""
    return _parse_row(response, row_cols, {row_key_col})


def _row_for_paper(ctx: AnswerContext, paper_id: str, row_cols: list[dict],
                   row_key_col: str, vision_llm) -> dict:
    """This paper's row cells: VISION over the rendered table/figure page when
    one is available and yields a row, else the text-quote extraction path."""
    ev = _visual_evidence_for(ctx.evidence, paper_id)
    if ev is not None:
        # Render from the runner-retained PDF bytes for this paper when present
        # (GCS-safe shared PDF source), else fall back to the on-disk cache.
        pdf_bytes = (getattr(ctx, "pdf_bytes_by_id", None) or {}).get(paper_id)
        png = render_evidence_png(ev, pdf_bytes=pdf_bytes)
        if png is not None:
            cells = _vision_row(ctx, ev, png, row_cols, row_key_col, vision_llm)
            if cells is not None:
                return cells
    return _text_row(ctx, paper_id, row_cols, row_key_col)


@register_strategy("table")
class TableAnswerer:
    answer_type = "table"

    def answer(self, ctx: AnswerContext) -> StrategyOutput:
        try:
            return self._answer(ctx)
        except Exception:  # noqa: BLE001 -- never raise; empty rows on total failure
            return StrategyOutput(value=[], confidence=0.0, attested_evidence=[])

    def _answer(self, ctx: AnswerContext) -> StrategyOutput:
        row_key_cols = _row_key_columns(ctx.table_schema)
        row_key_col = row_key_cols[0]
        value_cols = _value_columns(ctx.table_schema)

        if not value_cols:
            return self._paper_list(ctx, row_key_col)
        return self._metric_table(ctx, row_key_col, value_cols)

    @staticmethod
    def _paper_list(ctx: AnswerContext, row_key_col: str) -> StrategyOutput:
        """Deterministic paper-set table: one row per paper with a known
        title, verbatim from `ctx.paper_titles`. No LLM call. Rows are
        de-duplicated by row-key (last write wins)."""
        rows: dict[str, dict] = {}
        for pid in ctx.paper_ids:
            title = ctx.paper_titles.get(pid)
            if not title:
                continue
            rows[title] = {row_key_col: title}
        return StrategyOutput(value=list(rows.values()), confidence=1.0, attested_evidence=[])

    @staticmethod
    def _metric_table(ctx: AnswerContext, row_key_col: str,
                       value_cols: list[dict]) -> StrategyOutput:
        """One row per paper. When the paper has a table/figure evidence
        item, its cited page is RENDERED to PNG and read by the VISION model
        (`ctx.vision_llm or ctx.llm`): PyMuPDF scrambles dense metric tables
        column-major, so the values are only recoverable from the pixels.
        Falls back to the text-quote extraction path when there is no image to
        render, the render fails, or the vision call returns nothing -- never
        crashing. The row-key column is requested alongside the value columns,
        falling back to the paper's metadata title if none is supplied.
        `attested_evidence` grounds the row-key value; confidence is the
        fraction of rows with >=1 grounded/non-null value cell."""
        row_key_spec = next((col for col in (ctx.table_schema or [])
                             if col["name"] == row_key_col),
                            {"name": row_key_col, "type": "string", "is_row_key": True})
        row_cols = [row_key_spec] + value_cols
        vision_llm = ctx.vision_llm or ctx.llm

        rows: dict[str, dict] = {}
        attested: list = []
        grounded_row_count = 0
        total_rows = 0

        for pid in ctx.paper_ids:
            cells = _row_for_paper(ctx, pid, row_cols, row_key_col, vision_llm)
            row_key_value = cells.get(row_key_col)
            if not isinstance(row_key_value, str) or not row_key_value:
                row_key_value = ctx.paper_titles.get(pid)
            if not row_key_value:
                continue

            row = {row_key_col: row_key_value}
            for col in value_cols:
                row[col["name"]] = cells.get(col["name"])

            total_rows += 1
            row_attested = ground_value(row_key_value, ctx.evidence, ctx.parsed_by_id)
            # Consider VALUE columns only: the row-key cell falls back to the
            # paper title and is thus ALWAYS non-None, so counting it would let
            # a totally failed LLM report confidence ~= 1.0 and spuriously open
            # the pipeline's confidence -> evidence precision gate.
            has_cell = any(row.get(col["name"]) is not None for col in value_cols)
            if row_attested or has_cell:
                grounded_row_count += 1
            attested.extend(row_attested)
            rows[row_key_value] = row

        confidence = (grounded_row_count / total_rows) if total_rows else 0.0
        return StrategyOutput(value=list(rows.values()), confidence=confidence,
                              attested_evidence=attested)
