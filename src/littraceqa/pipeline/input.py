"""Input-record parser.

Defensively parses input records, extracting query metadata, question text,
and optional answer scaffolding (multiple choice options, table schema).

Never raises on missing/oddly-shaped fields — absent paths return None or
default values, ensuring the pipeline is resilient to input variance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputRecord:
    """Parsed input record containing query metadata and optional scaffold data.

    Fields:
    - query_id: Unique identifier for the query
    - question: The question text
    - answer_types: List of expected answer types (defaults to [])
    - mc_options: Normalized multiple-choice label -> text mapping if present,
      else None.  Both the current organizer list shape and the legacy dict
      shape normalize to this one downstream contract.
    - table_schema: Table schema list if present, else None
    """
    query_id: str
    question: str
    answer_types: list[str]
    mc_options: dict[str, str] | None
    table_schema: list[dict] | None


def _normalize_mc_options(raw: object) -> dict[str, str] | None:
    """Normalize organizer/legacy MC scaffolds to ``{label: text}``.

    Current released files use ``[{"label": "A", "text": "..."}]``;
    older local files used ``{"A": "..."}``.  Malformed entries are ignored
    independently so one bad option cannot drop the query.  Labels are
    upper-cased because the official validator compares them case-insensitively.
    """
    pairs: list[tuple[object, object]] = []
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        pairs = [
            (item.get("label"), item.get("text"))
            for item in raw
            if isinstance(item, dict)
        ]
    else:
        return None

    normalized: dict[str, str] = {}
    for raw_label, raw_text in pairs:
        if not isinstance(raw_label, str) or not raw_label.strip():
            continue
        if not isinstance(raw_text, str):
            # The official output validator derives legal labels without
            # inspecting option text. Preserve a valid label even when a
            # malformed input omitted/coerced its text, or fallback-A could
            # become validator-invalid for a sole label-B option.
            raw_text = "" if raw_text is None else str(raw_text)
        label = raw_label.strip().upper()
        # Keep the first occurrence: silently overwriting a duplicate label
        # would make the chosen option depend on malformed input order.
        normalized.setdefault(label, raw_text)
    return normalized or None


def parse_input_record(rec: dict) -> InputRecord:
    """Parse a raw input record into a defensive InputRecord.

    Defensively extracts:
    - query_id and question (required, pass-through)
    - answer_types from rec["answer_types"] (defaults to [])
    - mc_options: PREFER the real input file's TOP-LEVEL `multiple_choice_options`
      (the shape in data/validation_inputs.jsonl and the Aug-3 test); fall back
      to the gold-file NESTED `answer.multiple_choice.options`. None if neither.
    - table_schema: PREFER top-level `table_schema`; fall back to nested
      `answer.table.schema`. None if neither.

    The top-level shape is authoritative for the test input; the nested fallback
    keeps the parser working on the gold-annotated `validation.jsonl` shape too.
    Never raises on missing or oddly-shaped fields.
    """
    # Core fields: pass-through
    query_id = rec.get("query_id", "")
    question = rec.get("question", "")

    # answer_types defaults to []
    answer_types = rec.get("answer_types", [])
    if not isinstance(answer_types, list):
        answer_types = []

    answer = rec.get("answer")

    # MC options: top-level `multiple_choice_options` (real input) wins; else
    # nested `answer.multiple_choice.options` (gold shape). Current organizer
    # files use a list of {label,text}; legacy files use a dict.
    mc_options = None
    top_mc = rec.get("multiple_choice_options")
    if isinstance(top_mc, (dict, list)):
        mc_options = _normalize_mc_options(top_mc)
    elif isinstance(answer, dict):
        multiple_choice = answer.get("multiple_choice")
        if isinstance(multiple_choice, dict):
            mc_options = _normalize_mc_options(multiple_choice.get("options"))

    # Table schema: top-level `table_schema` (real input) wins; else nested
    # `answer.table.schema` (gold shape). Must be a list.
    table_schema = None
    top_ts = rec.get("table_schema")
    if isinstance(top_ts, list):
        table_schema = top_ts
    elif isinstance(answer, dict):
        table = answer.get("table")
        if isinstance(table, dict):
            schema = table.get("schema")
            if isinstance(schema, list):
                table_schema = schema

    return InputRecord(
        query_id=query_id,
        question=question,
        answer_types=answer_types,
        mc_options=mc_options,
        table_schema=table_schema,
    )
