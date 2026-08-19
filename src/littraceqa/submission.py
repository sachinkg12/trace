"""Submission assembler + validator.

Assembles per-query prediction lines into the exact shape of the official
`data/sample_submission.jsonl` and validates them before they hit disk.

Scorer-contract landmine encoded here (verified against the real sample):
the submission shape follows `data/sample_submission.jsonl`
(`query_id`, `gold_papers` as `[{"paper_id": ...}]`, `evidence`, `answer`),
NOT the stricter `schema.json` — the two contradict, and the sample wins
because that's what the evaluator actually consumes.

`validate_line` rejects any key not in `allowed_keys` so we never submit a
stray field (e.g. `role`, `justification`) that would trip the evaluator's
`additionalProperties` checks. `write_submission` refuses to write a file
that is missing predictions for any input `query_id`, since a missing
prediction scores 0 precision for that query. It also validates every line
against `allowed_keys` before writing anything, so the additionalProperties
landmine is enforced at the write boundary rather than being opt-in.
"""

import json
import math
import os
import pathlib
import tempfile
from typing import Any

from littraceqa.pipeline.input import InputRecord

# The exact key set `build_line` emits, mirrored from the official
# `data/sample_submission.jsonl` shape. Single source of truth for the
# submission-line contract (parallel to `scorer.METRIC_NAMES`) — callers and
# tests should reference this rather than re-declaring the key set.
SAMPLE_KEYS = frozenset({"query_id", "gold_papers", "evidence", "answer"})
PLACEHOLDER_TEXT = "PLACEHOLDER"
_SOURCE_TYPES = {
    "table", "figure", "text_span", "equation_algorithm", "citation_context"
}


def build_line(query_id, paper_ids, evidence_items, answer):
    """Assemble a single submission record in the official sample's shape."""
    return {
        "query_id": query_id,
        "gold_papers": [{"paper_id": pid} for pid in paper_ids],
        "evidence": list(evidence_items),
        "answer": answer,
    }


def _table_columns(record: InputRecord) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    seen: set[str] = set()
    for column in record.table_schema or []:
        if not isinstance(column, dict):
            continue
        name = column.get("name")
        kind = column.get("type")
        if not isinstance(name, str) or not name or name in seen:
            continue
        if kind not in {"string", "number", "boolean"}:
            continue
        seen.add(name)
        columns.append((name, kind))
    return columns


def _placeholder_table_row(columns: list[tuple[str, str]]) -> dict[str, Any]:
    # A non-empty rows list is mandatory.  Use a conspicuous placeholder for
    # string cells and null for typed numeric/boolean cells; null is explicitly
    # accepted by the official validator and cannot masquerade as a real zero.
    return {
        name: PLACEHOLDER_TEXT if kind == "string" else None
        for name, kind in columns
    }


def _valid_cell(value: Any, kind: str) -> bool:
    if value is None:
        return True
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if kind == "boolean":
        return isinstance(value, bool)
    return False


def answer_fallback_types(
    record: InputRecord,
    answer: Any,
    component_confidences: dict[str, float] | None = None,
) -> list[str]:
    """Return requested answer components that require semantic repair.

    ``normalize_answer`` deliberately manufactures validator-safe placeholders
    when generation fails.  That is a useful last line of defence, but it must
    not be reported as a successful model answer.  This predicate mirrors the
    official nested answer contract so callers can surface those repairs in
    traces and failure counts before normalization hides them.
    """
    source = answer if isinstance(answer, dict) else {}
    fallback_types: list[str] = []

    for answer_type in record.answer_types:
        raw = source.get(answer_type)

        if answer_type == "freeform":
            valid = (
                isinstance(raw, dict)
                and set(raw) == {"text"}
                and isinstance(raw.get("text"), str)
                and bool(raw["text"].strip())
            )

        elif answer_type == "multiple_choice":
            labels = {
                str(label).strip().upper()
                for label in (record.mc_options or {})
                if str(label).strip()
            }
            choice = (
                str(raw.get("gold") or "").strip().upper()
                if isinstance(raw, dict)
                else ""
            )
            valid = (
                isinstance(raw, dict)
                and set(raw) == {"gold"}
                and bool(labels)
                and choice in labels
            )

        elif answer_type == "table":
            columns = _table_columns(record)
            expected_names = {name for name, _kind in columns}
            kinds = dict(columns)
            rows = raw.get("rows") if isinstance(raw, dict) else None
            valid = (
                isinstance(raw, dict)
                and set(raw) == {"rows"}
                and isinstance(rows, list)
                and bool(rows)
                and all(
                    isinstance(row, dict)
                    and set(row) == expected_names
                    and all(_valid_cell(row[name], kinds[name]) for name in expected_names)
                    for row in rows
                )
            )

        else:
            # Unknown answer types are not repairable under the official
            # contract; surfacing them is safer than silently claiming success.
            valid = False

        confidence = (component_confidences or {}).get(answer_type)
        strategy_fallback = (
            answer_type == "multiple_choice"
            and confidence is not None
            and confidence <= 0.1
        ) or (
            answer_type == "table"
            and confidence is not None
            and confidence <= 0.0
        )
        if not valid or strategy_fallback:
            fallback_types.append(answer_type)

    return fallback_types


def normalize_answer(record: InputRecord, answer: Any) -> dict[str, Any]:
    """Return an answer that satisfies the official nested contract.

    Only requested answer types are emitted.  Valid generated values survive;
    missing/malformed components are repaired with organizer-style placeholders
    so failure isolation never produces the validator-invalid ``answer: {}``.
    """
    source = answer if isinstance(answer, dict) else {}
    normalized: dict[str, Any] = {}

    for answer_type in record.answer_types:
        if answer_type == "freeform":
            raw = source.get("freeform")
            text = raw.get("text") if isinstance(raw, dict) else None
            if not isinstance(text, str) or not text.strip():
                text = PLACEHOLDER_TEXT
            normalized["freeform"] = {"text": text}

        elif answer_type == "multiple_choice":
            raw = source.get("multiple_choice")
            choice = raw.get("gold") if isinstance(raw, dict) else None
            choice = str(choice or "").strip().upper()
            labels = list(dict.fromkeys(
                str(label).strip().upper()
                for label in (record.mc_options or {})
                if str(label).strip()
            ))
            if choice not in labels:
                choice = labels[0] if labels else "A"
            normalized["multiple_choice"] = {"gold": choice}

        elif answer_type == "table":
            columns = _table_columns(record)
            raw = source.get("table")
            raw_rows = raw.get("rows") if isinstance(raw, dict) else None
            rows: list[dict[str, Any]] = []
            if isinstance(raw_rows, list):
                for raw_row in raw_rows:
                    if not isinstance(raw_row, dict):
                        continue
                    repaired = {
                        name: raw_row.get(name) if _valid_cell(raw_row.get(name), kind) else None
                        for name, kind in columns
                    }
                    rows.append(repaired)
            if not rows:
                rows = [_placeholder_table_row(columns)]
            normalized["table"] = {"rows": rows}

    return normalized


def build_record_line(
    record: InputRecord,
    paper_ids: list[str],
    evidence_items: list[dict],
    answer: Any,
) -> dict:
    """Build a line and enforce the input-dependent nested answer contract."""
    return build_line(
        record.query_id,
        paper_ids,
        evidence_items,
        normalize_answer(record, answer),
    )


def build_fallback_line(record: InputRecord) -> dict:
    """Validator-safe zero-information line for an isolated query failure."""
    return build_record_line(record, [], [], {})


def repair_line(record: InputRecord, line: Any) -> dict:
    """Repair an arbitrary runner result at the production write boundary.

    The repair is deliberately conservative: recover valid paper IDs and
    evidence items, strip unexpected fields, normalize the answer, and force
    the original input query ID.  Anything malformed degrades locally instead
    of invalidating or dropping the whole submission.
    """
    if not isinstance(line, dict):
        return build_fallback_line(record)

    paper_ids: list[str] = []
    seen_papers: set[str] = set()
    papers = line.get("gold_papers")
    if isinstance(papers, list):
        for item in papers:
            raw_paper_id = item.get("paper_id") if isinstance(item, dict) else None
            paper_id = raw_paper_id.strip() if isinstance(raw_paper_id, str) else ""
            if paper_id and paper_id not in seen_papers:
                seen_papers.add(paper_id)
                paper_ids.append(paper_id)

    evidence_items: list[dict] = []
    evidence = line.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            raw_paper_id = item.get("paper_id")
            paper_id = raw_paper_id.strip() if isinstance(raw_paper_id, str) else ""
            source_type = item.get("source_type")
            locator = item.get("locator")
            if (
                paper_id
                and source_type in _SOURCE_TYPES
                and isinstance(locator, dict) and locator
            ):
                evidence_items.append({
                    "paper_id": paper_id,
                    "source_type": source_type,
                    "locator": dict(locator),
                })

    return build_record_line(
        record, paper_ids, evidence_items, line.get("answer")
    )


def validate_line(line, allowed_keys=SAMPLE_KEYS):
    """Raise ValueError unless `line`'s key set is EXACTLY `allowed_keys`.

    The check is symmetric on purpose: an EXTRA key (e.g. a stray `role`)
    trips the evaluator's `additionalProperties`, and a MISSING required key
    (e.g. no `answer`/`evidence`) makes the record malformed. Either way the
    line must not be submitted, so we assert exact key-set equality rather
    than only screening for extras.
    """
    if set(line) != set(allowed_keys):
        raise ValueError(
            f"line {line.get('query_id')} key mismatch: {set(line) ^ set(allowed_keys)}"
        )
    if not line.get("query_id"):
        raise ValueError("line missing query_id")


def write_submission(lines, path, input_query_ids, allowed_keys=SAMPLE_KEYS):
    """Write `lines` as JSONL to `path`.

    Raises (before writing anything) if any line's key set doesn't match
    `allowed_keys` (the additionalProperties/missing-key landmine, enforced
    here so it's never opt-in), if any `input_query_ids` has no corresponding
    output line — a dropped query_id scores 0 precision in the official
    scorer — or if `lines` contains a duplicate `query_id`. The output uses
    `ensure_ascii=False`, matching how the organizer's `evaluate.py` prints
    its own JSON, so unicode text is byte-compatible with
    `data/sample_submission.jsonl` rather than `\\uXXXX`-escaped: one JSON
    object per line, terminated by a trailing newline.
    """
    for line in lines:
        validate_line(line, allowed_keys)
    produced = {line["query_id"] for line in lines}
    if len(lines) != len(produced):
        raise ValueError("duplicate query_id in submission")
    missing = set(input_query_ids) - produced
    if missing:
        raise ValueError(f"submission is missing query_ids (0-precision risk): {missing}")
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
            for line in lines:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
