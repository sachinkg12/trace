#!/usr/bin/env python3
"""Validate a LitTraceQA prediction JSONL against a released input split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ANSWER_TYPES = {"freeform", "multiple_choice", "table"}
SOURCE_TYPES = {"table", "figure", "text_span", "equation_algorithm", "citation_context"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def index_by_query_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id") or "").strip()
        if not query_id:
            raise ValueError(f"{label}:{line_number}: missing query_id")
        if query_id in indexed:
            raise ValueError(f"{label}:{line_number}: duplicate query_id {query_id!r}")
        indexed[query_id] = row
    return indexed


def validate_value(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return False


def multiple_choice_option_labels(input_row: dict[str, Any]) -> set[str]:
    options = input_row.get("multiple_choice_options") or {}
    if isinstance(options, dict):
        return {str(label).strip().upper() for label in options}
    if isinstance(options, list):
        return {
            str(option.get("label") or "").strip().upper()
            for option in options
            if isinstance(option, dict)
        }
    return set()


def validate_submission(
    inputs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    require_evidence: bool,
    paper_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    input_by_id = index_by_query_id(inputs, "input")
    pred_by_id = index_by_query_id(predictions, "prediction")

    missing = sorted(set(input_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(input_by_id))
    if missing:
        errors.append(f"missing predictions: {len(missing)} (first: {', '.join(missing[:5])})")
    if extra:
        errors.append(f"unknown query_id values: {len(extra)} (first: {', '.join(extra[:5])})")

    for query_id, input_row in input_by_id.items():
        pred = pred_by_id.get(query_id)
        if pred is None:
            continue
        allowed_top = {"query_id", "gold_papers", "answer", "evidence"}
        unexpected = sorted(set(pred) - allowed_top)
        if unexpected:
            errors.append(f"{query_id}: unexpected fields: {', '.join(unexpected)}")

        papers = pred.get("gold_papers")
        if not isinstance(papers, list):
            errors.append(f"{query_id}: gold_papers must be a list")
        else:
            seen_papers: set[str] = set()
            for index, paper in enumerate(papers):
                if not isinstance(paper, dict) or set(paper) != {"paper_id"}:
                    errors.append(f"{query_id}: gold_papers[{index}] must contain only paper_id")
                    continue
                paper_id = str(paper.get("paper_id") or "").strip()
                if not paper_id:
                    errors.append(f"{query_id}: gold_papers[{index}].paper_id is empty")
                elif paper_id not in paper_ids:
                    errors.append(f"{query_id}: unknown paper_id {paper_id!r}")
                elif paper_id in seen_papers:
                    errors.append(f"{query_id}: duplicate paper_id {paper_id!r}")
                seen_papers.add(paper_id)

        evidence = pred.get("evidence")
        if require_evidence and not isinstance(evidence, list):
            errors.append(f"{query_id}: evidence must be a list for this split")
        if isinstance(evidence, list):
            for index, item in enumerate(evidence):
                if not isinstance(item, dict):
                    errors.append(f"{query_id}: evidence[{index}] must be an object")
                    continue
                if set(item) != {"paper_id", "source_type", "locator"}:
                    errors.append(
                        f"{query_id}: evidence[{index}] must contain paper_id, source_type, and locator"
                    )
                    continue
                paper_id = str(item.get("paper_id") or "").strip()
                source_type = str(item.get("source_type") or "").strip()
                locator = item.get("locator")
                if paper_id not in paper_ids:
                    errors.append(f"{query_id}: evidence[{index}] has unknown paper_id {paper_id!r}")
                if source_type not in SOURCE_TYPES:
                    errors.append(f"{query_id}: evidence[{index}] has invalid source_type {source_type!r}")
                if not isinstance(locator, dict) or not locator:
                    errors.append(f"{query_id}: evidence[{index}].locator must be a non-empty object")

        answer_types = set(input_row.get("answer_types") or [])
        answer = pred.get("answer")
        if not isinstance(answer, dict):
            errors.append(f"{query_id}: answer must be an object")
            continue
        if set(answer) != answer_types:
            errors.append(
                f"{query_id}: answer keys {sorted(answer)} do not match requested types {sorted(answer_types)}"
            )
            continue

        if "freeform" in answer_types:
            freeform = answer.get("freeform")
            if not isinstance(freeform, dict) or set(freeform) != {"text"}:
                errors.append(f"{query_id}: answer.freeform must contain only text")
            elif not isinstance(freeform["text"], str) or not freeform["text"].strip():
                errors.append(f"{query_id}: answer.freeform.text must be non-empty")

        if "multiple_choice" in answer_types:
            multiple_choice = answer.get("multiple_choice")
            if not isinstance(multiple_choice, dict) or set(multiple_choice) != {"gold"}:
                errors.append(f"{query_id}: answer.multiple_choice must contain only gold")
            else:
                choice = str(multiple_choice.get("gold") or "").strip().upper()
                options = multiple_choice_option_labels(input_row)
                if choice not in options:
                    errors.append(
                        f"{query_id}: answer.multiple_choice.gold must be one of {sorted(options)}"
                    )

        if "table" in answer_types:
            table = answer.get("table")
            if not isinstance(table, dict) or set(table) != {"rows"}:
                errors.append(f"{query_id}: answer.table must contain only rows")
                continue
            rows = table.get("rows")
            schema = input_row.get("table_schema") or []
            columns = {column["name"]: column["type"] for column in schema}
            if not isinstance(rows, list) or not rows:
                errors.append(f"{query_id}: answer.table.rows must be a non-empty list")
                continue
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict) or set(row) != set(columns):
                    errors.append(
                        f"{query_id}: table row {row_index} must use exactly these columns: {sorted(columns)}"
                    )
                    continue
                for column_name, column_type in columns.items():
                    if row[column_name] is not None and not validate_value(row[column_name], column_type):
                        errors.append(
                            f"{query_id}: table row {row_index} column {column_name!r} "
                            f"must be {column_type} or null"
                        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Released input JSONL.")
    parser.add_argument("--pred", required=True, type=Path, help="Prediction JSONL to validate.")
    parser.add_argument(
        "--paper-metadata",
        type=Path,
        default=Path("data/paper_metadata.jsonl"),
        help="Released paper metadata JSONL.",
    )
    parser.add_argument(
        "--require-evidence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require evidence in every prediction. Defaults to true for test and validation inputs.",
    )
    args = parser.parse_args()

    require_evidence = args.require_evidence
    if require_evidence is None:
        require_evidence = args.input.name != "test_extra.jsonl"

    inputs = read_jsonl(args.input)
    predictions = read_jsonl(args.pred)
    metadata = read_jsonl(args.paper_metadata)
    paper_ids = {str(row.get("paper_id") or "").strip() for row in metadata}
    errors = validate_submission(inputs, predictions, require_evidence, paper_ids)
    if errors:
        print(f"Submission is invalid: {len(errors)} error(s)")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} more error(s)")
        return 1
    print(f"Submission is valid: {len(predictions)} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
