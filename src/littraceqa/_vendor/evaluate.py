#!/usr/bin/env python3
"""Lightweight local evaluator for LitTraceQA development submissions.

This script is intended for participant-side validation on the public
development split. The official hidden-test evaluator may include additional
organizer-side checks, but follows the same high-level fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.strip("\"'“”‘’`")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_id(value: Any) -> str:
    return str(value or "").strip()


def normalize_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def paper_id_set(record: dict[str, Any]) -> set[str]:
    papers = record.get("gold_papers") or record.get("papers") or []
    paper_ids: set[str] = set()
    if not isinstance(papers, list):
        return paper_ids
    for item in papers:
        if isinstance(item, str):
            paper_id = item
        elif isinstance(item, dict):
            paper_id = item.get("paper_id", "")
        else:
            paper_id = ""
        paper_id = normalize_id(paper_id)
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def normalize_visible_id(value: Any, prefix: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    match = re.fullmatch(rf"{prefix.lower()}\s*(\d+[a-z]?)", text)
    if match:
        return f"{prefix.lower()} {match.group(1)}"
    if re.fullmatch(r"\d+[a-z]?", text):
        return f"{prefix.lower()} {text}"
    return text


def coarse_evidence_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    paper_id = normalize_id(item.get("paper_id"))
    source_type = normalize_id(item.get("source_type"))
    locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
    location = str(locator.get("page") or locator.get("section") or "").strip()
    object_id = ""
    if source_type == "table":
        object_id = normalize_visible_id(locator.get("table_id"), "table")
    elif source_type == "figure":
        object_id = normalize_visible_id(locator.get("figure_id"), "figure")
    elif source_type == "equation_algorithm":
        object_id = normalize_visible_id(
            locator.get("equation_id") or locator.get("algorithm_id"),
            "equation",
        )
    elif source_type == "citation_context":
        object_id = normalize_visible_id(locator.get("citation_id"), "citation")
    return (paper_id, source_type, location, object_id)


def evidence_set(record: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    evidence = record.get("evidence") or []
    if not isinstance(evidence, list):
        return set()
    keys = set()
    for item in evidence:
        if isinstance(item, dict):
            key = coarse_evidence_key(item)
            if key[0] and key[1] and (key[2] or key[3]):
                keys.add(key)
    return keys


def prf(gold: set[Any], pred: set[Any]) -> tuple[float, float, float]:
    if not gold and not pred:
        return (1.0, 1.0, 1.0)
    if not pred:
        return (0.0, 0.0 if gold else 1.0, 0.0)
    correct = len(gold & pred)
    precision = correct / len(pred) if pred else 0.0
    recall = correct / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f1)


def prediction_answer(record: dict[str, Any]) -> dict[str, Any]:
    answer = record.get("answer")
    return answer if isinstance(answer, dict) else {}


def multiple_choice_prediction(record: dict[str, Any]) -> str:
    answer = prediction_answer(record)
    mc = answer.get("multiple_choice")
    if isinstance(mc, dict):
        return normalize_id(mc.get("gold") or mc.get("answer") or mc.get("predicted_answer_id")).upper()
    return normalize_id(record.get("predicted_answer_id") or record.get("gold") or record.get("gold_answer")).upper()


def multiple_choice_gold(record: dict[str, Any]) -> str:
    mc = record.get("answer", {}).get("multiple_choice", {})
    return normalize_id(mc.get("gold")).upper() if isinstance(mc, dict) else ""


def freeform_prediction(record: dict[str, Any]) -> str:
    answer = prediction_answer(record)
    freeform = answer.get("freeform")
    if isinstance(freeform, dict):
        return normalize_text(freeform.get("text"))
    return normalize_text(record.get("answer_text") or record.get("freeform"))


def freeform_gold(record: dict[str, Any]) -> str:
    freeform = record.get("answer", {}).get("freeform", {})
    return normalize_text(freeform.get("text")) if isinstance(freeform, dict) else ""


def row_key_value(row: dict[str, Any], row_key_columns: list[str]) -> tuple[str, ...]:
    return tuple(normalize_text(row.get(column)) for column in row_key_columns)


def cell_equal(gold_value: Any, pred_value: Any, column_type: str) -> bool:
    if gold_value is None or pred_value is None:
        return gold_value is None and pred_value is None
    if column_type == "number":
        gold_number = normalize_number(gold_value)
        pred_number = normalize_number(pred_value)
        if gold_number is None or pred_number is None:
            return False
        return math.isclose(gold_number, pred_number, rel_tol=1e-6, abs_tol=1e-6)
    if column_type == "boolean":
        return bool(gold_value) == bool(pred_value)
    return normalize_text(gold_value) == normalize_text(pred_value)


def table_metrics(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, float | int]:
    gold_table = gold.get("answer", {}).get("table", {})
    pred_table = prediction_answer(pred).get("table", {})
    if not isinstance(gold_table, dict) or not isinstance(pred_table, dict):
        return {"row_precision": 0.0, "row_recall": 0.0, "row_f1": 0.0, "cell_accuracy": 0.0, "cell_correct": 0, "cell_total": 0}

    schema = gold_table.get("schema") if isinstance(gold_table.get("schema"), list) else []
    gold_rows = gold_table.get("rows") if isinstance(gold_table.get("rows"), list) else []
    pred_rows = pred_table.get("rows") if isinstance(pred_table.get("rows"), list) else []
    row_keys = [column["name"] for column in schema if isinstance(column, dict) and column.get("is_row_key")]
    if not row_keys and schema:
        row_keys = [str(schema[0].get("name", ""))]

    gold_by_key = {
        row_key_value(row, row_keys): row
        for row in gold_rows
        if isinstance(row, dict) and row_key_value(row, row_keys)
    }
    pred_by_key = {
        row_key_value(row, row_keys): row
        for row in pred_rows
        if isinstance(row, dict) and row_key_value(row, row_keys)
    }
    row_precision, row_recall, row_f1 = prf(set(gold_by_key), set(pred_by_key))

    cell_correct = 0
    cell_total = 0
    comparable_columns = [
        (str(column.get("name", "")), str(column.get("type", "string")))
        for column in schema
        if isinstance(column, dict) and str(column.get("name", "")) and not column.get("is_row_key")
    ]
    for key, gold_row in gold_by_key.items():
        pred_row = pred_by_key.get(key)
        if not isinstance(pred_row, dict):
            cell_total += len(comparable_columns)
            continue
        for column_name, column_type in comparable_columns:
            cell_total += 1
            if cell_equal(gold_row.get(column_name), pred_row.get(column_name), column_type):
                cell_correct += 1
    cell_accuracy = cell_correct / cell_total if cell_total else row_f1
    return {
        "row_precision": row_precision,
        "row_recall": row_recall,
        "row_f1": row_f1,
        "cell_accuracy": cell_accuracy,
        "cell_correct": cell_correct,
        "cell_total": cell_total,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = {normalize_id(record.get("query_id")): record for record in gold_records}
    pred_by_id = {normalize_id(record.get("query_id")): record for record in pred_records}
    pred_by_id.pop("", None)

    missing_predictions = sorted(set(gold_by_id) - set(pred_by_id))
    extra_predictions = sorted(set(pred_by_id) - set(gold_by_id))

    paper_precision: list[float] = []
    paper_recall: list[float] = []
    paper_f1: list[float] = []
    evidence_precision: list[float] = []
    evidence_recall: list[float] = []
    evidence_f1: list[float] = []
    mc_correct = 0
    mc_total = 0
    freeform_exact_correct = 0
    freeform_total = 0
    table_row_f1: list[float] = []
    table_cell_accuracy: list[float] = []
    table_cell_correct = 0
    table_cell_total = 0

    for query_id, gold in gold_by_id.items():
        pred = pred_by_id.get(query_id, {})
        evaluation_targets = set(
            gold.get("evaluation_targets") or ["paper", "evidence", "answer"]
        )

        if "paper" in evaluation_targets:
            p, r, f = prf(paper_id_set(gold), paper_id_set(pred))
            paper_precision.append(p)
            paper_recall.append(r)
            paper_f1.append(f)

        if "evidence" in evaluation_targets:
            p, r, f = prf(evidence_set(gold), evidence_set(pred))
            evidence_precision.append(p)
            evidence_recall.append(r)
            evidence_f1.append(f)

        answer_types = set(gold.get("answer_types") or [])
        if "answer" in evaluation_targets and "multiple_choice" in answer_types:
            mc_total += 1
            mc_correct += int(multiple_choice_prediction(pred) == multiple_choice_gold(gold))

        if "answer" in evaluation_targets and "freeform" in answer_types:
            freeform_total += 1
            freeform_exact_correct += int(freeform_prediction(pred) == freeform_gold(gold))

        if "answer" in evaluation_targets and "table" in answer_types:
            metrics = table_metrics(gold, pred)
            table_row_f1.append(float(metrics["row_f1"]))
            table_cell_accuracy.append(float(metrics["cell_accuracy"]))
            table_cell_correct += int(metrics["cell_correct"])
            table_cell_total += int(metrics["cell_total"])

    return {
        "metrics": {
            "paper_precision_macro": mean(paper_precision) if paper_precision else None,
            "paper_recall_macro": mean(paper_recall) if paper_recall else None,
            "paper_f1_macro": mean(paper_f1) if paper_f1 else None,
            "evidence_precision_macro": mean(evidence_precision) if evidence_precision else None,
            "evidence_recall_macro": mean(evidence_recall) if evidence_recall else None,
            "evidence_f1_macro": mean(evidence_f1) if evidence_f1 else None,
            "multiple_choice_accuracy": mc_correct / mc_total if mc_total else None,
            "freeform_exact_match": freeform_exact_correct / freeform_total if freeform_total else None,
            "table_row_f1_macro": mean(table_row_f1),
            "table_cell_accuracy_macro": mean(table_cell_accuracy),
            "table_cell_accuracy_micro": table_cell_correct / table_cell_total if table_cell_total else None,
        },
        "details": {
            "total": len(gold_records),
            "missing_prediction_count": len(missing_predictions),
            "extra_prediction_count": len(extra_predictions),
            "table_cell_correct": table_cell_correct,
            "table_cell_total": table_cell_total,
            "missing_predictions": missing_predictions,
            "extra_predictions": extra_predictions,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LitTraceQA development predictions.")
    parser.add_argument("--gold", default="data/validation.jsonl", help="Gold validation JSONL file.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL file.")
    args = parser.parse_args()

    gold_records = read_jsonl(Path(args.gold))
    pred_records = read_jsonl(Path(args.pred))
    metrics = evaluate(gold_records, pred_records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
