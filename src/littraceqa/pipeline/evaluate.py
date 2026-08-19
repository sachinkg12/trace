"""`run_submission`: read an input JSONL, run every record through a
`Pipeline`, and write the official submission JSONL.

Lives here (rather than `build.py`) because it pairs with `evaluate_dev`
(the dev-set scoring loop) that a later task adds to this same module --
both are "drive the pipeline end-to-end over a file" entry points, distinct
from `build.py`'s job of wiring concrete backends together.
"""
from __future__ import annotations

import json
import pathlib

from littraceqa.pipeline.input import parse_input_record
from littraceqa.scorer import score as _score
from littraceqa.submission import repair_line, write_submission

# (metric key, human-readable label) in the order `format_scorecard` renders
# them. Single source of truth for scorecard layout; the key set itself is
# owned by `littraceqa.scorer.METRIC_NAMES`, not re-declared here.
_SCORECARD_ROWS = [
    ("paper_precision_macro", "Paper precision (macro)"),
    ("paper_recall_macro", "Paper recall (macro)"),
    ("paper_f1_macro", "Paper F1 (macro)"),
    ("evidence_precision_macro", "Evidence precision (macro)"),
    ("evidence_recall_macro", "Evidence recall (macro)"),
    ("evidence_f1_macro", "Evidence F1 (macro)"),
    ("multiple_choice_accuracy", "Multiple-choice accuracy"),
    ("freeform_exact_match", "Freeform exact match"),
    ("table_row_f1_macro", "Table row F1 (macro)"),
    ("table_cell_accuracy_macro", "Table cell accuracy (macro)"),
    ("table_cell_accuracy_micro", "Table cell accuracy (micro)"),
]


def _read_jsonl(path) -> list[dict]:
    records = []
    with pathlib.Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def evaluate_dev(pred_path, gold_path) -> dict:
    """Score a submission JSONL against a gold JSONL using the REAL vendored
    scorer (`vendor/evaluate.py`'s `evaluate()`, reached here via
    `littraceqa.scorer.score` so the vendor-loading logic lives in exactly
    one place).

    Returns the `metrics` sub-dict only (not the `details` sub-dict) --
    the 11 known metric names in `littraceqa.scorer.METRIC_NAMES`, each a
    float in [0, 1] or `None` when the gold subset for that metric is empty
    (e.g. `multiple_choice_accuracy` is `None` if no gold record asks for
    `multiple_choice`).
    """
    pred_records = _read_jsonl(pred_path)
    gold_records = _read_jsonl(gold_path)
    return _score(gold_records, pred_records)


def format_scorecard(metrics: dict) -> str:
    """Render `metrics` (as returned by `evaluate_dev`) as a readable table,
    one row per known metric, in `_SCORECARD_ROWS` order. A metric absent
    from `metrics` or `None` (empty gold subset for that metric) prints as
    `n/a` rather than crashing on `None`-formatting."""
    lines = ["LitTraceQA dev scorecard", "-" * 40]
    label_width = max(len(label) for _, label in _SCORECARD_ROWS)
    for key, label in _SCORECARD_ROWS:
        value = metrics.get(key)
        rendered = f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
        lines.append(f"{label:<{label_width}} : {rendered}")
    return "\n".join(lines)


def run_submission(input_path, output_path, pipeline) -> None:
    """Read `input_path` (JSONL of raw input records), run each through
    `pipeline.run`, and write the collected lines to `output_path` via
    `write_submission` -- which enforces the never-drop / no-dup /
    exact-key-set submission contract.

    `pipeline` is any object exposing `run(InputRecord) -> dict` (the
    `Pipeline` Protocol-shaped orchestrator from `orchestrator.py`, or a
    fake in tests) -- this function never names a concrete pipeline class.
    """
    records = []
    with pathlib.Path(input_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(parse_input_record(json.loads(line)))

    lines = [repair_line(record, pipeline.run(record)) for record in records]
    input_query_ids = [record.query_id for record in records]

    write_submission(lines, output_path, input_query_ids=input_query_ids)
