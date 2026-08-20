from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from littraceqa.experiments.config import load_config
from littraceqa.experiments.submit import enforce_strict_release_safety
from littraceqa.pipeline.input import parse_input_record
from littraceqa.submission import build_fallback_line
from littraceqa.validator import assert_valid_submission


ROOT = Path(__file__).resolve().parents[1]


def test_public_config_is_full_generation_only():
    config = load_config(ROOT / "configs" / "trace-littraceqa.yaml")
    assert config.path == "new"
    assert config.params["selection"] == "answer_bearing"
    assert config.params["table_strategy"] == "planned"
    assert config.params["corpus_pdf_dir"] == "data/pdfs"
    assert not any("replay" in key for key in config.params)
    assert not any("override" in key for key in config.params)


def test_strict_release_refuses_an_external_parent():
    try:
        enforce_strict_release_safety(
            strict_release_profile=True,
            config_params={},
            frozen_source="external-parent.jsonl",
        )
    except ValueError as exc:
        assert "full-generation" in str(exc)
    else:
        raise AssertionError("strict release accepted an external parent")


def test_fallback_satisfies_the_official_nested_contract():
    raw = {
        "query_id": "synthetic_q1",
        "question": "Choose an option and report a score.",
        "answer_types": ["multiple_choice", "table"],
        "multiple_choice_options": [
            {"label": "A", "text": "first"},
            {"label": "B", "text": "second"},
        ],
        "table_schema": [
            {"name": "method", "type": "string", "is_row_key": True},
            {"name": "score", "type": "number", "is_row_key": False},
        ],
    }
    line = build_fallback_line(parse_input_record(raw))
    assert_valid_submission(
        [raw], [line], require_evidence=True, paper_ids={"synthetic_paper"}
    )


def test_reported_result_is_selected_clean_track():
    result = json.loads(
        (ROOT / "results" / "official-test-0.757968.json").read_text()
    )
    assert result["composite_score"] == 0.757968
    assert result["status"] == "official_selected_clean_track"
    assert "without_prediction_replay" in result["reproducibility_scope"]


def test_public_tree_passes_release_audit():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_release.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
