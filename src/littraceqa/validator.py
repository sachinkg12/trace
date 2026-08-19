"""Adapter over the pinned official completed-submission validator."""

from __future__ import annotations

import functools
import importlib.util
import pathlib
from types import ModuleType
from typing import Any


OFFICIAL_REVISION = "bd35dc14cf0483e0ffa51fa2a54d2689c13f9845"
VENDOR = pathlib.Path(__file__).resolve().parent / "_vendor" / "validate_submission.py"


@functools.lru_cache(maxsize=None)
def _load_vendor_module(path: pathlib.Path = VENDOR) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "littraceqa._vendored_validate_submission", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored validator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_submission(
    inputs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    require_evidence: bool,
    paper_ids: set[str],
) -> list[str]:
    """Return the exact error list from the official validator."""
    return _load_vendor_module().validate_submission(
        inputs, predictions, require_evidence, paper_ids
    )


def assert_valid_submission(
    inputs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    require_evidence: bool,
    paper_ids: set[str],
) -> None:
    """Raise one actionable error before an invalid file reaches disk."""
    errors = validate_submission(inputs, predictions, require_evidence, paper_ids)
    if errors:
        rendered = "\n".join(f"- {error}" for error in errors[:100])
        if len(errors) > 100:
            rendered += f"\n- ... {len(errors) - 100} more error(s)"
        raise ValueError(
            f"official submission validation failed ({len(errors)} error(s)):\n{rendered}"
        )
