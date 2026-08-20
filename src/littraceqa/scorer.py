"""Local scoring gate wrapping the packaged LitTraceQA evaluator primitives.

`littraceqa._vendor.evaluate` exposes a top-level
`evaluate(gold_records, pred_records)`
-> dict` function (as well as a CLI with `--gold`/`--pred` flags that prints
the same structure as JSON). We call the function directly rather than
shelling out, since it is available and avoids subprocess/tempfile overhead.

Consumers should depend on the `Scorer` Protocol, not on
`VendoredEvaluateScorer` directly (Open/Closed + DIP): the hidden
organizer-side evaluator, or a fake for tests, can be substituted by
constructing a different `Scorer` without modifying any caller.
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
from types import ModuleType
from typing import Protocol, runtime_checkable

# Package the organizer-compatible evaluator beside the validator so both a
# source checkout and an installed wheel expose the exact normalization seam.
VENDOR = pathlib.Path(__file__).resolve().parent / "_vendor" / "evaluate.py"

# The exact metric key set emitted by vendor/evaluate.py's evaluate()["metrics"].
# Single source of truth for the metric contract — tests import this rather than
# re-declaring the key set, so the contract lives in one place (the module that
# owns the scorer).
METRIC_NAMES = frozenset({
    "paper_precision_macro", "paper_recall_macro", "paper_f1_macro",
    "evidence_precision_macro", "evidence_recall_macro", "evidence_f1_macro",
    "multiple_choice_accuracy", "freeform_exact_match",
    "table_row_f1_macro", "table_cell_accuracy_macro", "table_cell_accuracy_micro",
})


@runtime_checkable
class Scorer(Protocol):
    def score(self, gold: list[dict], pred: list[dict]) -> dict: ...


@functools.lru_cache(maxsize=None)
def _load_vendor_module(path: pathlib.Path) -> ModuleType:
    """Load vendor/evaluate.py as a module without executing its CLI (`vendor/`
    has no `__init__.py`, so it isn't importable as a package). Cached by
    resolved path so repeated VendoredEvaluateScorer() construction is cheap."""
    spec = importlib.util.spec_from_file_location("littraceqa._vendored_evaluate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VendoredEvaluateScorer:
    """Adapter over the official `vendor/evaluate.py` `evaluate()` function.

    Consumers depend on the `Scorer` Protocol, so the hidden evaluator or a
    fake can be substituted without modifying any caller.
    """

    def __init__(self, evaluator_path: pathlib.Path = VENDOR):
        self._module = _load_vendor_module(evaluator_path)

    def score(self, gold: list[dict], pred: list[dict]) -> dict:
        result = self._module.evaluate(gold, pred)
        return result["metrics"]


_default: Scorer | None = None


def _get_default() -> Scorer:
    """Lazily construct and cache the default Scorer on first use.

    Constructing `VendoredEvaluateScorer()` touches the packaged evaluator on
    disk, so it must not happen at import time: other subsystems need to
    `import littraceqa.scorer` for the `Scorer` Protocol / `METRIC_NAMES` / a
    fake scorer without requiring `vendor/evaluate.py` to be present.
    """
    global _default
    if _default is None:
        _default = VendoredEvaluateScorer()
    return _default


def score(gold: list[dict], pred: list[dict]) -> dict:
    """Convenience delegate. Subsystems that need injection accept a Scorer."""
    return _get_default().score(gold, pred)


def evaluate_full(gold: list[dict], pred: list[dict]) -> dict:
    """FULL vendor evaluator output -- BOTH the `metrics` and `details`
    sub-dicts (missing/extra prediction ids, table cell tallies).

    `score()` above returns only the 11 metrics; the experiment harness
    persists the complete evaluator output to `scores.json`, so it needs the
    details too. Reuses the single `_load_vendor_module` seam so the
    vendor-loading path stays defined in exactly one place.
    """
    return _load_vendor_module(VENDOR).evaluate(gold, pred)
