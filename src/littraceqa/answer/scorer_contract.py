"""Single coupling point to the official scorer's matching primitives.

The table assembler (row-key normalization) and the evidence-dedup step must
use the EXACT functions the packaged organizer evaluator uses to match rows
and evidence -- any divergence silently zeroes a correct prediction. Rather
than reimplement `normalize_text` / `coarse_evidence_key` (and risk drift),
we load them from the vendored evaluator through `scorer._load_vendor_module`
(the same blessed loader `scorer.VendoredEvaluateScorer` uses), so there is
one place -- here -- that depends on the scorer internals.
"""
from __future__ import annotations

from typing import Any

from littraceqa.scorer import VENDOR, _load_vendor_module


def _vendor():
    return _load_vendor_module(VENDOR)


def normalize_text(value: Any) -> str:
    """The scorer's row-key / string-cell normalizer (lower, strip quotes,
    collapse whitespace). Rows match by `normalize_text` of each row-key
    column, so the assembler MUST group by this exact function."""
    return _vendor().normalize_text(value)


def normalize_visible_id(value: Any, prefix: str) -> str:
    """The scorer's visible table/figure/equation/citation ID normalizer.

    Source-object producers use this at ingestion time so every stored object
    identity has exactly the same spelling semantics as the evaluator.  The
    human-visible caption remains available in the source block text.
    """
    return _vendor().normalize_visible_id(value, prefix)


def coarse_evidence_key(item: dict) -> tuple[str, str, str, str]:
    """The scorer's evidence-set key: `(paper_id, source_type, page-or-section,
    normalized_visible_id)`. The evaluator converts predicted evidence to a
    SET of these keys, so duplicates share a key and give no extra credit while
    consuming the emit cap. Dedup by THIS key before capping."""
    return _vendor().coarse_evidence_key(item)
