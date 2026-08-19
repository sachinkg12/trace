"""Evidence locator emitter.

Builds the nested-`locator` evidence dicts the LitTraceQA scorer expects.
Dispatch is OCP: each `source_type` contributes its locator shape via a
builder registered with `register_source_type`; `make_evidence` looks the
builder up and never needs to change when a new source type is added.

Scorer-contract landmines encoded here (verified against `vendor/evaluate.py`):
- Evidence nests its page/id fields under a `locator` object.
- `page` must be present and truthy (an int). A falsy page does NOT get
  dropped by the scorer — it stringifies to "0"/"" and mis-matches the coarse
  evidence key, silently zeroing that item's score. `make_evidence` raises
  instead of emitting a mis-scoring tuple.
- PyMuPDF pages are 0-indexed; `pdf_page_to_gold` converts to the gold
  evaluator's 1-indexed convention.
- `object_id` normalizes EQUIVALENTLY to the scorer's `normalize_visible_id`
  (vendor/evaluate.py): label variants (`Tab.`, `Fig.`) and trailing captions
  collapse to canonical `Table N` / `Figure N`, but forms the scorer would not
  collapse (`Table A1`, `Table 4.1`) pass through unchanged so both sides
  normalize identically.
"""

import re
from typing import Callable

LocatorBuilder = Callable[[int, "str | None"], dict]
_BUILDERS: dict[str, LocatorBuilder] = {}


def register_source_type(name: str) -> Callable[[LocatorBuilder], LocatorBuilder]:
    """Decorator: register `fn` as the locator builder for `source_type=name`.

    This is the sole extension point — adding a new source type never
    requires touching `make_evidence`.
    """
    def deco(fn: LocatorBuilder) -> LocatorBuilder:
        _BUILDERS[name] = fn
        return fn
    return deco


def pdf_page_to_gold(pymupdf_page0: int) -> int:
    """PyMuPDF pages are 0-indexed; gold evidence pages are 1-indexed."""
    return pymupdf_page0 + 1


_LABEL = {
    "table": "Table",
    "figure": "Figure",
    "equation": "Equation",
    "algorithm": "Algorithm",
    "citation": "Citation",
}
# Label-word variants the scorer's `normalize_visible_id` collapses to `prefix N`.
_LABEL_PATTERN = {
    "table": r"tab(?:le|\.)?",
    "figure": r"fig(?:ure|\.)?",
    "equation": r"eq(?:uation|\.)?",
    "algorithm": r"alg(?:orithm|\.)?",
    "citation": r"(?:cit(?:ation|\.)?|ref(?:erence|\.)?)",
}


def normalize_object_id(raw: str, source_type: str) -> str:
    """Normalize a raw caption/label to the canonical id the scorer's
    `normalize_visible_id` (vendor/evaluate.py) reduces a clean gold id to.

    Equivalent to the scorer, not byte-identical to its implementation: a
    `<label> <n>` prefix (with `Tab.`/`Fig.` variants and any trailing
    caption) collapses to `Table N` / `Figure N`; a bare number gains the
    label; anything the scorer would NOT collapse (`Table A1`, `Table 4.1`)
    passes through cleaned so both sides normalize identically.
    The `(?![.\\w])` guard stops an interior digit run (the `.1` of `4.1`, or a
    letter-led `A1`) from being grabbed as if it were a clean `Table 4`.
    """
    label = _LABEL[source_type]
    text = raw.strip()
    bracketed = re.fullmatch(
        r"(?:\(|\[)?\s*(\d+[a-z]?)\s*(?:\)|\])?", text, re.IGNORECASE
    )
    if bracketed:
        return f"{label} {bracketed.group(1)}"
    m = re.match(
        rf"{_LABEL_PATTERN[source_type]}\s*[\(\[]?\s*(\d+[a-z]?)\s*[\)\]]?(?![.\w])",
        text,
        re.IGNORECASE,
    )
    if m:
        return f"{label} {m.group(1)}"
    if re.fullmatch(r"\d+[a-z]?", text):
        return f"{label} {text}"
    return text


def _page_only(page: int, object_id: "str | None") -> dict:
    return {"page": int(page)}


# Plain text contributes page only.  The current public evaluator also scores
# visible equation/algorithm and citation IDs, so those sources have dedicated
# builders below rather than silently discarding ``object_id``.
register_source_type("text_span")(_page_only)


@register_source_type("equation_algorithm")
def _equation_algorithm(page, object_id):
    loc = _page_only(page, object_id)
    if isinstance(object_id, int) and not isinstance(object_id, bool):
        object_id = str(object_id)
    if isinstance(object_id, str) and object_id.strip():
        raw = object_id.strip()
        if re.match(r"alg(?:orithm|\.)?\s*", raw, re.IGNORECASE):
            loc["algorithm_id"] = normalize_object_id(raw, "algorithm")
        else:
            loc["equation_id"] = normalize_object_id(raw, "equation")
    return loc


@register_source_type("citation_context")
def _citation_context(page, object_id):
    loc = _page_only(page, object_id)
    if isinstance(object_id, int) and not isinstance(object_id, bool):
        object_id = str(object_id)
    if isinstance(object_id, str) and object_id.strip():
        loc["citation_id"] = normalize_object_id(object_id, "citation")
    return loc


@register_source_type("table")
def _table(page, object_id):
    loc = _page_only(page, object_id)
    if object_id is not None:
        loc["table_id"] = normalize_object_id(object_id, "table")
    return loc


@register_source_type("figure")
def _figure(page, object_id):
    loc = _page_only(page, object_id)
    if object_id is not None:
        loc["figure_id"] = normalize_object_id(object_id, "figure")
    return loc


def make_evidence(paper_id: str, source_type: str, page: int, object_id: "str | None" = None) -> dict:
    """Build a nested-locator evidence dict for `paper_id`/`source_type`,
    dispatched through the `register_source_type` registry. Closed for
    modification: a new source type is added by registration, not by
    editing this function."""
    if not page:
        raise ValueError("page must be present and truthy or the scorer silently mis-scores the item")
    if source_type not in _BUILDERS:
        raise KeyError(f"no locator builder registered for source_type={source_type!r}")
    return {"paper_id": paper_id, "source_type": source_type,
            "locator": _BUILDERS[source_type](page, object_id)}
