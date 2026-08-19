"""Query-focused page shortlisting for whole-paper evidence localization.

The base LLM localizer is deliberately conservative and grounded, but asking a
small model to inspect every page at once dilutes the answer-bearing appendix
or experiment page.  This decorator keeps the same parser and validation
contract while showing the LLM only a deterministic shortlist:

* the highest-BM25 pages for the question;
* pages that visibly contain an explicitly requested table/figure/equation;
* the start of the bibliography for citation/reference questions.

Page numbers and objects are never rewritten.  The wrapped localizer still
validates every returned quote and visible ID against the original PageUnit.
"""
from __future__ import annotations

import re

from littraceqa.localize.interfaces import PageUnit, ParsedPdf
from littraceqa.retrieval.bm25 import BM25Index, tokenize


DEFAULT_SHORTLIST_K = 8
DEFAULT_SHORTLIST_MAX_PAGES = 12

_OBJECT_REF_RE = re.compile(
    r"\b(table|figure|fig[.]?)\s*\(?([0-9]+[a-z]?)\)?", re.IGNORECASE
)
_EQUATION_REF_RE = re.compile(
    r"\b(?:eq(?:uation)?|algorithm)\s*[.(]?\s*([0-9]+[a-z]?)\)?",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"\b(?:cite[sd]?|citation|reference|bibliograph\w*)\b", re.IGNORECASE
)
_ORDINAL_REFERENCE_RE = re.compile(
    r"\b(\d+)(?:st|nd|rd|th)\s+(?:reference|citation)\b", re.IGNORECASE
)
_REFERENCES_HEADING_RE = re.compile(r"(?im)^\s*references\s*$")


def _validate_limits(top_k: int, max_pages: int) -> None:
    for label, value in (("top_k", top_k), ("max_pages", max_pages)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if top_k > max_pages:
        raise ValueError("top_k cannot exceed max_pages")


def _visible_equation(text: str, number: str) -> bool:
    escaped = re.escape(number)
    return re.search(
        rf"(?:\b(?:eq(?:uation)?|algorithm)[.]?\s*\(?\s*{escaped}(?![0-9a-z])"
        rf"\s*\)?|\(\s*{escaped}\s*\))",
        text or "",
        re.IGNORECASE,
    ) is not None


def _anchored_pages(question: str, pages: list[PageUnit]) -> set[int]:
    """Pages grounded by literal locator/reference constraints in a question."""
    anchored: set[int] = set()

    for match in _OBJECT_REF_RE.finditer(question or ""):
        source_type = (
            "figure" if match.group(1).casefold().startswith("fig") else "table"
        )
        number = match.group(2).casefold()
        for page in pages:
            for obj in page.objects:
                obj_number = re.search(r"([0-9]+[a-z]?)\s*$", obj.object_id, re.I)
                if (
                    obj.source_type == source_type
                    and obj_number is not None
                    and obj_number.group(1).casefold() == number
                ):
                    anchored.add(page.page)

    for match in _EQUATION_REF_RE.finditer(question or ""):
        number = match.group(1)
        anchored.update(
            page.page for page in pages if _visible_equation(page.text, number)
        )

    ordinal = _ORDINAL_REFERENCE_RE.search(question or "")
    if ordinal is not None:
        number = re.escape(ordinal.group(1))
        anchored.update(
            page.page
            for page in pages
            if re.search(rf"(?<!\d)\[\s*{number}\s*\](?!\d)", page.text or "")
        )

    if _CITATION_RE.search(question or ""):
        # Bibliographies can span several pages.  Three pages from the printed
        # heading cover the common case without restoring the full-paper prompt.
        for position, page in enumerate(pages):
            if _REFERENCES_HEADING_RE.search(page.text or ""):
                anchored.update(item.page for item in pages[position:position + 3])
                break

    return anchored


def shortlist_parsed_pdf(
    question: str,
    parsed: ParsedPdf,
    *,
    top_k: int = DEFAULT_SHORTLIST_K,
    max_pages: int = DEFAULT_SHORTLIST_MAX_PAGES,
) -> ParsedPdf:
    """Return a page-number-preserving, query-focused view of ``parsed``."""
    _validate_limits(top_k, max_pages)
    if len(parsed.pages) <= top_k:
        return parsed

    page_ids = [str(page.page) for page in parsed.pages]
    index = BM25Index(page_ids, [tokenize(page.text) for page in parsed.pages])
    scores = index.score_all(tokenize(question))
    ranked = sorted(
        range(len(parsed.pages)),
        key=lambda position: (
            -scores.get(position, 0.0),
            parsed.pages[position].page,
        ),
    )

    by_page = {page.page: position for position, page in enumerate(parsed.pages)}
    anchors = _anchored_pages(question, parsed.pages)
    anchor_positions = sorted(
        (by_page[page] for page in anchors if page in by_page),
        key=lambda position: (
            -scores.get(position, 0.0),
            parsed.pages[position].page,
        ),
    )
    selected: list[int] = []
    for position in [*anchor_positions, *ranked[:top_k]]:
        if position not in selected:
            selected.append(position)
        if len(selected) >= max_pages:
            break

    # Preserve document order in the prompt while retaining original 1-based
    # page numbers.  This is a view, not a renumbering operation.
    selected.sort()
    return ParsedPdf(parsed.paper_id, [parsed.pages[position] for position in selected])


class FocusedEvidenceLocalizer:
    """Decorate an existing grounded localizer with deterministic shortlisting."""

    def __init__(
        self,
        base,
        *,
        top_k: int = DEFAULT_SHORTLIST_K,
        max_pages: int = DEFAULT_SHORTLIST_MAX_PAGES,
    ):
        _validate_limits(top_k, max_pages)
        self._base = base
        self._top_k = top_k
        self._max_pages = max_pages

    def locate(self, question, paper, parsed):
        focused = shortlist_parsed_pdf(
            question,
            parsed,
            top_k=self._top_k,
            max_pages=self._max_pages,
        )
        return self._base.locate(question, paper, focused)
