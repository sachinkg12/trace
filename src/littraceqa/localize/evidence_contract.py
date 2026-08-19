"""Gold-blind repairs for scorer-visible citation evidence contracts.

The localizer often finds the right quote and page but emits a locator that the
official scorer cannot identify.  Two common cases are bibliography entries
without their ordinal and questions that ask for a count of citations in a
section (whose answer-bearing unit is the section text, not five anonymous
``citation_context`` rows).  This wrapper repairs only those narrow cases from
the parsed PDF itself; it never consults answers or gold annotations.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from littraceqa.localize.interfaces import LocatedEvidence, ParsedPdf


_ORDINAL_REFERENCE_RE = re.compile(
    r"\b(\d+)(?:st|nd|rd|th)\s+(?:reference|citation)\b", re.IGNORECASE
)
_COUNT_CITATIONS_IN_SECTION_RE = re.compile(
    r"\bhow\s+many\b.*\b(?:cited|citations?)\b.*\bsection\b", re.IGNORECASE
)
_REFERENCES_HEADING_RE = re.compile(r"(?im)^\s*references\s*$")
_NUMBERED_REFERENCE_RE = re.compile(r"(?m)^\s*\[(\d+)\]\s+")
_REFERENCE_END_RE = re.compile(
    r"(?:19|20)\d{2}[a-z]?\.\s*\n(?=[A-Z])"
)


@dataclass(frozen=True)
class _ReferenceEntry:
    ordinal: int
    page: int
    text: str


def _normalize_tokens(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text or "")
    return re.findall(r"[a-z0-9]+", decomposed.casefold())


def _reference_pages(parsed: ParsedPdf) -> list[tuple[int, str]]:
    """Return page text from the first References heading onward."""
    out: list[tuple[int, str]] = []
    started = False
    for page in parsed.pages:
        text = page.text or ""
        if not started:
            heading = _REFERENCES_HEADING_RE.search(text)
            if heading is None:
                continue
            text = text[heading.end():]
            started = True
        out.append((page.page, text))
    return out


def _reference_entries(parsed: ParsedPdf) -> list[_ReferenceEntry]:
    """Extract numbered or author-year bibliography entries with ordinals.

    Bracket-numbered lists retain their printed IDs.  Author-year lists are
    ordered bibliographies, so their scorer-visible citation ID is the 1-based
    entry ordinal.  The latter split uses the publication-year terminator that
    ACL/PMLR PDFs consistently preserve in extracted text.
    """
    pages = _reference_pages(parsed)
    if not pages:
        return []

    joined = ""
    page_offsets: list[tuple[int, int]] = []
    for page, text in pages:
        page_offsets.append((len(joined), page))
        joined += text + "\n"

    def page_at(offset: int) -> int:
        current = page_offsets[0][1]
        for start, page in page_offsets:
            if start > offset:
                break
            current = page
        return current

    numbered = list(_NUMBERED_REFERENCE_RE.finditer(joined))
    if numbered:
        return [
            _ReferenceEntry(
                int(match.group(1)),
                page_at(match.start()),
                joined[
                    match.end():
                    numbered[position + 1].start()
                    if position + 1 < len(numbered)
                    else len(joined)
                ],
            )
            for position, match in enumerate(numbered)
        ]

    starts = [0, *[match.end() for match in _REFERENCE_END_RE.finditer(joined)]]
    return [
        _ReferenceEntry(
            position + 1,
            page_at(start),
            joined[start:starts[position + 1]
                   if position + 1 < len(starts) else len(joined)],
        )
        for position, start in enumerate(starts)
        if joined[start:].strip()
    ]


def _quote_entry_overlap(quote: str, entry: str) -> float:
    quote_tokens = set(_normalize_tokens(quote))
    # One- or two-token snippets ("paper", "et al.") occur in nearly every
    # bibliography row and cannot identify an ordinal safely.
    if len(quote_tokens) < 3:
        return 0.0
    return len(quote_tokens & set(_normalize_tokens(entry))) / len(quote_tokens)


def _entry_ordinal_for_quote(
    quote: str, entries: list[_ReferenceEntry]
) -> int | None:
    ranked = sorted(
        ((_quote_entry_overlap(quote, entry.text), entry.ordinal) for entry in entries),
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.65:
        return None
    # A tie means the quote is too generic to identify one bibliography row.
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def _printed_bracket_reference(page_text: str, ordinal: str) -> bool:
    return re.search(
        rf"(?<!\d)\[{re.escape(ordinal)}\](?!\d)", page_text or ""
    ) is not None


def _dedup(items: list[LocatedEvidence]) -> list[LocatedEvidence]:
    seen: set[tuple[str, str, int, str | None]] = set()
    out: list[LocatedEvidence] = []
    for item in items:
        key = (item.paper_id, item.source_type, item.page, item.object_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def normalize_evidence_contract(
    question: str,
    parsed: ParsedPdf,
    evidence: list[LocatedEvidence],
) -> list[LocatedEvidence]:
    """Repair narrow, parser-attested citation/source serialization gaps."""
    ordinal_match = _ORDINAL_REFERENCE_RE.search(question or "")
    count_in_section = bool(
        _COUNT_CITATIONS_IN_SECTION_RE.search(question or "")
    )
    needs_entries = any(
        item.source_type == "citation_context" and not item.object_id
        for item in evidence
    )
    entries = _reference_entries(parsed) if needs_entries else []

    out: list[LocatedEvidence] = []
    for item in evidence:
        replacement = item
        if ordinal_match and item.source_type == "text_span":
            ordinal = ordinal_match.group(1)
            page = parsed.page(item.page)
            if page is not None and _printed_bracket_reference(page.text, ordinal):
                replacement = LocatedEvidence(
                    item.paper_id,
                    "citation_context",
                    item.page,
                    ordinal,
                    item.quote,
                    item.confidence,
                )
        elif count_in_section and item.source_type == "citation_context":
            replacement = LocatedEvidence(
                item.paper_id,
                "text_span",
                item.page,
                None,
                item.quote,
                item.confidence,
            )
        elif (
            item.source_type == "citation_context"
            and not item.object_id
            and entries
        ):
            ordinal = _entry_ordinal_for_quote(item.quote, entries)
            if ordinal is not None:
                replacement = LocatedEvidence(
                    item.paper_id,
                    item.source_type,
                    item.page,
                    str(ordinal),
                    item.quote,
                    item.confidence,
                )
        out.append(replacement)
    return _dedup(out)


class EvidenceContractLocalizer:
    """Opt-in decorator over any configured evidence localizer."""

    def __init__(self, base):
        self._base = base

    def locate(self, question, paper, parsed):
        located = self._base.locate(question, paper, parsed)
        return normalize_evidence_contract(question, parsed, located)
