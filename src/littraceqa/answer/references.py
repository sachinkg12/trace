"""Deterministic reference-list parsing and reference-question answering.

An LLM cannot reliably count or index a long numbered bibliography (dozens
of entries spanning several pages of dense text). Instead of asking the LLM
"which reference is #24?", we PARSE the paper's own reference list out of
the already-parsed pages and answer ordinal/count questions in code. Pure,
defensive functions throughout: bad/missing input degrades to an empty
result, never an exception -- callers (the freeform strategy) fall back to
the LLM path when parsing yields nothing.
"""
from __future__ import annotations

import re

from littraceqa.localize.interfaces import ParsedPdf

# A real bibliography has on the order of tens-to-low-hundreds of entries. A
# parse that yields far more has over-collected past the reference list into
# appendix/body text (the extraction swept in in-text citations, numbered
# equations, list items, etc.). Such a parse is untrustworthy -- we return []
# so the caller falls back to the LLM rather than emitting a wrong count/index
# from a bogus list. (Verified: SecEmb parses to ~1854 bogus "entries" and its
# author-count question regressed vs the LLM; this guard restores the fallback.)
_MAX_PLAUSIBLE_REFERENCES = 300

# A "References" / "Bibliography" heading on its own line (optionally
# numbered, e.g. "7. References"), used to find where the bibliography
# starts. `finditer` + taking the LAST match biases toward the real
# end-of-paper bibliography rather than an earlier incidental mention (e.g.
# a table of contents or a sentence referencing "the references section").
_HEADING_RE = re.compile(r"(?m)^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$", re.IGNORECASE)

# Headings that mark the end of the bibliography block (appendix material
# that follows the reference list and must not be swept into it).
_STOP_HEADING_RE = re.compile(
    r"(?m)^\s*(?:\d+\.?\s*)?(appendix|supplementary material|acknowledg(?:e)?ments)\b",
    re.IGNORECASE,
)

# A numbered entry marker at the start of a line: "[24] ", "24. ", "24) ".
_MARKER_RE = re.compile(r"(?:(?<=\n)|^)\s*(?:\[(\d+)\]|(\d+)[.)])\s+", re.MULTILINE)

# Splits a reference entry's leading author-list text at the first comma,
# " and ", or "et al." -- the text before that split is the first author.
_FIRST_AUTHOR_SPLIT_RE = re.compile(r",|\band\b|\bet al\.?\b", re.IGNORECASE)

_REF_NUM_PREFIX_RE = re.compile(r"^\s*(?:\[\d+\]|\d+[.)])\s*")


def parse_reference_list(parsed: ParsedPdf | None) -> list[str]:
    """Find the References/Bibliography section in `parsed`'s pages and
    split it into individual entry strings, index 0 == reference 1. Never
    raises; returns [] if there is no parsed doc, no heading, or nothing
    splits out."""
    if parsed is None:
        return []
    pages = getattr(parsed, "pages", None)
    if not pages:
        return []
    try:
        pages_sorted = sorted(pages, key=lambda p: p.page)
        full_text = "\n".join(p.text or "" for p in pages_sorted)
        matches = list(_HEADING_RE.finditer(full_text))
        if not matches:
            return []
        heading = matches[-1]
        remainder = full_text[heading.end():]
        stop = _STOP_HEADING_RE.search(remainder)
        block = remainder[:stop.start()] if stop else remainder
        entries = _split_entries(block)
        # Sanity guard: an implausibly long list means we over-collected past
        # the bibliography -> distrust it, let the caller fall back to the LLM.
        if len(entries) > _MAX_PLAUSIBLE_REFERENCES:
            return []
        return entries
    except Exception:  # noqa: BLE001 -- parsing never raises, degrades to []
        return []


def _split_entries(block: str) -> list[str]:
    markers: list[tuple[int, int, int]] = []
    for m in _MARKER_RE.finditer(block):
        num_str = m.group(1) or m.group(2)
        try:
            num = int(num_str)
        except (TypeError, ValueError):
            continue
        markers.append((num, m.start(), m.end()))

    # Only keep a strictly increasing 1, 2, 3, ... run: this rejects
    # incidental digit-dot/bracket noise (stray page numbers, in-text
    # citation markers) that isn't actually the numbered bibliography.
    filtered: list[tuple[int, int, int]] = []
    expected = 1
    for num, start, end in markers:
        if num == expected:
            filtered.append((num, start, end))
            expected += 1

    if len(filtered) < 2:
        return _split_entries_fallback(block)

    entries: list[str] = []
    for i, (_num, _start, end) in enumerate(filtered):
        content_end = filtered[i + 1][1] if i + 1 < len(filtered) else len(block)
        entry = re.sub(r"\s+", " ", block[end:content_end]).strip()
        if entry:
            entries.append(entry)
    return entries


def _split_entries_fallback(block: str) -> list[str]:
    return [ln.strip() for ln in block.splitlines() if ln.strip()]


def nth_reference_first_author(refs: list[str], n: int) -> str | None:
    """Reference `n` (1-indexed) -> its first author's name, or None if out
    of range / unparseable. Never raises."""
    try:
        if not refs or n is None or n < 1 or n > len(refs):
            return None
        entry = _REF_NUM_PREFIX_RE.sub("", refs[n - 1] or "")
        if not entry.strip():
            return None
        m = _FIRST_AUTHOR_SPLIT_RE.search(entry)
        author = entry[:m.start()] if m else entry
        author = author.strip().strip(".").strip()
        return author or None
    except Exception:  # noqa: BLE001
        return None


def reference_count(refs: list[str]) -> int:
    return len(refs) if refs else 0


def references_with_author(refs: list[str], name: str) -> int:
    """Count entries whose raw text contains `name`, case-insensitive."""
    try:
        if not refs or not name or not name.strip():
            return 0
        needle = name.strip().lower()
        return sum(1 for r in refs if needle in (r or "").lower())
    except Exception:  # noqa: BLE001
        return 0


# --- question detection + deterministic routing -----------------------

# A reference-LIST question names "reference(s)"/"bibliography" directly.
# Deliberately does NOT match "how many papers were cited in the
# Introduction Section" style questions: those ask about in-text citations
# scoped to a named section, which is a different (unsolved-here) problem,
# not an ordinal/count over the numbered bibliography -- routing them here
# would silently produce a wrong number instead of falling back to the LLM.
_REF_KEYWORD_RE = re.compile(r"\b(references?|bibliography)\b", re.IGNORECASE)
_SECTION_SCOPE_RE = re.compile(r"\bsection\b", re.IGNORECASE)

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}
_ORDINAL_DIGIT_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)?$", re.IGNORECASE)

_LAST_REFERENCE_RE = re.compile(r"\blast\s+reference\b", re.IGNORECASE)
_HOW_MANY_RE = re.compile(r"\bhow many\b", re.IGNORECASE)
_HOW_MANY_REFERENCES_RE = re.compile(r"\bhow many\s+references?\b", re.IGNORECASE)
_ORDINAL_BEFORE_REFERENCE_RE = re.compile(r"(\S+)\s+reference\b", re.IGNORECASE)
_AUTHOR_NAME_RE = re.compile(
    r"\b(?:include|includes|including|have|has|with|by)\s+"
    r"([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)?)\s+as\s+an?\s+author",
    re.IGNORECASE,
)


def is_reference_list_question(question: str) -> bool:
    """True iff `question` asks about the numbered reference/bibliography
    list itself (ordinal, last-index, or author-count over it) -- as
    opposed to an in-text, section-scoped citation count."""
    if not question:
        return False
    if _SECTION_SCOPE_RE.search(question):
        return False
    return bool(_REF_KEYWORD_RE.search(question))


def _parse_ordinal(word: str) -> int | None:
    w = (word or "").strip().lower()
    if w in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[w]
    m = _ORDINAL_DIGIT_RE.match(w)
    if m:
        return int(m.group(1))
    return None


def _extract_author_name(question: str) -> str | None:
    m = _AUTHOR_NAME_RE.search(question or "")
    if not m:
        return None
    name = m.group(1).strip()
    return name or None


def answer_reference_question(question: str, refs: list[str]) -> str | None:
    """Deterministically answer a reference-list question from the parsed
    `refs`. Returns None (never raises) when the question doesn't match a
    known reference-list pattern or `refs` can't answer it -- callers fall
    back to the LLM path in that case."""
    try:
        if not refs:
            return None
        q = question or ""

        if _LAST_REFERENCE_RE.search(q):
            return str(reference_count(refs))

        name = _extract_author_name(q)
        if name and _HOW_MANY_RE.search(q):
            return str(references_with_author(refs, name))

        m = _ORDINAL_BEFORE_REFERENCE_RE.search(q)
        if m:
            n = _parse_ordinal(m.group(1))
            if n:
                author = nth_reference_first_author(refs, n)
                if author:
                    return author

        if _HOW_MANY_REFERENCES_RE.search(q):
            return str(reference_count(refs))

        return None
    except Exception:  # noqa: BLE001
        return None
