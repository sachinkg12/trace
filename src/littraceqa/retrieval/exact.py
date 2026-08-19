"""`ExactAcronymIndex`: high-precision, no-LLM EXACT / ACRONYM lookup.

This is a distinct lookup index, not a `Retriever` (query -> ranked list) --
it recognizes a paper by an EXACT title match or a method acronym, and is
consumed by seed-finding (Tier 2) as a fast, deterministic first pass before
falling back to lexical/dense retrieval. Deliberately lean (YAGNI): no fuzzy
matching, no stemming, no LLM -- exact string equality on a normalized title,
and a regex-based acronym extraction over title+abstract.

Acronym extraction heuristic: parenthesized tokens like "(COCA)", "(BERT)",
"(GPT-4)", "(mAP)" immediately following a method name, e.g. "Foo Bar (FOO)".
The token may start with any letter (so lowercase-prefixed ML acronyms like
"(mAP)", "(sRGB)", "(cVAE)", "(vLLMs)" are captured) but must contain at
least 2 uppercase letters, so ordinary parenthesized words ("(see below)",
"(w/o)", "(Figure 3)") don't false-positive as acronyms.

Collisions are handled honestly, not silently: an acronym can legitimately
name multiple papers (e.g. "LLMs" appears across thousands of abstracts) --
`lookup_acronym` returns ALL matching paper_ids, sorted by paper_id for a
deterministic order. Two papers sharing a normalized title are both kept and
both returned by `lookup_title`, rather than one silently overwriting the
other.
"""
from __future__ import annotations

import re
import string

from littraceqa.retrieval.bm25 import STOPWORDS
from littraceqa.retrieval.pool import PoolIndex

_WHITESPACE_RE = re.compile(r"\s+")
_ACRONYM_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9-]{1,15})\)")
# Whitespace and any hyphen/dash character -- stripped by `squash_title` so an
# OCR-injected mid-word space ("C ompressor") or a spaced hyphen ("S - RAG")
# doesn't defeat a title/name lookup.
_SQUASH_STRIP_RE = re.compile(r"[\s\-‐-―]+")

# A title-initialism must look acronym-like to be worth indexing: at least 2
# letters (a single letter is not a distinctive acronym) and not absurdly long
# (a 12-word title's initialism is never queried and would only add noise).
_INITIALISM_MIN = 2
_INITIALISM_MAX = 8


def normalize_title(title: str) -> str:
    """Lowercase, collapse internal whitespace, strip surrounding punctuation.

    Used as the key for exact-title lookups, so a query that differs from
    the pool's title only in case, run-of whitespace, or a leading/trailing
    quote/period still hits.
    """
    collapsed = _WHITESPACE_RE.sub(" ", title).strip()
    return collapsed.lower().strip(string.punctuation + " ")


def squash_title(text: str) -> str:
    """Lowercase and remove ALL whitespace and hyphens/dashes.

    An ADDITIONAL, looser match key than `normalize_title`: it collapses OCR
    artifacts that inject spaces mid-word ("500x C ompressor" -> "500xcompressor")
    or space out a hyphen ("S - RAG" -> "srag"), so a clean name in the question
    ("500xCompressor", "S-RAG") still resolves the garbled pool title.
    """
    return _SQUASH_STRIP_RE.sub("", text.lower())


def _squash_keys(title: str) -> list[str]:
    """Squash keys to index/look-up a title under.

    Always the whole title, PLUS -- for a "Method: subtitle" title -- the
    pre-colon method-name segment on its own, so a question that names just the
    method ("500xCompressor") resolves a paper whose full title carries a
    subtitle ("500x C ompressor: Generalized Prompt Compression ...").
    """
    keys = [squash_title(title)]
    if ":" in title:
        keys.append(squash_title(title.split(":", 1)[0]))
    return [k for k in keys if k]


def _title_initialism(title: str) -> str:
    """Acronym built from the first letter of each significant title word.

    Stopwords (the shared BM25 list) are skipped, so "Truncated Consistency
    Models" -> "TCM" and "Inductive Moment Matching" -> "IMM". The first
    ALPHABETIC character of each kept word is taken (so a leading digit/paren
    doesn't emit a non-letter), and the result is upper-cased to match the
    acronym index's key convention.
    """
    letters: list[str] = []
    for word in title.split():
        bare = word.lower().strip(string.punctuation)
        if not bare or bare in STOPWORDS:
            continue
        for ch in word:
            if ch.isalpha():
                letters.append(ch.upper())
                break
    return "".join(letters)


def _is_acronym_token(token: str) -> bool:
    """At least 2 uppercase letters -- filters out normal parenthesized
    words (e.g. "(see)") while still accepting mixed-case method names like
    "LayTextLLM" or hyphenated ones like "GPT-4"."""
    return sum(1 for c in token if c.isupper()) >= 2


def _extract_acronyms(text: str) -> set[str]:
    return {m.upper() for m in _ACRONYM_RE.findall(text) if _is_acronym_token(m)}


class ExactAcronymIndex:
    """Built once from a `PoolIndex`; O(1) title/acronym lookups after that."""

    def __init__(self, pool: PoolIndex):
        title_index: dict[str, list[str]] = {}
        squash_index: dict[str, set[str]] = {}
        acronym_index: dict[str, set[str]] = {}
        for paper_id in pool.ids:
            paper = pool.by_id(paper_id)
            title = paper.title or ""
            abstract = paper.abstract or ""

            norm_title = normalize_title(title)
            if norm_title:
                title_index.setdefault(norm_title, []).append(paper_id)

            # ADDITIONAL squash keys (whole title + pre-colon method segment)
            # so an OCR-garbled title still resolves from a clean method name.
            for key in _squash_keys(title):
                squash_index.setdefault(key, set()).add(paper_id)

            for acronym in _extract_acronyms(f"{title} {abstract}"):
                acronym_index.setdefault(acronym, set()).add(paper_id)

            # ADDITIONAL title-initialism acronym (e.g. "Truncated Consistency
            # Models" -> "TCM"), merged into the acronym index as an extra key.
            initialism = _title_initialism(title)
            if _INITIALISM_MIN <= len(initialism) <= _INITIALISM_MAX:
                acronym_index.setdefault(initialism, set()).add(paper_id)

        self._title_index = title_index
        self._squash_index = squash_index
        self._acronym_index = acronym_index

    def lookup_title(self, query: str) -> list[str]:
        """paper_ids whose title matches `query` by normalized OR squash key.

        Matches on `normalize_title(query)` (exact, case/whitespace-insensitive)
        AND on `squash_title(query)` (whitespace/hyphen-stripped, incl. the
        pre-colon method segment) -- so both a verbatim title and a clean
        method name for an OCR-garbled title resolve. Sorted by paper_id for
        determinism; empty list if nothing matches.
        """
        ids: set[str] = set(self._title_index.get(normalize_title(query), []))
        for key in _squash_keys(query):
            ids |= self._squash_index.get(key, set())
        return sorted(ids)

    def lookup_acronym(self, acronym: str) -> list[str]:
        """paper_ids extracted for this acronym (case-insensitive).

        Sorted by paper_id for determinism; empty list if unknown.
        """
        return sorted(self._acronym_index.get(acronym.upper(), set()))
