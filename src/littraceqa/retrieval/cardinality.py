"""Conservative, deterministic target-paper cardinality extraction.

The production selector needs an exact count only when the question states one
unambiguously.  Numbers describing answer values (trials, scenes, tasks,
matrices, and so on) must never be mistaken for a paper count.  Ambiguous or
conflicting mentions therefore return ``None`` and leave the existing
multiplicity/max-papers policy unchanged.
"""
from __future__ import annotations

import re

_COUNT_VALUES = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    **{str(value): value for value in range(2, 11)},
}
_COUNT_ALT = "|".join(_COUNT_VALUES)
_VENUE_ALT = "acl|naacl|emnlp|neurips|iclr|icml|cvpr|iccv|eccv"

# A number followed by a singular/plural paper noun is the strongest explicit
# form: "two papers", "these two ICCV 2025 diffusion papers", etc.  The
# middle is validated below so "two values in the paper" does not match.
_PAPER_COUNT_RE = re.compile(
    rf"\b(?P<count>{_COUNT_ALT})\b(?!-)"
    rf"(?P<middle>[^,.;:?!]{{0,100}}?)\bpaper(?:s)?\b",
    re.IGNORECASE,
)

# "works", "studies", and "methods" can mean non-paper entities.  Accept them
# only in the benchmark's explicit venue/year target form.
_SCOPED_ENTITY_COUNT_RE = re.compile(
    rf"\b(?P<count>{_COUNT_ALT})\b(?!-)"
    rf"(?P<middle>[^,.;:?!]{{0,100}}?)"
    rf"\b(?:works?|studies|methods?)\b",
    re.IGNORECASE,
)
_VENUE_YEAR_RE = re.compile(
    rf"\b(?:{_VENUE_ALT})\b[\s-]*\b20\d{{2}}\b",
    re.IGNORECASE,
)

_COUNT_OF_PAPERS_RE = re.compile(
    rf"\b(?P<count>{_COUNT_ALT})\b(?!-)\s+of\s+"
    rf"(?:these|those|the)\s+papers\b",
    re.IGNORECASE,
)
_BOTH_PAPERS_RE = re.compile(
    r"\bboth(?:\s+of)?(?:\s+(?:these|those|the))?\s+papers\b",
    re.IGNORECASE,
)
_BOTH_PAPER_PAIR_RE = re.compile(
    r"\bboth\s+the\b[^.?!;]{0,160}?\bpaper\b"
    r"[^.?!;]{0,80}?\band\b[^.?!;]{0,160}?\bpaper\b",
    re.IGNORECASE,
)

# These words make the number describe something inside/about a paper rather
# than the paper set itself.  Prepositions catch forms such as "two values in
# the paper"; the negative hyphen lookahead preserves adjectives such as
# "in-context".
_MIDDLE_BLOCKER_RE = re.compile(
    r"\b(?:of|in|on|from|by|with|using|against|versus|per|for)\b(?!-)"
    r"|\b(?:values?|results?|scores?|trials?|scenes?|samples?|tasks?|"
    r"keyframes?|texts?|stages?|modules?|parts?|types?|masks?|images?|"
    r"objects?|iterations?|matrices?|equations?|hypotheses|prompts?|"
    r"timesteps?)\b",
    re.IGNORECASE,
)


def _count_value(raw: str) -> int:
    return _COUNT_VALUES[raw.casefold()]


def explicit_target_paper_count(question: str) -> int | None:
    """Return a high-confidence explicit target-paper count, else ``None``.

    Multiple agreeing phrases are accepted.  Conflicting counts are treated as
    ambiguous rather than guessed.  The function intentionally does not infer a
    count from named methods or answer rows; that belongs to the later
    clause-aware selector.
    """
    text = question or ""
    counts: list[int] = []

    counts.extend(
        _count_value(match.group("count"))
        for match in _COUNT_OF_PAPERS_RE.finditer(text)
    )

    for match in _PAPER_COUNT_RE.finditer(text):
        if not _MIDDLE_BLOCKER_RE.search(match.group("middle")):
            counts.append(_count_value(match.group("count")))

    for match in _SCOPED_ENTITY_COUNT_RE.finditer(text):
        middle = match.group("middle")
        if _VENUE_YEAR_RE.search(middle) and not _MIDDLE_BLOCKER_RE.search(middle):
            counts.append(_count_value(match.group("count")))

    if _BOTH_PAPERS_RE.search(text) or _BOTH_PAPER_PAIR_RE.search(text):
        counts.append(2)

    unique = set(counts)
    return unique.pop() if len(unique) == 1 else None
