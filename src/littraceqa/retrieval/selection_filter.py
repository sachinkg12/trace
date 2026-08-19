"""Deterministic venue/year hard-constraint filter for paper selection.

Applied to the retrieved candidates BEFORE any ranking policy, for EVERY policy
(property_first, cascade, bypass), so a question that states a venue/year
constraint never emits a paper from the wrong venue/year (audit bug #1: 41/76
emitted papers on the 24 venue-constrained test questions violated the stated
venue). Independent of the planner -- which can silently degrade and drop the
constraint -- the constraint is parsed directly from the question text against
the pool's fixed venue vocabulary (the nine venues that appear as the
`<venue><year>_<n>` paper_id prefix).

SAFE BY CONSTRUCTION: never filters the candidate set to empty. If no candidate
satisfies the parsed constraint (e.g. a spurious venue mention, or the gold
genuinely lives elsewhere), the UNFILTERED set is returned -- the filter can only
remove clear violators when a valid alternative exists, so it never destroys
recall. Composes as `filter -> ranking`, never replacing the ranking policy.
"""
from __future__ import annotations

import re

# The nine venue prefixes present in the pool's paper_ids. Ordered longest-first
# so a substring venue never shadows a longer one during matching (word
# boundaries already prevent "acl" matching inside "naacl", but explicit order
# documents the intent).
_POOL_VENUES = ("neurips", "emnlp", "iclr", "icml", "cvpr", "iccv", "eccv", "naacl", "acl")
_VENUE_RE = {v: re.compile(rf"\b{v}\b", re.IGNORECASE) for v in _POOL_VENUES}
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_PID_RE = re.compile(r"^([a-z]+)(\d{4})")

# A venue/year is a hard constraint only when it belongs to the TARGET paper
# phrase, not merely any paper mentioned by the question.  For example, in
# "Two ICCV 2025 papers cite the CVPR paper ...", ICCV 2025 is the target scope
# and CVPR describes the cited source.  Pair the venue with its adjacent year and
# a following paper-like head noun; do not independently collect every venue and
# every year from the entire question.
_VENUE_ALT = "|".join(_POOL_VENUES)
_TARGET_SCOPE_RE = re.compile(
    rf"\b(?P<venue>{_VENUE_ALT})"
    rf"(?:[\s-]*(?P<year>20\d{{2}}))?\b"
    rf"[^.?!;:]{{0,80}}?"
    rf"\b(?P<noun>papers?|works?|studies|methods?)\b",
    re.IGNORECASE,
)
_TARGET_SCOPE_YEAR_FIRST_RE = re.compile(
    rf"\b(?P<year>20\d{{2}})[\s-]+(?P<venue>{_VENUE_ALT})\b"
    rf"[^.?!;:]{{0,80}}?"
    rf"\b(?P<noun>papers?|works?|studies|methods?)\b",
    re.IGNORECASE,
)


def parse_venue_year(paper_id: str) -> tuple[str | None, str | None]:
    """Parse canonical ``venueYEAR_identifier`` strings into venue and year."""
    m = _PID_RE.match(paper_id or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def constraint_from_question(question: str) -> tuple[set[str], set[str]]:
    """Parse the target paper phrase's hard venue/year constraint.

    Prefer a scoped phrase such as ``ICCV 2025 papers`` or ``CVPR 2025 work``.
    This prevents a cited-paper mention later in the question from broadening the
    allowed target venues/years.  When no scoped phrase exists, retain the old
    behavior only for an unambiguous single venue (and a single year); multiple
    unscoped venues/years are ambiguous and therefore produce no hard filter.

    A bare year is intentionally never a constraint -- it may describe a
    baseline rather than the target papers.
    """
    q = question or ""
    scoped = [
        *_TARGET_SCOPE_RE.finditer(q),
        *_TARGET_SCOPE_YEAR_FIRST_RE.finditer(q),
    ]
    if scoped:
        # An explicit venue+year target is stronger than a venue-only paper
        # mention (the latter is commonly a cited source).  Preserve question
        # order among equally-specific matches.
        chosen = min(
            scoped,
            key=lambda m: (m.group("year") is None, m.start()),
        )
        venue = chosen.group("venue").casefold()
        year = chosen.group("year")
        return {venue}, ({year} if year else set())

    venues = {v for v in _POOL_VENUES if _VENUE_RE[v].search(q)}
    if len(venues) != 1:
        return set(), set()
    years = set(_YEAR_RE.findall(q))
    return venues, (years if len(years) == 1 else set())


def _satisfies(paper_id: str, venues: set[str], years: set[str]) -> bool:
    v, y = parse_venue_year(paper_id)
    if venues and v not in venues:
        return False
    if years and y not in years:
        return False
    return True


def filter_by_venue_year(candidates, question: str, *, get_id=lambda c: c.paper_id):
    """Drop candidates whose paper_id venue/year violates the question's stated
    constraint. Returns the input unchanged when the question states no
    constraint, and never returns an empty list when the input was non-empty
    (recall-safe: a constraint with zero satisfying candidates is ignored)."""
    venues, years = constraint_from_question(question)
    if not venues and not years:
        return candidates
    kept = [c for c in candidates if _satisfies(get_id(c), venues, years)]
    return kept if kept else candidates
