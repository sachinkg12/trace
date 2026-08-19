"""Tier 2 (of the seed-finding cascade): exact/acronym recognition.

Turns an `Anchor` (already extracted from the question by Tier 1's LLM
call -- see `littraceqa.seed.anchor`) into `SeedCandidate`s via the no-LLM
`ExactAcronymIndex` lookup. Pure lookup, deterministic, no LLM call here.

Two signal sources, two precision levels:
  - `named_titles`: an EXACT title match is unambiguous -- score 1.0.
  - `method_acronyms`: a DISTINCTIVE acronym (few hits) is a strong signal
    -- score `_ACRONYM_SCORE`. A COMMON acronym (e.g. "LLM", "RAG") matches
    hundreds of papers and carries almost no identifying signal, so any
    acronym whose hit count exceeds `max_acronym_hits` is suppressed
    entirely rather than flooding the candidate list with low-precision
    noise.
"""

from __future__ import annotations

from littraceqa.retrieval.exact import ExactAcronymIndex
from littraceqa.seed.interfaces import Anchor, SeedCandidate

_TITLE_SCORE = 1.0
_ACRONYM_SCORE = 0.8


def tier_exact(
    anchor: Anchor,
    exact: ExactAcronymIndex,
    *,
    max_acronym_hits: int = 25,
) -> list[SeedCandidate]:
    """Look up `anchor`'s named titles and method acronyms in `exact`.

    Dedups by paper_id, keeping the highest-scoring candidate when a paper
    is reachable via more than one signal (e.g. both its exact title and a
    distinctive acronym it carries). Sorted by `(-score, paper_id)` for a
    deterministic, most-confident-first order.
    """
    best: dict[str, SeedCandidate] = {}

    def _consider(candidate: SeedCandidate) -> None:
        current = best.get(candidate.paper_id)
        if current is None or candidate.score > current.score:
            best[candidate.paper_id] = candidate

    for title in anchor.named_titles:
        for paper_id in exact.lookup_title(title):
            _consider(
                SeedCandidate(
                    paper_id=paper_id,
                    score=_TITLE_SCORE,
                    route="exact",
                    reason=f"exact title match: {title!r}",
                )
            )

    for acronym in anchor.method_acronyms:
        hits = exact.lookup_acronym(acronym)
        if not hits or len(hits) > max_acronym_hits:
            # Unknown, or too common to be identifying -- suppress rather
            # than flood the candidate list with low-precision noise.
            continue
        for paper_id in hits:
            _consider(
                SeedCandidate(
                    paper_id=paper_id,
                    score=_ACRONYM_SCORE,
                    route="exact",
                    reason=f"distinctive acronym match: {acronym!r}",
                )
            )

    return sorted(best.values(), key=lambda c: (-c.score, c.paper_id))
