"""Tier 3 (of the seed-finding cascade): dense/hybrid retrieval recognition.

Falls back to a `Retriever` (dense embedding search, or a hybrid of dense +
BM25 -- see `littraceqa.retrieval`) when the higher-precision tiers (1:
parametric LLM naming, 2: exact/acronym lookup -- `tier_exact`) don't
confidently identify a paper. Retrieval is anchor-boosted: any signals
`extract_anchor` already pulled out of the question (named titles, method
acronyms, datasets) are appended to the query so the retriever has more to
match on than the bare question text.

Scoring: retrievers return raw, backend-specific, UNBOUNDED scores (cosine
similarity, BM25, RRF-fused sums, ...) that aren't comparable across
backends and can't be fused with the exact tier's calibrated 0.8-1.0 scores
(#5 -- score fusion) without distortion. Instead we score by reciprocal
rank, `1 / (_RRF_K + rank)` (1-based rank, same `_RRF_K = 60` constant used
by `RRFHybridRetriever`), and rescale so the best possible score (rank 1)
lands at `_MAX_SCORE = 0.6` -- comfortably below the exact tier's 0.8 floor,
so a dense guess can never outrank a real exact/acronym match, while still
preserving relative order among dense candidates themselves.
"""

from __future__ import annotations

from littraceqa.retrieval.interfaces import Retriever
from littraceqa.seed.interfaces import Anchor, SeedCandidate

_RRF_K = 60
_MAX_SCORE = 0.6
_SCALE = _MAX_SCORE * (_RRF_K + 1)  # normalizes rank-1's 1/(_RRF_K+1) to _MAX_SCORE


def _build_query(question: str, anchor: Anchor) -> str:
    """Anchor-boost the retrieval query: the bare question plus any
    title/acronym/dataset terms the LLM already extracted, so retrieval has
    more identifying signal to match on than free text alone."""
    terms = [*anchor.named_titles, *anchor.method_acronyms, *anchor.datasets]
    if not terms:
        return question
    return " ".join([question, *terms])


def tier_dense(
    question: str,
    anchor: Anchor,
    retriever: Retriever,
    *,
    k: int = 10,
) -> list[SeedCandidate]:
    """Retrieve candidate papers for `question` (anchor-boosted) via
    `retriever`, mapping each hit to a `SeedCandidate` scored by reciprocal
    rank rather than the retriever's raw score, so it's bounded and
    comparable across backends. Preserves the retriever's own (already
    best-first) order -- no re-sort needed, so ties are exactly as the
    retriever broke them, not re-broken here.
    """
    query = _build_query(question, anchor)
    # Defensive `[:k]`: the contract is "at most k hits", but a misbehaving
    # retriever backend that returns more than k rows must not be allowed to
    # emit more than k SeedCandidates -- cheap insurance against a contract
    # violation upstream.
    hits = retriever.retrieve(query, k)[:k]

    return [
        SeedCandidate(
            paper_id=paper_id,
            score=_SCALE / (_RRF_K + rank),
            route="dense",
            reason=f"dense retrieval rank {rank} for query {query!r}",
        )
        for rank, (paper_id, _raw_score) in enumerate(hits, start=1)
    ]
