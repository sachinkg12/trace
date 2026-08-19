"""`RRFHybridRetriever`: Reciprocal Rank Fusion over multiple `Retriever`s.

Fuses N component retrievers (e.g. `dense` + `bm25`) into a single ranking
without needing their raw scores to be on comparable scales -- RRF only uses
each component's *rank order*, which is what makes it a robust default for
hybrid search.

For each component, we over-fetch (`max(k, _MIN_FETCH)`) so that fusion has
enough candidates to work with even when the caller asks for a small `k`.
Each component is free to return fewer hits than requested (or none at all --
e.g. BM25 on a query with no lexical overlap); this is a normal, expected
consumer-contract case (not an error), so the fusion loop below simply
enumerates whatever each component returns and never indexes past the end of
a short/empty list.
"""
from __future__ import annotations

from littraceqa.retrieval.interfaces import Retriever, register_retriever

_MIN_FETCH = 100


@register_retriever("rrf")
class RRFHybridRetriever:
    """Fuses `retrievers` via Reciprocal Rank Fusion.

    score(paper_id) = sum over components c that ranked paper_id at position
    `rank` (1-based) of `1 / (rrf_k + rank)`. Components that don't rank a
    paper simply don't contribute a term for it.
    """

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60):
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        fetch_k = max(k, _MIN_FETCH)
        fused: dict[str, float] = {}
        for retriever in self.retrievers:
            hits = retriever.retrieve(query, fetch_k)
            # `hits` may be shorter than `fetch_k` (or empty) -- enumerate
            # only what's actually returned; no indexing past the end.
            for rank, (paper_id, _score) in enumerate(hits, start=1):
                fused[paper_id] = fused.get(paper_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
