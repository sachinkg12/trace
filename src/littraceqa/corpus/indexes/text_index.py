"""`BM25TextIndex`: a thin, reusable BM25 text index for the Level-2 builders.

Wraps the verified `retrieval.bm25.BM25Index` (K1=1.5/B=0.75, +1-smoothed idf)
and its `tokenize` so the passages AND objects indexes share ONE search path --
no tokenizer or scoring math is re-implemented here. Adds only a generic
id/text -> `search(query, k) -> [(id, score)]` surface with the same
deterministic `(-score, id)` tie-break the pool retriever uses.
"""
from __future__ import annotations

from littraceqa.retrieval.bm25 import BM25Index, tokenize


class BM25TextIndex:
    """BM25 over arbitrary (id, text) documents, reusing `retrieval.bm25`."""

    def __init__(self, ids: list[str], texts: list[str]):
        assert len(ids) == len(texts)
        self.ids = list(ids)
        self._index = BM25Index(self.ids, [tokenize(t or "") for t in texts])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-`k` doc ids for `query`, score-descending, tie-broken by id
        ascending (deterministic even when raw BM25 scores tie exactly)."""
        scores = self._index.score_all(tokenize(query or ""))
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.ids[kv[0]]))[:k]
        return [(self.ids[idx], score) for idx, score in ranked]
