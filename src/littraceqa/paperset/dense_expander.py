"""Seed-kNN expander backed by the dense retriever's passage-to-passage
centroid search. Registered `"dense"`. A Vertex-backed expander swaps in
later behind the same Protocol by registration.
"""
from __future__ import annotations

from littraceqa.paperset.interfaces import register_expander


@register_expander("dense")
class DenseSeedKnnExpander:
    def __init__(self, retriever):
        # `retriever` exposes retrieve_by_seed(seed_ids, k) -> list[(id, score)];
        # littraceqa.retrieval.dense.DenseRetriever qualifies.
        self._retriever = retriever

    def expand(self, seed_ids: list[str], k: int) -> list[tuple[str, float]]:
        if not seed_ids:                 # backend raises on empty; degrade quietly
            return []
        return self._retriever.retrieve_by_seed(seed_ids, k)
