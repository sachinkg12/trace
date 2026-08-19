"""`DenseRetriever`: cosine (dot-on-normalized) retrieval over the pool
embedding matrix, plus `retrieve_by_seed`, the seed-kNN primitive the
seed-kNN candidate generator (tracker #6) reuses.

Both `retrieve` and `retrieve_by_seed` embed/compose SYMMETRICALLY -- no
query-instruction prefix is ever applied (see `embedder_local.py`'s module
docstring): `retrieve`'s query text is encoded the same way pool documents
are, and `retrieve_by_seed` is pure passage-to-passage (mean of already
normalized pool rows, re-normalized).
"""

from __future__ import annotations

import numpy as np

from littraceqa.retrieval.embedder_local import embed_pool
from littraceqa.retrieval.interfaces import Embedder, register_retriever
from littraceqa.retrieval.pool import PoolIndex


def _topk_sorted(scores: np.ndarray, ids: list[str], k: int) -> list[tuple[str, float]]:
    """Rank `ids` by `scores` descending, tie-broken by `paper_id` ascending,
    and return the top `k` as `(paper_id, score)` pairs.

    Sorting (rather than `argpartition`) keeps this simple and deterministic
    at the pool's current scale (~27k); it is not a hot loop.
    """
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    top = order[:k]
    return [(ids[i], float(scores[i])) for i in top]


@register_retriever("dense")
class DenseRetriever:
    """Dense retriever built from an injected `Embedder` + `PoolIndex`.

    Precomputes/loads the full pool embedding matrix on construction (via
    `embed_pool`, which is itself disk-cached) so `retrieve`/`retrieve_by_seed`
    calls are cheap.
    """

    def __init__(self, pool: PoolIndex, embedder: Embedder):
        self.pool = pool
        self.embedder = embedder
        self.matrix, self.ids = embed_pool(pool, embedder)
        self._id_to_row = {pid: i for i, pid in enumerate(self.ids)}

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Embed `query` with NO prefix (symmetric with passage encoding),
        dot against the normalized pool matrix, return top-k sorted by
        score descending, tie-broken by paper_id."""
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        return _topk_sorted(scores, self.ids, k)

    def retrieve_by_seed(self, seed_ids: list[str], k: int) -> list[tuple[str, float]]:
        """Passage-to-passage seed-kNN: mean of the seed papers' (already
        normalized) pool-embedding rows, re-normalized, ranked against the
        pool with the seed ids excluded from results."""
        if not seed_ids:
            raise ValueError("retrieve_by_seed requires at least one seed id")
        rows = []
        for sid in seed_ids:
            if sid not in self._id_to_row:
                raise KeyError(f"seed id not in pool: {sid!r}")
            rows.append(self.matrix[self._id_to_row[sid]])
        seed_rows = np.stack(rows)
        centroid = seed_rows.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        scores = self.matrix @ centroid

        seed_set = set(seed_ids)
        keep = [i for i, pid in enumerate(self.ids) if pid not in seed_set]
        kept_ids = [self.ids[i] for i in keep]
        kept_scores = scores[keep]
        return _topk_sorted(kept_scores, kept_ids, k)
