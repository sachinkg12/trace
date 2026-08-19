"""Paper-set selection: fuse seed candidates (#4) and evidence-bearing
papers (#5) into the final paper_id set, precision-first.

The base set is the union of every evidence paper (we grounded an answer
there) and the top seed(s) -- both high-confidence. Seed-kNN expansion is
the ONLY recall lever and is deliberately gated: it fires only when the
question signals multiple papers AND an expander is injected, because the
curated thematic families are hard to recover (measured) and spraying
neighbours collapses precision on the F1 metric. No LLM, fully deterministic.
"""
from __future__ import annotations

from typing import Iterable

from littraceqa.paperset.interfaces import SeedKnnExpander
from littraceqa.seed.interfaces import SeedCandidate


def _dedup_preserve_order(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for pid in ids:
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


class PaperSetSelector:
    def __init__(self, *, expander: SeedKnnExpander | None = None,
                 expand_k: int = 10, max_single_seeds: int = 1):
        self._expander = expander
        self._expand_k = expand_k
        self._max_single_seeds = max_single_seeds

    def select(self, seeds: list[SeedCandidate], evidence_paper_ids: list[str],
               *, asks_multiple: bool = False) -> list[str]:
        # Seeds highest-score-first; stable so equal scores keep input order.
        ranked = sorted(seeds, key=lambda s: s.score, reverse=True)
        n = len(ranked) if asks_multiple else min(self._max_single_seeds, len(ranked))
        chosen_seed_ids = [s.paper_id for s in ranked[:n]]

        # Base set: evidence papers first (strongest signal), then chosen seeds.
        ids: list[str] = list(evidence_paper_ids) + chosen_seed_ids

        # Gated recall expansion. The centroid basis is DELIBERATELY every
        # seed (`ranked`), not just the chosen top-`n` — a richer centroid
        # locates thematic-family siblings better. Do not narrow this to
        # `chosen_seed_ids`. (In the asks_multiple branch that fires expansion,
        # n == len(ranked) anyway, so the two are identical today; this note
        # guards against a future reader "simplifying" the two into one.)
        all_seed_ids = [s.paper_id for s in ranked]
        if asks_multiple and self._expander is not None and all_seed_ids:
            try:
                neighbours = self._expander.expand(all_seed_ids, self._expand_k)
            except Exception:
                neighbours = []
            ids.extend(pid for pid, _score in neighbours)

        return _dedup_preserve_order(ids)
