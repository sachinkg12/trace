"""`SeedFinder`: the seed-finding orchestrator.

Fuses the three recognition tiers -- Tier 1 parametric (`tier_parametric`,
LLM naming from memory), Tier 2 exact/acronym (`tier_exact`), Tier 3 dense
retrieval (`tier_dense`) -- plus abstract-level confirmation
(`confirm_candidate`) into a single ranked list of `SeedCandidate`s for a
question. This is the payoff of the whole seed-finding subsystem: everything
built in Tasks 1-4 is a component this orchestrator wires together.

FUSION CHOICE: sum-of-tier-scores. Each tier's `SeedCandidate` list is
already deduped per-tier (each tier returns at most one candidate per
paper_id), so merging across tiers and summing the scores of every tier that
proposed a given paper_id is a simple, explainable way to reward
corroborating evidence -- a paper independently recognized by two or three
tiers scores higher than one recognized by only one, even if that one tier's
own score is the highest single score around (see the dedicated multi-tier
-boost test in tests/seed/test_finder.py). This is deliberately simpler than
RRF-over-three-lists: RRF only uses rank order and would throw away the
tiers' own calibrated score gaps (e.g. exact's 1.0 vs. 0.8, parametric's 0.9
vs. 0.5) that already encode real confidence differences.

CONFIRMATION FOLD-IN: only the top `_CONFIRM_TOP_K` fused candidates get an
abstract-level `confirm_candidate` call (bounding LLM cost -- confirming the
whole candidate list would be one extra call per candidate). Its confidence
nudges (not dominates) the fused score: `+ _CONFIRM_WEIGHT * confidence` if
confirmed a match, `- _CONFIRM_WEIGHT * confidence` if not.

NO-FABRICATION: every raw tier candidate is re-validated against
`pool.by_id` before fusion -- belt-and-suspenders on top of what
tier_exact/tier_parametric already guarantee internally, since tier_dense's
candidates come straight from whatever a `Retriever` returns and aren't
otherwise checked against the pool.

GRACEFUL DEGRADATION: every tier call (and confirmation) is wrapped so a
single failure -- an LLM call raising, a retriever backend raising -- never
crashes `find()`; it simply drops that source's contribution and the
remaining tiers still produce a result. `extract_anchor`/`tier_parametric`/
`confirm_candidate` already catch LLM failures internally (see their own
modules), but `tier_dense` does NOT catch a raising `Retriever` -- so this
wrapping is genuinely load-bearing, not just redundant defense (see
`test_dense_tier_retriever_failure_does_not_crash_find` in
tests/seed/test_finder.py).
"""

from __future__ import annotations

from littraceqa.llm.interfaces import LLMClient
from littraceqa.retrieval.exact import ExactAcronymIndex
from littraceqa.retrieval.interfaces import Retriever
from littraceqa.retrieval.pool import PoolIndex
from littraceqa.seed.anchor import extract_anchor
from littraceqa.seed.confirm import confirm_candidate
from littraceqa.seed.interfaces import Anchor, SeedCandidate
from littraceqa.seed.name_resolve import resolve_names
from littraceqa.seed.tier_dense import tier_dense
from littraceqa.seed.tier_exact import tier_exact
from littraceqa.seed.tier_parametric import tier_parametric

# How many top-fused candidates get an abstract-level confirmation call.
# Bounds LLM cost: confirming the whole candidate list would be one extra
# call per candidate.
_CONFIRM_TOP_K = 5

# How much confirm_candidate's confidence shifts a candidate's fused score:
# a nudge, not a dominant term -- kept well below a single tier's own score
# range (exact: 0.8-1.0, parametric: 0.5-0.9) so confirmation refines the
# tier-fused ranking rather than overriding it.
_CONFIRM_WEIGHT = 0.3


class SeedFinder:
    """Orchestrates the recognition-first seed-finding cascade for a question.

    DIP: constructed from an `LLMClient`, `PoolIndex`, `Retriever`, and
    `ExactAcronymIndex` -- all abstractions, so tests inject `FakeLLM` and a
    scripted fake retriever, while production injects real Gemini, the real
    27k-paper pool, and the real hybrid retriever, with zero changes to
    `find()`.
    """

    def __init__(
        self,
        llm: LLMClient,
        pool: PoolIndex,
        retriever: Retriever,
        exact: ExactAcronymIndex,
    ):
        self.llm = llm
        self.pool = pool
        self.retriever = retriever
        self.exact = exact

    def find(self, question: str, *, top_n: int = 3) -> list[SeedCandidate]:
        """Return up to `top_n` ranked `SeedCandidate`s for `question`.

        1. Extract an `Anchor` (paper-identifying signals) from the question.
        2. Gather raw candidates from all three tiers (parametric, exact,
           dense), each wrapped so one failing tier degrades to "no
           contribution" rather than crashing the whole call.
        3. Validate every raw candidate against `pool.by_id` (no fabrication).
        4. Fuse by paper_id: sum the scores of every tier that proposed it.
        5. Confirm the top `_CONFIRM_TOP_K` fused candidates against their
           abstract, folding confidence into the final score.
        6. Take the top `top_n`, sorted by `(-score, paper_id)`.
        7. Multi-name UNION (Fix B): resolve EACH extracted name
           (`named_titles` + `method_acronyms`) to a pool paper independently
           and append any that the fused top_n missed -- so a question naming
           several distinct papers returns all of them, not just the one or
           two most salient. Single-name questions are unaffected (their one
           named paper is already in the top_n, so nothing is appended).
        """
        anchor = self._safe_anchor(question)

        raw_candidates: list[SeedCandidate] = []
        raw_candidates += self._safe_tier(
            lambda: tier_parametric(
                self.llm, question, self.exact, self.retriever, self.pool
            )
        )
        raw_candidates += self._safe_tier(lambda: tier_exact(anchor, self.exact))
        raw_candidates += self._safe_tier(
            lambda: tier_dense(question, anchor, self.retriever)
        )

        # No-fabrication, belt-and-suspenders: drop any candidate whose
        # paper_id isn't a real pool paper before it can influence fusion or
        # be handed to confirm_candidate (which needs a real Paper).
        real_candidates = [
            c for c in raw_candidates if self.pool.by_id(c.paper_id) is not None
        ]

        fused = self._fuse(real_candidates)
        confirmed = self._confirm_top(question, fused)

        ranked = sorted(confirmed, key=lambda c: (-c.score, c.paper_id))[:top_n]

        # Fix B: per-name resolution, unioned in AFTER top_n so each named
        # paper is guaranteed present even when top_n would clip it. Kept out
        # of `_fuse` on purpose: fusing these would double-count names already
        # scored by tier_exact and shift the fused ranking; this pass only
        # ADDS papers the fused top_n didn't already return.
        named = self._safe_tier(
            lambda: resolve_names(
                anchor, question, self.llm, self.exact, self.retriever, self.pool
            )
        )
        return self._union_named(ranked, named)

    def _safe_anchor(self, question: str) -> Anchor:
        # extract_anchor is documented to never raise (it degrades to an
        # empty Anchor internally), but this call is wrapped anyway so a
        # future change to that contract can't take find() down with it.
        try:
            return extract_anchor(self.llm, question)
        except Exception:  # noqa: BLE001 -- degrade, never crash find()
            return Anchor(raw={"error": "anchor extraction failed at finder level"})

    def _safe_tier(self, fn) -> list[SeedCandidate]:
        try:
            return fn()
        except Exception:  # noqa: BLE001 -- one failing tier must never crash find()
            return []

    def _union_named(
        self, ranked: list[SeedCandidate], named: list[SeedCandidate]
    ) -> list[SeedCandidate]:
        """Append per-name resolved papers (Fix B) that the fused top_n didn't
        already return, in `(-score, paper_id)` order. Dedups by paper_id
        (keeping the already-present fused candidate) and re-validates each
        appended id against the pool (no fabrication), so the only effect is to
        WIDEN the result set for multi-paper questions -- never to reorder or
        rescore the fused candidates a single-name question already returns."""
        result = list(ranked)
        have = {c.paper_id for c in result}
        for cand in sorted(named, key=lambda c: (-c.score, c.paper_id)):
            if cand.paper_id in have:
                continue
            if self.pool.by_id(cand.paper_id) is None:
                continue
            result.append(cand)
            have.add(cand.paper_id)
        return result

    def _fuse(self, candidates: list[SeedCandidate]) -> list[SeedCandidate]:
        """Merge/dedup by paper_id, summing the score of every tier that
        proposed it -- so a paper corroborated by multiple tiers outranks
        one proposed by only a single tier (see module docstring). Note this
        corroboration isn't always fully independent evidence: when a title
        is named in the question, both `tier_exact` and `tier_parametric`'s
        exact-resolution path key off the same `exact.lookup_title` signal,
        so their scores can be partly self-correlated rather than two truly
        independent recognitions of the same paper."""
        grouped: dict[str, list[SeedCandidate]] = {}
        for c in candidates:
            grouped.setdefault(c.paper_id, []).append(c)

        fused: list[SeedCandidate] = []
        for paper_id, group in grouped.items():
            total_score = sum(c.score for c in group)
            routes = "+".join(sorted({c.route for c in group}))
            reason = "; ".join(c.reason for c in group)
            fused.append(
                SeedCandidate(
                    paper_id=paper_id, score=total_score, route=routes, reason=reason
                )
            )
        return fused

    def _confirm_top(
        self, question: str, fused: list[SeedCandidate]
    ) -> list[SeedCandidate]:
        """Confirm the top `_CONFIRM_TOP_K` fused candidates against their
        abstract, folding confidence into the final score. Candidates beyond
        the top K keep their fused score unchanged (no confirm call).

        NOTE: `confirm_candidate` is abstract-level only and can false-negative
        on the correct paper (see the CAVEAT in `confirm.py`'s docstring) --
        that's why its confidence only nudges the score (`_CONFIRM_WEIGHT`)
        rather than gating/eliminating a candidate outright.
        """
        ordered = sorted(fused, key=lambda c: (-c.score, c.paper_id))
        top, rest = ordered[:_CONFIRM_TOP_K], ordered[_CONFIRM_TOP_K:]

        result: list[SeedCandidate] = []
        for c in top:
            paper = self.pool.by_id(c.paper_id)
            if paper is None:
                # Already filtered out in find(), but defensive here too --
                # never call confirm_candidate without a real Paper.
                result.append(c)
                continue
            try:
                is_match, confidence, reason = confirm_candidate(
                    self.llm, question, paper
                )
            except Exception:  # noqa: BLE001 -- degrade, never crash find()
                result.append(c)
                continue

            delta = _CONFIRM_WEIGHT * confidence if is_match else -_CONFIRM_WEIGHT * confidence
            result.append(
                SeedCandidate(
                    paper_id=c.paper_id,
                    score=c.score + delta,
                    route=c.route,
                    reason=f"{c.reason} | confirm({is_match}, {confidence:.2f}): {reason}",
                )
            )
        return result + rest
