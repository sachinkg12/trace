"""Seed-finding data model: `SeedCandidate` (a scored, tiered guess at
which paper a question is about) and `Anchor` (the paper-identifying
signals pulled out of a question by `extract_anchor`).

Kept dependency-free (no imports from `littraceqa.llm`/`retrieval`) so
every seed-finding module can depend on these types without coupling to
how candidates are produced or anchors extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeedCandidate:
    """A single ranked guess at which paper a question targets.

    `route` names the recognition tier that produced it -- `"parametric"`
    (LLM naming the paper from memory), `"exact"` (title/acronym lookup),
    or `"dense"` (embedding retrieval) -- so downstream fusion (#5) can
    weigh evidence by tier and callers can explain *why* a candidate was
    proposed via `reason`.

    `score` is an UNCALIBRATED fused rank magnitude -- a sum of tier scores
    (plus/minus a confirm-step delta once fused by `SeedFinder`), not a
    probability. It can exceed 1.0 (e.g. ~1.9-3.3 when multiple tiers agree
    on the same paper -- see `SeedFinder._fuse`). It orders candidates
    within a single `find()` call but is NOT a [0, 1] confidence/probability
    and should NOT be read as one; downstream consumers (#5/#6) should treat
    the ORDERING as a prior, not the value itself as calibrated.
    """

    paper_id: str
    score: float
    route: str
    reason: str


@dataclass(frozen=True)
class Anchor:
    """Paper-identifying signals extracted from a question by one LLM call.

    `raw` always holds the parsed JSON dict (or, on a parse failure, an
    `{"error": ...}` dict) so callers/tests can inspect exactly what the
    LLM returned without re-deriving it.

    `asks_multiple` is ADVISORY only: `extract_anchor` parses it faithfully
    from the LLM's JSON, but nothing in this subsystem currently consumes
    it -- `SeedFinder.find()` always returns a single ranked `top_n` list
    and does not branch on it. It is carried here for downstream (#5/#6) to
    decide whether to widen the paper set (e.g. retrieve/confirm against
    more than one candidate paper) when a question spans multiple papers.
    """

    named_titles: list[str] = field(default_factory=list)
    method_acronyms: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    asks_multiple: bool = False
    raw: dict = field(default_factory=dict)
