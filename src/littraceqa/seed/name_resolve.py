"""Fix B: per-NAME resolution for multi-paper questions.

A multi-paper question names SEVERAL distinct methods/models at once (e.g.
"2-step FID for TCM, sCT, ECM-XL, and IMM" -- four different papers). The
question-level tiers (`tier_parametric`/`tier_exact`/`tier_dense`) score the
question as a WHOLE and, fused into a single ranked list, typically surface
only the one or two most salient papers -- the others fall off the top_n
window even though each is named explicitly.

This module resolves EACH extracted name (`Anchor.named_titles` +
`Anchor.method_acronyms`) to a pool paper INDEPENDENTLY, so `SeedFinder` can
UNION every named paper into its result. Resolution is a precision-ordered
cascade per name:

  1. exact/normalized/squash title lookup (Fix A) -- highest precision, no LLM.
  2. acronym / title-initialism lookup (Fix A) -- strong if not too ambiguous.
  3. PARAMETRIC fallback: ask the LLM for the EXACT title of the single paper
     that introduces the name, then resolve THAT title to the pool via the
     same exact-then-fuzzy(+plausibility-gate) path `tier_parametric` uses.
  4. DENSE fallback: retrieve "<name> <question>" and accept the top hit ONLY
     if the (squashed) name is a substring of the (squashed) resolved title --
     a cheap, high-precision gate so a distinctive name ("YOLOv12") resolves
     while an incidental retrieval hit does not.

NO-FABRICATION: every returned `SeedCandidate.paper_id` comes from a real
`pool.by_id` lookup. NEVER raises -- an LLM/retriever failure on one name
degrades that name to "unresolved" (empty), never crashing the caller; the
`route` is always `"named"` so downstream can see these came from Fix B.
"""

from __future__ import annotations

import json
import re

from littraceqa.llm.interfaces import LLMClient
from littraceqa.retrieval.bm25 import tokenize
from littraceqa.retrieval.exact import ExactAcronymIndex, squash_title
from littraceqa.retrieval.interfaces import Retriever
from littraceqa.retrieval.pool import PoolIndex
from littraceqa.seed.interfaces import Anchor, SeedCandidate

_ROUTE = "named"

# Per-resolution-path scores (used only to ORDER the unioned named papers
# among themselves -- these candidates are appended by SeedFinder, not fused).
_TITLE_SCORE = 1.0
_ACRONYM_SCORE = 0.85
_PARAM_EXACT_SCORE = 0.9
_PARAM_FUZZY_SCORE = 0.5
_DENSE_SCORE = 0.4

# An acronym/initialism matching more than this many papers is too ambiguous
# to trust as a single-paper resolution -- fall through to the parametric
# path (which disambiguates by asking for the exact title) instead.
_ACRONYM_MAX_HITS = 10

# Fuzzy plausibility gate (mirrors tier_parametric): the LLM-proposed title and
# the resolved paper's actual title must share content tokens with at least
# this Jaccard, else the fuzzy hit is judged implausible and discarded.
_FUZZY_MIN_TITLE_OVERLAP = 0.5

_DENSE_K = 10

_NAME_SYSTEM_PROMPT = (
    "You are given the NAME or ACRONYM of a method, model, system, or "
    "benchmark taken from a research question. From your own training-time "
    "knowledge, give the EXACT title of the SINGLE paper that INTRODUCES or "
    "is NAMED by it -- if, and only if, you can recall the title with real "
    "confidence. Respond with STRICT minified JSON only -- no prose, no "
    'markdown, no code fences. The JSON object must have exactly one key: '
    '"title" (a string: the exact paper title, or an empty string if you '
    "cannot confidently recall one). Do not guess, paraphrase, or invent a "
    "title -- an empty string is preferred over a guess."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _title_token_overlap(proposed: str, actual: str) -> float:
    """Jaccard overlap of the content-token sets of two titles (0.0 if either
    has no content tokens). Shared with the retriever's own tokenization."""
    a = set(tokenize(proposed))
    b = set(tokenize(actual))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_title(response: str) -> str:
    """Defensively parse `{"title": "..."}`; any failure yields ""."""
    cleaned = _strip_code_fences(response or "")
    if not cleaned:
        return ""
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    title = parsed.get("title")
    return title.strip() if isinstance(title, str) else ""


def _parametric_resolve(
    name: str,
    question: str,
    llm: LLMClient,
    exact: ExactAcronymIndex,
    retriever: Retriever,
    pool: PoolIndex,
) -> SeedCandidate | None:
    """Ask the LLM for the exact title of the paper named `name`, then resolve
    that title to a real pool paper (exact, then fuzzy+plausibility-gate).
    Returns None on any failure or implausible resolution."""
    prompt = (
        f"Name/acronym: {name}\n"
        f"For context, it appears in this question: {question}\n\n"
        "Respond with the JSON object only."
    )
    try:
        response = llm.complete(prompt, system=_NAME_SYSTEM_PROMPT, temperature=0.0)
    except Exception:  # noqa: BLE001 -- degrade, never crash the caller
        return None

    title = _parse_title(response)
    if not title:
        return None

    exact_hits = exact.lookup_title(title)
    if exact_hits:
        pid = exact_hits[0]
        if pool.by_id(pid) is not None:
            return SeedCandidate(
                paper_id=pid,
                score=_PARAM_EXACT_SCORE,
                route=_ROUTE,
                reason=f"LLM named exact title for {name!r} (exact match): {title!r}",
            )

    try:
        fuzzy = retriever.retrieve(title, _DENSE_K)
    except Exception:  # noqa: BLE001
        fuzzy = []
    if not fuzzy:
        return None

    pid = fuzzy[0][0]
    paper = pool.by_id(pid)
    if paper is None:
        return None
    overlap = _title_token_overlap(title, paper.title)
    if overlap < _FUZZY_MIN_TITLE_OVERLAP:
        return None
    return SeedCandidate(
        paper_id=pid,
        score=_PARAM_FUZZY_SCORE,
        route=_ROUTE,
        reason=(
            f"LLM named title for {name!r} (fuzzy resolve, overlap "
            f"{overlap:.2f}): {title!r}"
        ),
    )


def _dense_resolve(
    name: str,
    question: str,
    retriever: Retriever,
    pool: PoolIndex,
) -> SeedCandidate | None:
    """Retrieve "<name> <question>" and accept the first hit whose (squashed)
    title CONTAINS the (squashed) name -- a high-precision gate so only a paper
    actually carrying the name resolves. Returns None otherwise."""
    name_sq = squash_title(name)
    if not name_sq:
        return None
    try:
        hits = retriever.retrieve(f"{name} {question}", _DENSE_K)
    except Exception:  # noqa: BLE001
        return None
    for pid, _score in hits:
        paper = pool.by_id(pid)
        if paper is not None and name_sq in squash_title(paper.title):
            return SeedCandidate(
                paper_id=pid,
                score=_DENSE_SCORE,
                route=_ROUTE,
                reason=f"dense resolution for {name!r} (title contains name)",
            )
    return None


def _resolve_one(
    name: str,
    question: str,
    llm: LLMClient,
    exact: ExactAcronymIndex,
    retriever: Retriever,
    pool: PoolIndex,
) -> list[SeedCandidate]:
    name = name.strip()
    if not name:
        return []

    # 1. exact / normalized / squash title (Fix A) -- no LLM.
    title_hits = [pid for pid in exact.lookup_title(name) if pool.by_id(pid) is not None]
    if title_hits:
        return [
            SeedCandidate(
                paper_id=pid,
                score=_TITLE_SCORE,
                route=_ROUTE,
                reason=f"exact/normalized title match for named signal {name!r}",
            )
            for pid in title_hits
        ]

    # 2. acronym / title-initialism (Fix A) -- no LLM, unless too ambiguous.
    acronym_hits = [
        pid for pid in exact.lookup_acronym(name) if pool.by_id(pid) is not None
    ]
    if acronym_hits and len(acronym_hits) <= _ACRONYM_MAX_HITS:
        return [
            SeedCandidate(
                paper_id=pid,
                score=_ACRONYM_SCORE,
                route=_ROUTE,
                reason=f"acronym/initialism match for named signal {name!r}",
            )
            for pid in acronym_hits
        ]

    # 3. parametric LLM fallback (exact-then-fuzzy+gate).
    cand = _parametric_resolve(name, question, llm, exact, retriever, pool)
    if cand is not None:
        return [cand]

    # 4. dense fallback (title-contains-name gate).
    cand = _dense_resolve(name, question, retriever, pool)
    if cand is not None:
        return [cand]

    return []


def resolve_names(
    anchor: Anchor,
    question: str,
    llm: LLMClient,
    exact: ExactAcronymIndex,
    retriever: Retriever,
    pool: PoolIndex,
) -> list[SeedCandidate]:
    """Resolve every extracted name (`named_titles` + `method_acronyms`) to
    pool papers, INDEPENDENTLY, deduped by paper_id (keeping the highest-scoring
    resolution). Deterministic order `(-score, paper_id)`; every paper_id is a
    real `pool.by_id`. NEVER raises."""
    best: dict[str, SeedCandidate] = {}
    for name in [*anchor.named_titles, *anchor.method_acronyms]:
        for cand in _resolve_one(name, question, llm, exact, retriever, pool):
            current = best.get(cand.paper_id)
            if current is None or cand.score > current.score:
                best[cand.paper_id] = cand
    return sorted(best.values(), key=lambda c: (-c.score, c.paper_id))
