"""Tier 1 (of the seed-finding cascade): parametric recognition.

Asks the LLM, from its own training-time memory (no retrieval, no document
in context), which paper the question is about -- it may name zero, one, or
several EXACT titles. This is the highest-recall but also the least
trustworthy tier: an LLM can confidently name a paper that does not exist in
our pool (or exists under a slightly different title), so every proposed
title MUST be resolved against a real pool paper before it is allowed to
become a `SeedCandidate`.

NO-FABRICATION is the load-bearing rule here: a proposed title resolves via
`exact.lookup_title` first (an exact, normalized string match -- the LLM
quoted the real title verbatim); if that comes up empty, we fall back to
`retriever.retrieve(title, k)` and consider its top-1 hit -- but only accept
it if a retriever-agnostic PLAUSIBILITY GATE passes: the LLM's proposed
title and the resolved paper's ACTUAL title must share enough content (token
Jaccard >= `_FUZZY_MIN_TITLE_OVERLAP`). A retriever will always return
*some* top hit for any query that shares even one non-stopword token with
any pool paper, so accepting it unconditionally would let a hallucinated or
garbled LLM title resolve to a real-but-WRONG paper; the gate rejects those.
If the exact path is empty and the fuzzy top hit fails the gate (or the
retriever returns nothing), the proposed title is discarded outright -- we
never mint a `SeedCandidate` for a paper_id that didn't come from the pool
via one of these two real lookups, and never for one whose title doesn't
actually resemble what the LLM proposed.
"""

from __future__ import annotations

import json
import re

from littraceqa.llm.interfaces import LLMClient
from littraceqa.retrieval.bm25 import tokenize
from littraceqa.retrieval.exact import ExactAcronymIndex
from littraceqa.retrieval.interfaces import Retriever
from littraceqa.retrieval.pool import PoolIndex
from littraceqa.seed.interfaces import SeedCandidate

_SYSTEM_PROMPT = (
    "You are answering from your own training-time knowledge only -- you "
    "have NOT been given any document or search results. Given a question "
    "about an academic paper, name the EXACT title(s) of the paper(s) you "
    "recall the question is about, if -- and only if -- you can recall the "
    "title with real confidence. Respond with STRICT minified JSON only -- "
    "no prose, no markdown, no code fences. The JSON object must have "
    'exactly one key: "titles" (a list of strings: the exact title(s) of '
    "the paper(s) you believe this question concerns; an empty list if you "
    "cannot confidently recall one). Do not guess, paraphrase, or invent a "
    "title -- an empty list is a valid and preferred answer over a guess."
)

# Matches an optional ```json / ``` opening fence and a trailing ``` closing
# fence, so a fenced response is stripped down to the bare JSON body first.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# Resolved via an exact, normalized title match -- the LLM quoted the real
# title verbatim, so this is high-confidence (on par with tier_exact's own
# title-match score), though still a notch below it since a parametric
# recall carries more uncertainty about faithfulness than a direct lookup
# of a signal already present in the question text.
_EXACT_RESOLUTION_SCORE = 0.9

# Resolved only via the retriever's top-1 hit for the proposed title (no
# exact match) -- a much weaker signal, since the LLM's proposed title
# merely landed close to a real paper rather than matching it exactly.
# Kept below the exact tier's acronym floor (0.8) so a fuzzy parametric
# guess never outranks a real exact/acronym match downstream (#5 fusion).
_FUZZY_RESOLUTION_SCORE = 0.5

# Plausibility gate for the fuzzy path: the LLM-proposed title and the
# resolved paper's actual title must have a token Jaccard AT LEAST this high
# for the fuzzy resolution to be accepted. Below it, the top hit is judged an
# implausible match (the LLM's title merely shares an incidental token with
# it) and is discarded. Deterministic, no LLM/embeddings -- just normalized
# token-set overlap. 0.5 = at least half of the combined vocabulary of the
# two titles is shared.
_FUZZY_MIN_TITLE_OVERLAP = 0.5


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _title_token_overlap(proposed_title: str, actual_title: str) -> float:
    """Jaccard overlap of the content-token SETS of two titles.

    Reuses `bm25.tokenize` (lowercased alphanumeric tokens, minus the shared
    stopword list, minus single chars) so the gate is consistent with how the
    retriever itself tokenizes. Returns 0.0 if either title has no content
    tokens (so an empty/garbage proposed title can never pass the gate).
    """
    proposed = set(tokenize(proposed_title))
    actual = set(tokenize(actual_title))
    if not proposed or not actual:
        return 0.0
    return len(proposed & actual) / len(proposed | actual)


def _as_str_list(value: object) -> list[str]:
    """Coerce a parsed JSON value into `list[str]`, defensively (mirrors
    `littraceqa.seed.anchor._as_str_list`). Non-string list items are
    dropped rather than `str(...)`-stringified -- a repr'd non-string item
    is not a usable title string."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _parse_titles(response: str) -> list[str]:
    """Defensively parse the LLM's raw response into a list of proposed
    titles. Never raises -- any parse failure yields an empty list, which
    simply means Tier 1 proposes nothing (falls through to Tier 2/3)."""
    cleaned = _strip_code_fences(response or "")
    if not cleaned:
        return []
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    return _as_str_list(parsed.get("titles"))


def tier_parametric(
    llm: LLMClient,
    question: str,
    exact: ExactAcronymIndex,
    retriever: Retriever,
    pool: PoolIndex,
    *,
    k: int = 5,
) -> list[SeedCandidate]:
    """Ask `llm` (temperature 0) which paper(s) `question` is about, then
    resolve each proposed title to a REAL pool paper.

    Resolution order per proposed title:
      1. `exact.lookup_title(title)` -- exact match, all hits kept.
      2. If that's empty, `retriever.retrieve(title, k)` -- take the top-1
         hit ONLY if it passes the plausibility gate: the proposed title and
         the hit's actual title (`pool.by_id(pid).title`) must share content
         tokens with Jaccard >= `_FUZZY_MIN_TITLE_OVERLAP`. Otherwise
         discard (a retriever always returns *some* hit for any query with a
         shared token, so this gate is what stops a garbled title resolving
         to a real-but-wrong paper).
      3. If both are empty (or the fuzzy hit fails the gate), the title is
         discarded -- no fabrication, no implausible resolution.

    Never raises: `llm.complete` can raise on a blocked/safety-filtered
    response, and the response can be malformed JSON -- either degrades to
    "no titles proposed" (empty candidate list) rather than propagating.
    Dedups by paper_id (keeping the higher-scoring resolution) and sorts by
    `(-score, paper_id)` for a deterministic, most-confident-first order.
    """
    prompt = f"Question: {question}\n\nRespond with the JSON object only."

    try:
        response = llm.complete(prompt, system=_SYSTEM_PROMPT, temperature=0.0)
    except Exception:  # noqa: BLE001 -- any LLM failure degrades to [], never crashes
        return []

    titles = _parse_titles(response)
    if not titles:
        return []

    best: dict[str, SeedCandidate] = {}

    def _consider(candidate: SeedCandidate) -> None:
        current = best.get(candidate.paper_id)
        if current is None or candidate.score > current.score:
            best[candidate.paper_id] = candidate

    for title in titles:
        if not title:
            continue

        exact_hits = exact.lookup_title(title)
        if exact_hits:
            for paper_id in exact_hits:
                _consider(
                    SeedCandidate(
                        paper_id=paper_id,
                        score=_EXACT_RESOLUTION_SCORE,
                        route="parametric",
                        reason=f"LLM named title (exact match): {title!r}",
                    )
                )
            continue

        # No exact match: fall back to fuzzy resolution via the retriever.
        # Any exception here (a flaky retriever backend) is treated the
        # same as "no plausible match" -- discard, don't crash.
        try:
            fuzzy_hits = retriever.retrieve(title, k)
        except Exception:  # noqa: BLE001
            fuzzy_hits = []

        if not fuzzy_hits:
            # No real pool paper resolves this title -- discard rather than
            # fabricate a candidate for it.
            continue

        top_paper_id, _top_score = fuzzy_hits[0]
        resolved = pool.by_id(top_paper_id)
        if resolved is None:
            # Retriever returned an id not in this pool -- treat as no match
            # (defensive; shouldn't happen when retriever/pool share a pool).
            continue

        # Plausibility gate: only accept the fuzzy hit if the proposed title
        # actually resembles the resolved paper's real title. This is what
        # stops a hallucinated/garbled LLM title -- which the retriever will
        # still map to *some* top hit on an incidental shared token -- from
        # injecting a real-but-wrong seed.
        overlap = _title_token_overlap(title, resolved.title)
        if overlap < _FUZZY_MIN_TITLE_OVERLAP:
            continue

        _consider(
            SeedCandidate(
                paper_id=top_paper_id,
                score=_FUZZY_RESOLUTION_SCORE,
                route="parametric",
                reason=(
                    f"LLM named title (no exact match, resolved via fuzzy "
                    f"retrieval top hit, title overlap {overlap:.2f}): {title!r}"
                ),
            )
        )

    return sorted(best.values(), key=lambda c: (-c.score, c.paper_id))
