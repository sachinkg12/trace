"""Abstract-level confirmation: given a question and a CANDIDATE paper
(title + abstract), ask the LLM whether this specific paper is the one the
question targets.

This is the stand-in for #5's full-PDF read-to-confirm step -- cheaper (one
title+abstract, not a whole paper) but the same shape of judgment: "is THIS
the right paper?" rather than "which paper is it?" (that's Tier 1's job, see
`tier_parametric`). Used to sanity-check candidates proposed by any tier
(parametric, exact, or dense) before they're trusted.

CAVEAT -- ABSTRACT-LEVEL ONLY, FALSE NEGATIVES EXPECTED: `confirm_candidate`
only ever sees a title + abstract, never the paper's body. A question whose
answer lives in a deep table, figure, or section that the abstract doesn't
reflect can legitimately get `is_match=False` on the CORRECT paper -- that
is a false negative of this function, not evidence the paper is wrong.
Downstream (#5 evidence localization) MUST treat the seed ranking as a
PRIOR and must NOT discard a lower-ranked (or confirm-penalized) seed purely
on an abstract-level `is_match=False`; the real confirmation is #5's
full-PDF read.
"""

from __future__ import annotations

import json
import re

from littraceqa.llm.interfaces import LLMClient
from littraceqa.retrieval.interfaces import Paper

_SYSTEM_PROMPT = (
    "You are given a question and a candidate paper's title and abstract. "
    "Decide whether THIS paper is the one the question is asking about -- "
    "i.e. whether the question's subject matter (the method, dataset, "
    "result, or claim it asks about) plausibly belongs to this paper based "
    "on its title and abstract. Respond with STRICT minified JSON only -- "
    "no prose, no markdown, no code fences. The JSON object must have "
    'exactly these three keys: "is_match" (boolean), "confidence" (number '
    "between 0 and 1), and \"reason\" (a brief string explaining the "
    "decision). Do not fabricate details not present in the title/abstract."
)

# Matches an optional ```json / ``` opening fence and a trailing ``` closing
# fence, so a fenced response is stripped down to the bare JSON body first.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _coerce_confidence(value: object) -> float:
    """Coerce a parsed JSON value into a float clamped to [0, 1].

    Defensive: a malformed/missing/non-numeric confidence degrades to 0.0
    rather than raising or propagating an out-of-range value.
    """
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if conf != conf:  # NaN check without importing math
        return 0.0
    return max(0.0, min(1.0, conf))


def confirm_candidate(
    llm: LLMClient, question: str, paper: Paper
) -> tuple[bool, float, str]:
    """Ask `llm` (temperature 0) whether `paper` is the one `question`
    targets, given only its title + abstract.

    Returns `(is_match, confidence, reason)`. Defensive parse: never raises
    -- `llm.complete` can raise on a blocked/safety-filtered response, and
    the response can be malformed JSON; either degrades to
    `(False, 0.0, "<failure reason>")` rather than propagating.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Candidate paper title: {paper.title}\n"
        f"Candidate paper abstract: {paper.abstract}\n\n"
        "Respond with the JSON object only."
    )

    try:
        response = llm.complete(prompt, system=_SYSTEM_PROMPT, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 -- any LLM failure degrades, never crashes
        return (False, 0.0, f"llm.complete raised: {exc!r}")

    cleaned = _strip_code_fences(response or "")
    if not cleaned:
        return (False, 0.0, "empty LLM response")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return (False, 0.0, f"invalid JSON: {exc}")

    if not isinstance(parsed, dict):
        return (False, 0.0, f"expected a JSON object, got {type(parsed).__name__}")

    # Strict boolean: accept ONLY an actual JSON `true`. `bool(...)` would
    # coerce a truthy non-bool -- notably the STRING "false" -- to True, so
    # any non-bool value is treated as not-a-match instead.
    is_match = parsed.get("is_match") is True
    confidence = _coerce_confidence(parsed.get("confidence"))
    reason = str(parsed.get("reason", ""))

    return (is_match, confidence, reason)
