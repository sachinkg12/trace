"""Anchor extraction: one LLM call that pulls paper-identifying signals out
of a question -- named paper titles, method acronyms, dataset names, and
whether the question spans multiple papers -- for the recognition-first
seed-finding cascade (tiers 1-3 consume the resulting `Anchor`).

Defensive by design: the LLM may refuse, wrap its JSON in a code fence, or
return something unparseable. `extract_anchor` NEVER raises on a bad
response -- it degrades to an empty `Anchor` with `raw={"error": ...}` so a
single flaky LLM call can't crash the pipeline. Every call runs at
`temperature=0.0` for reproducibility.
"""

from __future__ import annotations

import json
import re

from littraceqa.llm.interfaces import LLMClient
from littraceqa.seed.interfaces import Anchor

_SYSTEM_PROMPT = (
    "You extract paper-identifying signals from a question about an academic "
    "paper. Respond with STRICT minified JSON only -- no prose, no markdown, "
    "no code fences. The JSON object must have exactly these four keys: "
    '"named_titles" (list of strings: any paper titles explicitly named in '
    'the question), "method_acronyms" (list of strings: any method/model/'
    'system acronyms or short names mentioned, e.g. "TCM", "RAG"), '
    '"datasets" (list of strings: any dataset/benchmark names mentioned), '
    'and "asks_multiple" (boolean: true if the question compares or spans '
    "more than one paper). Use an empty list for any signal that is absent. "
    "Do not invent titles, acronyms, or datasets that are not present in "
    "the question."
)

# Matches an optional ```json / ``` opening fence and a trailing ``` closing
# fence, so a fenced response (however the model chooses to wrap it) is
# stripped down to the bare JSON body before parsing.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _empty_anchor(error: str) -> Anchor:
    return Anchor(
        named_titles=[],
        method_acronyms=[],
        datasets=[],
        asks_multiple=False,
        raw={"error": error},
    )


def _as_str_list(value: object) -> list[str]:
    """Coerce a parsed JSON value into `list[str]`, defensively.

    A well-behaved LLM returns a JSON list of strings, but a malformed one
    might return a bare string (which `list(...)` would explode into
    individual characters) or some other type. Anything that isn't a list
    is treated as absent (empty list) rather than mis-coerced. Non-string
    list ITEMS (e.g. a stray `{"a": 1}` object) are dropped rather than
    `str(...)`-stringified -- a repr'd dict/number is not a usable title/
    acronym/dataset string and would silently corrupt downstream matching.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def extract_anchor(llm: LLMClient, question: str) -> Anchor:
    """Extract an `Anchor` from `question` via one temperature-0 LLM call.

    Parses defensively: strips code fences, `json.loads`s the result, and
    on ANY failure (LLM raises, empty/blocked response, invalid JSON, JSON
    that isn't an object) returns an empty `Anchor` carrying the failure
    reason in `raw["error"]` rather than propagating an exception.
    """
    prompt = f"Question: {question}\n\nRespond with the JSON object only."

    try:
        response = llm.complete(prompt, system=_SYSTEM_PROMPT, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 -- any LLM failure degrades, never crashes
        return _empty_anchor(f"llm.complete raised: {exc!r}")

    cleaned = _strip_code_fences(response or "")
    if not cleaned:
        return _empty_anchor("empty LLM response")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return _empty_anchor(f"invalid JSON: {exc}")

    if not isinstance(parsed, dict):
        return _empty_anchor(f"expected a JSON object, got {type(parsed).__name__}")

    return Anchor(
        named_titles=_as_str_list(parsed.get("named_titles")),
        method_acronyms=_as_str_list(parsed.get("method_acronyms")),
        datasets=_as_str_list(parsed.get("datasets")),
        # Strict boolean: accept ONLY an actual JSON `true` (mirrors
        # confirm.py's is_match parsing). `bool(...)` would coerce a truthy
        # non-bool -- notably the STRING "false" -- to True.
        asks_multiple=parsed.get("asks_multiple") is True,
        raw=parsed,
    )
