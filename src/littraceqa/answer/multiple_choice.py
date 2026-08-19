"""Multiple-choice answering: value-match against a grounded freeform
extraction first, LLM-pick fallback second, and a never-blank final
fallback last. A blank MC answer scores identically to a wrong one under
the shared-task scorer, and the constant-B baseline already scores ~0.41,
so a real (if low-confidence) guess always strictly dominates abstaining.

Cascade, mirroring the recognition-first strategy used elsewhere in the
pipeline (seed/anchor.py, answer/freeform.py):
  1. Value-match: reuse `FreeformAnswerer` to extract+ground the verbatim
     answer value, then compare it (normalized) against each option's
     value. A UNIQUE match wins -- ambiguous (0 or >1) matches fall
     through, since a non-unique match isn't trustworthy.
  2. LLM-pick: ask the LLM directly for the option letter; only accepted
     if it names one of the real option keys.
  3. Final fallback: the first option key (or "A" if there are no
     options at all) -- never blank.
"""

from __future__ import annotations

import json
import re

from littraceqa.answer.freeform import FreeformAnswerer
from littraceqa.answer.grounding import _normalize, ground_value
from littraceqa.answer.interfaces import AnswerContext, StrategyOutput, register_strategy

_SYSTEM_PROMPT = (
    "You answer multiple-choice questions about academic papers using ONLY "
    "the evidence quotes provided. Respond with STRICT minified JSON only "
    "-- no prose, no markdown, no code fences. The JSON object must have "
    'exactly one key: "letter" (string: the single letter of the correct '
    "option, e.g. \"A\")."
)

# Matches an optional ```json / ``` opening fence and a trailing ``` closing
# fence, so a fenced response (however the model chooses to wrap it) is
# stripped down to the bare JSON body before parsing.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_SOURCE_CONTEXT_RADIUS = 550
_SOURCE_CONTEXT_MAX_CHARS = 1200


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _source_context(ctx: AnswerContext, ev) -> str:
    """Return a bounded cited-page window around the localized fact.

    Localizers often emit the exact scalar (for example ``"73.0"``) as the
    quote.  A bare scalar is enough for evidence attestation but not enough to
    distinguish table rows or columns.  Give the MC reasoner the surrounding
    *cited source text* so it can bind that value to its method, metric, and
    setting.  This never searches another page or paper and fails closed when
    the cited page/anchor is unavailable.
    """
    parsed = ctx.parsed_by_id.get(ev.paper_id)
    page = parsed.page(ev.page) if parsed else None
    if page is None or not page.text:
        return ""

    page_text = page.text
    anchors = [
        value for value in (ev.quote, ev.object_id)
        if isinstance(value, str) and value.strip()
    ]
    position = -1
    matched_anchor = ""
    for anchor in anchors:
        position = page_text.casefold().find(anchor.strip().casefold())
        if position >= 0:
            matched_anchor = anchor
            break
    if position < 0:
        return ""

    start = max(0, position - _SOURCE_CONTEXT_RADIUS)
    end = min(
        len(page_text),
        position + len(matched_anchor) + _SOURCE_CONTEXT_RADIUS,
    )
    compact = re.sub(r"\s+", " ", page_text[start:end]).strip()
    if len(compact) > _SOURCE_CONTEXT_MAX_CHARS:
        compact = compact[:_SOURCE_CONTEXT_MAX_CHARS].rstrip()
    return compact


def _build_prompt(ctx: AnswerContext) -> str:
    lines = [f"Question: {ctx.question}", "", "Options:"]
    for letter, value in (ctx.mc_options or {}).items():
        lines.append(f"{letter}. {value}")
    lines.append("")
    lines.append("Evidence:")
    for ev in ctx.evidence:
        where = f"{ev.paper_id} p.{ev.page}"
        if ev.object_id:
            where += f" ({ev.object_id})"
        lines.append(f"- [{where}] {ev.quote}")
        context = _source_context(ctx, ev)
        if context and _normalize(context) != _normalize(ev.quote or ""):
            lines.append(f"  Cited source context: {context}")
    lines.append("")
    lines.append("Respond with the JSON object only.")
    return "\n".join(lines)


@register_strategy("multiple_choice")
class MultipleChoiceAnswerer:
    answer_type = "multiple_choice"

    def answer(self, ctx: AnswerContext) -> StrategyOutput:
        options = ctx.mc_options
        if options:
            # 1. Value-match. FreeformAnswerer itself reads a cited figure/table
            #    via the vision path, so a page-derived value (e.g. "DKO", "1")
            #    that equals an option is matched here and mapped to its letter,
            #    grounded on the visual evidence -- no separate image call.
            value_match = self._value_match(ctx, options)
            if value_match is not None:
                return value_match

            llm_pick = self._llm_pick(ctx, options)
            if llm_pick is not None:
                return llm_pick

        return self._final_fallback(options)

    @staticmethod
    def _value_match(ctx: AnswerContext, options: dict[str, str]) -> StrategyOutput | None:
        """Extract a verbatim value via `FreeformAnswerer` and match it
        against the option values. Only a UNIQUE match is trusted -- an
        empty or ambiguous match falls through to the LLM-pick fallback."""
        extracted = FreeformAnswerer().answer(ctx)
        nval = _normalize(extracted.value or "")
        if not nval:
            return None

        matches = [letter for letter, opt_value in options.items()
                   if _normalize(opt_value) == nval]
        if len(matches) != 1:
            return None

        letter = matches[0]
        attested = extracted.attested_evidence
        confidence = 1.0 if attested else 0.6
        return StrategyOutput(value=letter, confidence=confidence, attested_evidence=attested)

    @staticmethod
    def _llm_pick(ctx: AnswerContext, options: dict[str, str]) -> StrategyOutput | None:
        """Ask the LLM to pick a letter directly. Defensive by design,
        mirroring `seed/anchor.py`/`answer/freeform.py`: any LLM failure or
        unparseable/unrecognized response degrades to `None` (caller falls
        through to the final fallback) rather than raising or guessing."""
        prompt = _build_prompt(ctx)
        try:
            response = ctx.llm.complete(prompt, system=_SYSTEM_PROMPT, temperature=0.0)
        except Exception:  # noqa: BLE001 -- any LLM failure degrades, never crashes
            response = ""

        cleaned = _strip_code_fences(response or "")
        if not cleaned:
            return None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None

        letter_raw = parsed.get("letter")
        if not isinstance(letter_raw, str):
            return None
        letter = letter_raw.strip().upper()
        if letter not in options:
            return None

        attested = ground_value(options[letter], ctx.evidence, ctx.parsed_by_id)
        return StrategyOutput(value=letter, confidence=0.5, attested_evidence=attested)

    @staticmethod
    def _final_fallback(options: dict[str, str] | None) -> StrategyOutput:
        """Never blank: the first option key if options exist, else "A"."""
        letter = next(iter(options)) if options else "A"
        return StrategyOutput(value=letter, confidence=0.1, attested_evidence=[])
