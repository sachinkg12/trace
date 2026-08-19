"""Extractive-verbatim freeform answering: one LLM call copies the answer
value verbatim out of the cited evidence -- the scorer matches
character-for-character, so the value is NEVER recomputed or reformatted,
only copied. Defensive by design, mirroring `seed/anchor.py`: the LLM may
refuse, wrap its JSON in a code fence, or return something unparseable;
`FreeformAnswerer` never raises on a bad response, it degrades to an empty
string. Always-answer: even an ungrounded value is still emitted, just at
lower confidence, so a real (if unattested) guess is never withheld.

Reference/citation-list questions ("first author of the 24th reference",
"index of the last reference", "how many references include X as an
author") are answered DETERMINISTICALLY instead: an LLM can't reliably
count or index a long numbered bibliography, but the reference list is
parseable text, so `references.parse_reference_list` + the ordinal/count
helpers answer these in code, and the LLM call is skipped entirely. If the
paper's reference list doesn't parse (or the deterministic helper can't
answer), this falls back to the normal LLM path below -- the reference
route only ever adds a path, never removes the existing one.
"""

from __future__ import annotations

import json
import re

from littraceqa.answer.grounding import ground_value
from littraceqa.answer.interfaces import AnswerContext, StrategyOutput, register_strategy
from littraceqa.answer.references import (
    answer_reference_question, is_reference_list_question, parse_reference_list,
)
from littraceqa.answer.vision import _trim_answer, first_visual_evidence, vision_answer_text

# Confidence for a figure/table-read answer. The value often won't appear in
# the parsed TEXT (it lives in the figure's pixels, or the table text is
# column-scrambled), so `ground_value` can't reliably attest it; instead the
# answer grounds on the page it was read from. Kept above the pipeline's 0.5
# evidence floor so that the visual evidence is retained.
_VISION_CONFIDENCE = 0.7

_SYSTEM_PROMPT = (
    "You answer questions about academic papers using ONLY the evidence "
    "quotes provided. Copy the answer value VERBATIM from the evidence -- "
    "do not reformat, round, rephrase, or recompute it. Respond with STRICT "
    "minified JSON only -- no prose, no markdown, no code fences. The JSON "
    'object must have exactly one key: "answer" (string: the verbatim '
    "answer value copied from the evidence)."
)

# Matches an optional ```json / ``` opening fence and a trailing ``` closing
# fence, so a fenced response (however the model chooses to wrap it) is
# stripped down to the bare JSON body before parsing.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _build_prompt(ctx: AnswerContext) -> str:
    lines = [f"Question: {ctx.question}", "", "Evidence:"]
    for ev in ctx.evidence:
        where = f"{ev.paper_id} p.{ev.page}"
        if ev.object_id:
            where += f" ({ev.object_id})"
        lines.append(f"- [{where}] {ev.quote}")
    lines.append("")
    lines.append("Respond with the JSON object only.")
    return "\n".join(lines)


@register_strategy("freeform")
class FreeformAnswerer:
    answer_type = "freeform"

    def answer(self, ctx: AnswerContext) -> StrategyOutput:
        deterministic = self._answer_reference_question(ctx)
        if deterministic is not None:
            attested = ground_value(deterministic, ctx.evidence, ctx.parsed_by_id)
            confidence = 1.0 if attested else 0.3
            return StrategyOutput(value=deterministic, confidence=confidence,
                                  attested_evidence=attested)

        # Vision path: when a FIGURE or TABLE is the cited evidence, the answer
        # lives in the page's pixels (absent from, or scrambled in, the parsed
        # text), so read the rendered page image directly. Additive and
        # defensive -- on no cached PDF / render / LLM failure or a
        # NOT_IN_FIGURE reply this returns None and we fall through to text.
        vision = self._answer_from_visual(ctx)
        if vision is not None:
            return vision

        prompt = _build_prompt(ctx)

        try:
            response = ctx.llm.complete(prompt, system=_SYSTEM_PROMPT, temperature=0.0)
        except Exception:  # noqa: BLE001 -- any LLM failure degrades, never crashes
            response = ""

        value = _trim_answer(self._parse_value(response))
        attested = ground_value(value, ctx.evidence, ctx.parsed_by_id)
        confidence = 1.0 if attested else 0.3
        return StrategyOutput(value=value, confidence=confidence, attested_evidence=attested)

    @staticmethod
    def _answer_from_visual(ctx: AnswerContext) -> StrategyOutput | None:
        """Read a cited figure/table page and extract the answer value.
        Returns None (fall through to the text path) when there is no visual
        evidence, no cached PDF, or the model reads no usable value -- never
        raises."""
        ev = first_visual_evidence(ctx.evidence)
        if ev is None:
            return None
        try:
            value = vision_answer_text(ctx, ev)
        except Exception:  # noqa: BLE001 -- vision is best-effort; degrade to text path
            return None
        if not value:
            return None
        # Ground on the page it was read from (no fabricated evidence).
        return StrategyOutput(value=value, confidence=_VISION_CONFIDENCE,
                              attested_evidence=[ev])

    @staticmethod
    def _answer_reference_question(ctx: AnswerContext) -> str | None:
        """Deterministic reference-list route (see module docstring). Never
        raises: any failure here just means "no deterministic answer",
        falling through to the LLM path."""
        try:
            if not is_reference_list_question(ctx.question) or not ctx.paper_ids:
                return None
            parsed = ctx.parsed_by_id.get(ctx.paper_ids[0])
            refs = parse_reference_list(parsed)
            if not refs:
                return None
            return answer_reference_question(ctx.question, refs)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _parse_value(response: str) -> str:
        cleaned = _strip_code_fences(response or "")
        if not cleaned:
            return ""
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return ""
        if not isinstance(parsed, dict):
            return ""
        value = parsed.get("answer")
        # A JSON string is copied verbatim. A JSON INTEGER is coerced via
        # str() (so gold "5" matches {"answer": 5}) -- but NOT a bool (a JSON
        # true/false is not an answer value; `bool` is an `int` subclass so it
        # must be excluded first) and NOT a float: str(14.70) -> "14.7" would
        # drop the trailing zero the scorer matches byte-for-byte.
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        return ""
