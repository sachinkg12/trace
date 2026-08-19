"""Paper-specific query planning for evidence localization.

The benchmark contains cohort questions where the original question names
several methods, datasets, or comparison clauses, but localization runs one
candidate paper at a time. Passing the complete question unchanged can make the
localizer chase the most salient name even when the current PDF belongs to a
different requested row. This opt-in decorator first rewrites the question for
the current paper, then delegates locator and quote grounding to the ordinary
evidence localizer.

The planner is deliberately not trusted with evidence. It cannot emit pages,
object IDs, or quotes, and malformed replies fall back to the original question.
Only a high-confidence ``unrelated`` decision may skip a paper.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from littraceqa.llm.interfaces import LLMClient
from littraceqa.localize.interfaces import EvidenceLocalizer, LocatedEvidence, ParsedPdf
from littraceqa.retrieval.interfaces import Paper


DEFAULT_ABSTRACT_CHARS = 3000
DEFAULT_PDF_PREVIEW_CHARS = 3500
DEFAULT_MAX_QUERY_CHARS = 1800
DEFAULT_UNRELATED_CONFIDENCE = 0.9

_RELATIONSHIPS = {"direct", "cohort", "unrelated"}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You rewrite a scientific evidence question for ONE candidate paper. "
    "Do not answer the question and do not invent facts. Determine how this "
    "paper relates to the requested evidence, then produce a paper-specific "
    "search request. Respond with STRICT minified JSON only, with exactly: "
    '{"relationship":"direct|cohort|unrelated","evidence_request":"...",'
    '"confidence":0.0}. '
    "Use relationship=direct when this paper is an explicitly requested "
    "method/paper or contains the requested object. Use relationship=cohort "
    "when this paper is another requested comparison member and should supply "
    "the analogous benchmark, metric, dataset, training, citation, equation, "
    "or method-property evidence. Use unrelated only when the title, abstract, "
    "and PDF preview provide no plausible connection. Preserve every row, "
    "dataset, metric, value field, object label, equation/reference identifier, "
    "and comparison requested from this paper. For a table with several rows "
    "in one paper, request all of them. The evidence_request must name the "
    "candidate paper and be usable to locate exact evidence inside its PDF."
)


@dataclass(frozen=True)
class TargetedQuery:
    relationship: str
    evidence_request: str
    confidence: float


def _bounded_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit].rstrip()


def _pdf_preview(parsed: ParsedPdf, limit: int) -> str:
    """Use the paper front matter only to resolve title/acronym identity."""
    parts: list[str] = []
    remaining = limit
    for page in parsed.pages[:2]:
        if remaining <= 0:
            break
        text = page.text[:remaining].strip()
        if text:
            parts.append(f"--- PAGE {page.page} PREVIEW ---\n{text}")
            remaining -= len(text)
    return "\n".join(parts)


def _build_planner_prompt(question: str, paper: Paper, parsed: ParsedPdf) -> str:
    abstract = _bounded_text(paper.abstract, DEFAULT_ABSTRACT_CHARS)
    preview = _pdf_preview(parsed, DEFAULT_PDF_PREVIEW_CHARS)
    return "\n".join(
        [
            f"Original question: {question}",
            f"Candidate paper ID: {paper.paper_id}",
            f"Candidate paper title: {paper.title}",
            f"Candidate abstract: {abstract or '[missing]'}",
            preview or "[PDF preview missing]",
            "Return the JSON object only.",
        ]
    )


def _parse_targeted_query(response: str) -> TargetedQuery | None:
    cleaned = _FENCE_RE.sub("", (response or "").strip()).strip()
    if not cleaned:
        return None
    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    relationship = value.get("relationship")
    request = value.get("evidence_request")
    if relationship not in _RELATIONSHIPS:
        return None
    if not isinstance(request, str) or not request.strip():
        return None
    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return TargetedQuery(
        relationship=relationship,
        evidence_request=request.strip(),
        confidence=confidence,
    )


class TargetAwareEvidenceLocalizer:
    """Rewrite a broad question for one paper before grounded localization."""

    def __init__(
        self,
        base: EvidenceLocalizer,
        llm: LLMClient,
        *,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        unrelated_confidence: float = DEFAULT_UNRELATED_CONFIDENCE,
    ):
        if (
            isinstance(max_query_chars, bool)
            or not isinstance(max_query_chars, int)
            or max_query_chars < 1
        ):
            raise ValueError("max_query_chars must be a positive integer")
        if (
            isinstance(unrelated_confidence, bool)
            or not isinstance(unrelated_confidence, (int, float))
            or not 0.0 <= float(unrelated_confidence) <= 1.0
        ):
            raise ValueError("unrelated_confidence must be between 0 and 1")
        self._base = base
        self._llm = llm
        self._max_query_chars = max_query_chars
        self._unrelated_confidence = float(unrelated_confidence)

    def locate(
        self, question: str, paper: Paper, parsed: ParsedPdf
    ) -> list[LocatedEvidence]:
        prompt = _build_planner_prompt(question, paper, parsed)
        try:
            response = self._llm.complete(
                prompt,
                system=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=700,
            )
        except Exception:  # noqa: BLE001 -- planner failure preserves the floor
            response = ""
        planned = _parse_targeted_query(response)
        if planned is None:
            return self._base.locate(question, paper, parsed)
        if (
            planned.relationship == "unrelated"
            and planned.confidence >= self._unrelated_confidence
        ):
            return []

        request = planned.evidence_request[: self._max_query_chars].rstrip()
        targeted_question = (
            f"Original question: {question}\n\n"
            f"Candidate-paper relationship: {planned.relationship}\n"
            f"Candidate-specific evidence request: {request}\n\n"
            "Locate only evidence that this candidate paper itself verifies."
        )
        return self._base.locate(targeted_question, paper, parsed)

