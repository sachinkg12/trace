"""Whole-paper evidence localization: one LLM call that reads the entire
parsed paper and points at the exact page + source_type + object_id + a
VERBATIM quote that supports an answer to the question.

NO-FABRICATION is the whole point of this module. The LLM is asked nicely to
cite only what it sees, but it cannot be trusted -- every candidate item it
returns is independently re-checked against the parser's own ground truth
(`ParsedPdf`) before being emitted: the page must exist, the source_type must
be one of the five known types, table/figure citations must resolve to an
object the PARSER (not the LLM) detected on that exact page, and the quote
must be a real (whitespace-collapsed) substring of that page's text. Anything
that fails any check is dropped, never raised -- a single fabricated or
malformed item degrades that item, not the whole response.
"""

from __future__ import annotations

import json
import re
import unicodedata

from littraceqa.llm.interfaces import LLMClient
from littraceqa.localize.interfaces import (
    DetectedObject,
    LocatedEvidence,
    ParsedPdf,
    register_localizer,
)
from littraceqa.retrieval.interfaces import Paper

_KNOWN_TYPES = {"text_span", "table", "figure", "equation_algorithm", "citation_context"}
_OBJECT_TYPES = {"table", "figure"}
_VISIBLE_ID_TYPES = {"equation_algorithm", "citation_context"}

# Four hidden-test forensic misses included two answer-bearing sentences just
# beyond the old 4,000-character page prefix.  Keep this explicit (and
# configurable at the composition root) so a dense first column cannot make
# the second half of a page invisible to the localizer.
DEFAULT_MAX_CHARS_PER_PAGE = 8000

_SYSTEM_PROMPT = (
    "You are given a question and the FULL TEXT of a candidate academic "
    "paper, broken into pages. Find the evidence in the paper that best "
    "supports answering the question, and report WHERE it is.\n\n"
    "For a multi-paper question, this candidate paper may support only one "
    "clause or requested item. Return evidence for every clause this paper "
    "supports; do not require this paper to answer the entire question.\n\n"
    "When the question explicitly asks what a table or figure depicts, shows, "
    "or reports, emit that table/figure object as evidence rather than only "
    "nearby prose. You may also emit a nearby text_span when it independently "
    "states the answer.\n\n"
    "Respond with STRICT minified JSON only -- no prose, no markdown, no "
    "code fences. The response MUST be a JSON ARRAY (not an object) of "
    "evidence items, each with exactly these keys:\n"
    '  "page": integer, the 1-indexed page number the evidence appears on.\n'
    '  "source_type": one of "text_span", "table", "figure", '
    '"equation_algorithm", "citation_context".\n'
    '  "object_id": for table/figure, the exact detected label listed for '
    'that page (e.g. "Table 4"); for equation_algorithm or citation_context, '
    'the exact visible identifier (e.g. "Equation 6", "Algorithm 2", "24") '
    'when printed on that page; otherwise null.\n'
    '  "quote": a short quote copied VERBATIM from the page you cite -- '
    "character-for-character, no paraphrasing, no ellipses, no combining "
    "text across pages. Keep the quote SHORT (a few words or a single "
    "value). For a table or figure, quote a single cell value or one short "
    "contiguous phrase -- do NOT stitch together text from multiple cells "
    "or rows.\n"
    '  "confidence": a number between 0 and 1.\n\n'
    "Only cite object_ids that are explicitly listed as detected objects on "
    "the page you name -- never invent a table/figure number. Only quote "
    "text that literally appears on the page you cite -- never paraphrase "
    "or fabricate a quote. If you find no supporting evidence, return an "
    "empty JSON array []."
)

_COHORT_COVERAGE_PROMPT = (
    "\n\nCOHORT COVERAGE MODE: The candidate paper was selected as a member "
    "of the question's comparison cohort. If it does not contain a fact for a "
    "specifically named target, still return its closest directly comparable "
    "evidence for the same benchmark, metric, dataset, citation, or method "
    "property. Do not return an unrelated object merely to avoid an empty "
    "answer, and keep every locator and quote subject to the same grounding "
    "rules."
)

# Matches an optional ```json / ``` opening fence and a trailing ``` closing
# fence, so a fenced response (however the model chooses to wrap it) is
# stripped down to the bare JSON body before parsing.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _collapse_ws(s: str) -> str:
    """Collapse any run of whitespace to a single space and strip ends, so
    quote/text comparisons are insensitive to newlines, tabs, and PDF
    extraction's irregular spacing."""
    return re.sub(r"\s+", " ", s).strip()


# Content tokens for the coverage fallback: alphabetic words of >=3 chars
# (skips glue words like "a"/"of"/"the" that carry no attestation weight)
# and numbers/decimals (a table's actual values -- the load-bearing part of
# a quantitative quote).
_TOKEN_RE = re.compile(r"[a-z]{3,}|\d[\d.]*")


def _normalize_for_match(s: str) -> str:
    """Normalize page/quote text so the grounding check survives real-PDF
    extraction artifacts without weakening no-fabrication: NFKC folds
    ligatures (fi -> fi) and width variants, line-wrap hyphenation is
    joined (`imple-\\nment` -> `implement`), soft hyphens are dropped, and
    whitespace is collapsed. De-hyphenate BEFORE collapsing whitespace --
    the hyphenation join keys on the `\\n` that a whitespace-collapse would
    otherwise erase."""
    s = unicodedata.normalize("NFKC", s)
    # Remove line-wrap hyphenation: "imple-\nment" -> "implement".
    s = re.sub(r"-\s*\n\s*", "", s)
    s = s.replace("­", "")          # soft hyphen
    s = re.sub(r"\s+", " ", s)           # collapse whitespace
    return s.strip().casefold()


def _page_text_has_quote(page_text: str, quote: str) -> bool:
    """Is `quote` really grounded in `page_text`? Attestation check that
    keeps the localizer from emitting a quote the LLM paraphrased or
    hallucinated -- but robust to the two extraction artifacts that
    otherwise drop CORRECT evidence: line-wrap hyphenation and
    column-major table reordering.

    First a normalized exact-substring test (handles hyphenation,
    ligatures, whitespace). Failing that, a content-token-coverage
    fallback for reordered spans (column-major table cells extracted out of
    order): require >=75% of the quote's content tokens (words >=3 chars or
    numbers) to be present on THIS page. Still grounds against the cited
    page, so a hallucinated quote whose tokens aren't on the page won't
    pass."""
    npage = _normalize_for_match(page_text)
    nquote = _normalize_for_match(quote)
    if not nquote:
        return False
    if nquote in npage:                  # robust exact substring
        return True
    # Fallback: reordered spans (column-major table cells). Require most of
    # the quote's CONTENT tokens to be present on the cited page.
    qtokens = _TOKEN_RE.findall(nquote)
    if not qtokens:
        return False
    ptokens = set(_TOKEN_RE.findall(npage))
    present = sum(1 for t in qtokens if t in ptokens)
    return present / len(qtokens) >= 0.75


def _coerce_confidence(value: object) -> float:
    """Coerce a parsed JSON value into a float clamped to [0, 1], defaulting
    to 0.5 (neutral) on any malformed/missing/non-numeric value -- unlike a
    hard 0.0 default, this doesn't quietly bury an otherwise-valid, verified
    item at the bottom of any downstream confidence-sorted ranking."""
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if conf != conf:  # NaN check without importing math
        return 0.5
    return max(0.0, min(1.0, conf))


def _object_ids_on_page(objects: list[DetectedObject]) -> set[str]:
    return {_collapse_ws(obj.object_id).lower() for obj in objects}


def _visible_id_on_page(object_id: str, page_text: str, source_type: str) -> bool:
    """Ground an equation/citation ID against page text.

    PDF extraction commonly renders ``Equation 6`` as ``(6)`` and citations
    as ``[24]``.  Require either those delimiters or an equation/reference
    label: a bare number in ordinary prose is not a grounded visible ID.
    """
    normalized_page = _normalize_for_match(page_text)
    normalized_id = _normalize_for_match(object_id)
    match = re.search(r"(\d+[a-z]?)\s*$", normalized_id)
    if match is None:
        return False
    visible = re.escape(match.group(1))
    if source_type == "citation_context":
        labelled = (
            rf"(?:cit(?:ation)?|ref(?:erence)?)[.]?\s*[\[(]?\s*"
            rf"{visible}(?![0-9a-z])\s*(?:\]|\))?"
        )
        delimited = rf"\[\s*{visible}\s*\]"
    else:
        labelled = (
            rf"(?:eq(?:uation)?|alg(?:orithm)?)[.]?\s*[\[(]?\s*"
            rf"{visible}(?![0-9a-z])\s*(?:\]|\))?"
        )
        delimited = rf"\(\s*{visible}\s*\)"
    pattern = rf"(?:{labelled})|(?:{delimited})"
    return re.search(pattern, normalized_page, re.IGNORECASE) is not None


def _build_prompt(question: str, paper: Paper, parsed: ParsedPdf, max_chars_per_page: int) -> str:
    """Render the question plus the whole paper as `--- PAGE {n} ---` blocks
    (each page's text truncated to `max_chars_per_page`, plus a listing of
    that page's detected object labels) for the single whole-paper LLM call.
    Isolated in its own helper so page-truncation is unit-testable and the
    prompt shape can change without touching `locate`'s control flow."""
    parts = [
        f"Question: {question}",
        f"Paper title: {paper.title}",
        "",
        "--- PAPER TEXT ---",
    ]
    for page_unit in parsed.pages:
        text = page_unit.text[:max_chars_per_page]
        parts.append(f"--- PAGE {page_unit.page} ---")
        if page_unit.objects:
            labels = ", ".join(obj.object_id for obj in page_unit.objects)
            parts.append(f"[detected objects on this page: {labels}]")
        parts.append(text)
    parts.append("")
    parts.append("Respond with the JSON array only.")
    return "\n".join(parts)


@register_localizer("llm")
class LlmEvidenceLocalizer:
    """`EvidenceLocalizer` backed by one whole-paper LLM call, with every
    emitted item independently re-validated against the parser's own
    ground truth (never trusting the LLM's claims about page/object/quote)."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_evidence: int = 5,
        max_chars_per_page: int = DEFAULT_MAX_CHARS_PER_PAGE,
        cohort_coverage: bool = False,
    ):
        if (
            isinstance(max_chars_per_page, bool)
            or not isinstance(max_chars_per_page, int)
            or max_chars_per_page <= 0
        ):
            raise ValueError("max_chars_per_page must be a positive integer")
        self._llm = llm
        self._max_evidence = max_evidence
        self._max_chars_per_page = max_chars_per_page
        self._system_prompt = (
            _SYSTEM_PROMPT + _COHORT_COVERAGE_PROMPT
            if cohort_coverage
            else _SYSTEM_PROMPT
        )

    def locate(self, question: str, paper: Paper, parsed: ParsedPdf) -> list[LocatedEvidence]:
        prompt = _build_prompt(question, paper, parsed, self._max_chars_per_page)

        try:
            response = self._llm.complete(
                prompt, system=self._system_prompt, temperature=0.0
            )
        except Exception:  # noqa: BLE001 -- any LLM failure degrades, never crashes
            return []

        items = self._parse_items(response)

        out: list[LocatedEvidence] = []
        for item in items:
            if len(out) >= self._max_evidence:
                break
            evidence = self._validate_item(item, paper, parsed)
            if evidence is not None:
                out.append(evidence)
        return out

    @staticmethod
    def _parse_items(response: str) -> list[dict]:
        """Defensive parse mirroring `seed/anchor.py`'s tolerant approach:
        strip code fences, `json.loads`, and on ANY failure (empty response,
        invalid JSON, JSON that isn't an array, or an array of non-dicts)
        degrade to an empty list rather than raising."""
        cleaned = _strip_code_fences(response or "")
        if not cleaned:
            return []
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    def _validate_item(self, item: dict, paper: Paper, parsed: ParsedPdf) -> LocatedEvidence | None:
        page = item.get("page")
        if not isinstance(page, int) or isinstance(page, bool):
            return None
        page_unit = parsed.page(page)
        if page_unit is None:
            return None

        source_type = item.get("source_type")
        if source_type not in _KNOWN_TYPES:
            return None

        object_id = item.get("object_id")
        if source_type in _OBJECT_TYPES:
            if not isinstance(object_id, str) or not object_id.strip():
                return None
            if _collapse_ws(object_id).lower() not in _object_ids_on_page(page_unit.objects):
                return None
        elif source_type in _VISIBLE_ID_TYPES:
            # PyMuPDF object detection currently indexes table/figure captions,
            # not equations/references.  Retain a visible ID only when its text
            # actually occurs on the cited page; otherwise discard the ID while
            # keeping the independently quote-attested page evidence.
            if isinstance(object_id, int) and not isinstance(object_id, bool):
                object_id = str(object_id)
            if (
                not isinstance(object_id, str)
                or not object_id.strip()
                or not _visible_id_on_page(object_id, page_unit.text, source_type)
            ):
                object_id = None
        else:
            object_id = None

        quote = item.get("quote")
        if not isinstance(quote, str) or not _page_text_has_quote(page_unit.text, quote):
            return None

        confidence = _coerce_confidence(item.get("confidence"))

        return LocatedEvidence(
            paper_id=paper.paper_id,
            source_type=source_type,
            page=page,
            object_id=object_id,
            quote=quote,
            confidence=confidence,
        )
