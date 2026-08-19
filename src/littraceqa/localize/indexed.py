"""Index-first evidence localization with a conservative LLM fallback.

The corpus indexes already contain page-scoped passages and exact table/figure
visible IDs. This module turns those records directly into ``LocatedEvidence``
for a selected paper, avoiding a whole-paper rediscovery call when the index has
decisive evidence. The hybrid wrapper falls back to the existing grounded LLM
localizer when the index is empty, cannot satisfy an explicitly numbered
object, or merely finds *some* table/figure for an unnumbered object request.
That last distinction matters: ``Table 3`` is not evidence for a question whose
answer is in an unspecified "main comparison table" just because both are
tables.
"""
from __future__ import annotations

import re

from littraceqa.localize.interfaces import DetectedObject, LocatedEvidence, ParsedPdf
from littraceqa.retrieval.interfaces import Paper

_OBJECT_REF_RE = re.compile(r"\b(table|figure|fig[.]?)\s*([0-9]+[a-z]?)?\b", re.I)
_EQUATION_RE = re.compile(r"\b(eq(?:uation)?|algorithm|formula|objective)\b", re.I)
_CITATION_RE = re.compile(
    r"\b(cite[sd]?|citation|reference|related[ -]work|bibliograph\w*)\b", re.I
)


class IndexEvidenceLocalizer:
    """Localize within one paper using page/object records from built indexes."""

    def __init__(self, passages, objects, *, passage_k: int = 3, object_k: int = 3):
        for label, value in (("passage_k", passage_k), ("object_k", object_k)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self._passages = passages
        self._objects = objects
        self._passage_k = passage_k
        self._object_k = object_k

    def locate(
        self, question: str, paper: Paper, parsed: ParsedPdf
    ) -> list[LocatedEvidence]:
        out: list[LocatedEvidence] = []
        if self._objects is not None:
            search = getattr(self._objects, "search_paper", None)
            if callable(search):
                for object_key, _score in search(
                    paper.paper_id, question, k=self._object_k
                ):
                    record = self._objects.get_by_key(object_key)
                    if record is None or parsed.page(record.page) is None:
                        continue
                    out.append(LocatedEvidence(
                        paper_id=paper.paper_id,
                        source_type=record.source_type,
                        page=record.page,
                        object_id=record.scorer_visible_id,
                        quote=record.caption,
                        confidence=_object_confidence(question, record),
                    ))

        if self._passages is not None:
            search = getattr(self._passages, "search_paper", None)
            if callable(search):
                promoted_object = False
                for chunk_id, _score in search(
                    paper.paper_id, question, k=self._passage_k
                ):
                    record = self._passages.get(chunk_id)
                    if record is None:
                        continue
                    page = parsed.page(record.page)
                    if page is None:
                        continue
                    out.append(LocatedEvidence(
                        paper_id=paper.paper_id,
                        source_type="text_span",
                        page=record.page,
                        object_id=None,
                        quote=record.text,
                        confidence=0.72,
                    ))
                    # Caption BM25 and passage BM25 are intentionally separate,
                    # but PDF extraction often places the table body and its
                    # caption in one answer-bearing passage. When that retrieved
                    # passage EXPLICITLY names a parser-detected object on the
                    # same page, retain the stronger object locator too. This is
                    # deterministic grounding, not an inferred page->object
                    # guess: an unmentioned table/figure is never promoted.
                    named_objects = (
                        [] if promoted_object
                        else _objects_named_in_passage(record.text, page.objects)
                    )
                    for obj in named_objects[:1]:
                        out.append(LocatedEvidence(
                            paper_id=paper.paper_id,
                            source_type=obj.source_type,
                            page=record.page,
                            object_id=obj.object_id,
                            quote=record.text,
                            confidence=0.88,
                        ))
                        promoted_object = True
        return _dedup(out)


def _objects_named_in_passage(
    text: str, objects: list[DetectedObject]
) -> list[DetectedObject]:
    """Parser-detected objects whose canonical label is printed in ``text``.

    Matching is whitespace-tolerant but deliberately case-sensitive. Academic
    caption/reference labels are printed as ``Table 5``/``Figure 2``; requiring
    that capitalization avoids promoting numeric prose such as
    ``dining table 86.38`` that a permissive page parser may also detect.
    """
    collapsed = re.sub(r"\s+", " ", text or "")
    matched: list[DetectedObject] = []
    for obj in objects or []:
        # Keep regex construction outside the f-string expression: Python 3.11
        # rejects backslashes inside f-string expressions (the production VM is
        # 3.11 even when local development happens on 3.12).
        label = re.sub(r"\s+", " ", obj.object_id)
        pattern = rf"(?<!\w){re.escape(label)}(?![\w.])"
        if re.search(pattern, collapsed):
            matched.append(obj)
    return matched


def _object_confidence(question: str, record) -> float:
    for match in _OBJECT_REF_RE.finditer(question or ""):
        requested_type = "figure" if match.group(1).lower().startswith("fig") else "table"
        requested_num = match.group(2)
        if requested_type != record.source_type:
            continue
        record_num = re.search(r"([0-9]+[a-z]?)\s*$", record.scorer_visible_id, re.I)
        if requested_num is None or (
            record_num is not None
            and requested_num.casefold() == record_num.group(1).casefold()
        ):
            return 0.95
    return 0.82


def _required_modality(question: str) -> tuple[str, str | None] | None:
    match = _OBJECT_REF_RE.search(question or "")
    if match:
        source_type = (
            "figure" if match.group(1).lower().startswith("fig") else "table"
        )
        return source_type, match.group(2)
    if _EQUATION_RE.search(question or ""):
        return "equation_algorithm", None
    if _CITATION_RE.search(question or ""):
        return "citation_context", None
    return None


def _satisfies_required(
    item: LocatedEvidence, required: tuple[str, str | None]
) -> bool:
    source_type, visible_num = required
    if item.source_type != source_type:
        return False
    if visible_num is None:
        return True
    match = re.search(r"([0-9]+[a-z]?)\s*$", item.object_id or "", re.I)
    return match is not None and match.group(1).casefold() == visible_num.casefold()


def _needs_fallback(
    indexed: list[LocatedEvidence],
    required: tuple[str, str | None] | None,
) -> bool:
    """Whether indexed results are decisive enough to skip whole-paper review.

    A visible table/figure number is an exact contract and can safely
    short-circuit when present.  An unnumbered table/figure request cannot:
    object BM25 always returns *a* same-modality object when a paper contains
    one, which previously suppressed fallback even when the returned object ID
    was wrong.  Equations/citations keep the old behavior; those visible-ID
    types are not produced by the object index and therefore naturally fall
    back unless another grounded route already produced one.
    """

    if not indexed:
        return True
    if required is None:
        return False
    source_type, visible_num = required
    if source_type in {"table", "figure"} and visible_num is None:
        return True
    return not any(_satisfies_required(item, required) for item in indexed)


def _dedup(items: list[LocatedEvidence]) -> list[LocatedEvidence]:
    out: list[LocatedEvidence] = []
    seen: set[tuple] = set()
    for item in items:
        key = (item.paper_id, item.source_type, item.page, item.object_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


class HybridEvidenceLocalizer:
    """Use indexed locators first; invoke the grounded LLM only when needed."""

    def __init__(self, indexed: IndexEvidenceLocalizer, fallback):
        self._indexed = indexed
        self._fallback = fallback

    def locate(
        self, question: str, paper: Paper, parsed: ParsedPdf
    ) -> list[LocatedEvidence]:
        try:
            indexed = self._indexed.locate(question, paper, parsed)
        except Exception:  # noqa: BLE001 -- an index mismatch must degrade safely
            indexed = []
        required = _required_modality(question)
        needs_fallback = _needs_fallback(indexed, required)
        if not needs_fallback:
            return indexed
        try:
            fallback = self._fallback.locate(question, paper, parsed)
        except Exception:  # noqa: BLE001 -- total localizer contract
            fallback = []
        return _dedup([*indexed, *fallback])


class UnionEvidenceLocalizer:
    """Always preserve whole-paper candidates and add deterministic index hits.

    ``HybridEvidenceLocalizer`` is a latency-oriented short circuit: decisive
    index output can suppress the whole-paper localizer. That is useful in
    production, but it cannot answer the calibration question "does the index
    add scorer-exact candidates without deleting the current LLM floor?" This
    opt-in localizer runs both routes, keeps the fallback route first, and
    appends only distinct indexed locators. Either route may fail without
    erasing candidates produced by the other route.
    """

    def __init__(self, indexed: IndexEvidenceLocalizer, fallback):
        self._indexed = indexed
        self._fallback = fallback

    def locate(
        self, question: str, paper: Paper, parsed: ParsedPdf
    ) -> list[LocatedEvidence]:
        try:
            fallback = self._fallback.locate(question, paper, parsed)
        except Exception:  # noqa: BLE001 -- preserve the deterministic route
            fallback = []
        try:
            indexed = self._indexed.locate(question, paper, parsed)
        except Exception:  # noqa: BLE001 -- preserve the established LLM floor
            indexed = []
        return _dedup([*fallback, *indexed])
