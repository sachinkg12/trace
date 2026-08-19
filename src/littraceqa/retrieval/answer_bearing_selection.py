"""Conservative evidence-first rescue on top of target-coverage selection.

The retrieval-depth gate proved that some correct papers are already present but
buried because property-first ranks a generic high-BM25 passage above the one
passage that contains the requested table row.  This selector does not replace
the established target-coverage ordering.  It starts from that exact floor and
promotes a candidate only for a narrow, high-confidence case:

* the question requests one paper;
* the structured plan carries at least two concrete constraints; and
* one candidate's real passage/object evidence matches at least half of those
  constraints and at least two more constraints than the current top paper.

Alias and dense title/abstract support are excluded from rescue evidence.  They
are useful retrieval signals but cannot prove that the requested table content
appears in the paper.  The policy is pure, deterministic, and opt-in.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from littraceqa.paperset.cascade import Candidate
from littraceqa.pipeline.planner import Plan
from littraceqa.retrieval.bm25 import tokenize
from littraceqa.retrieval.target_selection import (
    TargetSelectionResult,
    select_target_coverage,
)

_EVIDENCE_SOURCES = frozenset({"passage", "citation_passage", "object_caption"})
_ALNUM_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"(?<![a-z0-9])\d+(?:\.\d+)?%?(?![a-z0-9])")


@dataclass(frozen=True)
class EvidenceCoverage:
    paper_id: str
    matched_constraints: tuple[str, ...]
    question_token_hits: int
    numeric_token_count: int
    property_rank: int | None
    candidate_position: int


@dataclass(frozen=True)
class AnswerBearingSelectionResult:
    paper_ids: list[str]
    baseline: TargetSelectionResult
    rescue_paper_id: str | None = None
    replaced_paper_id: str | None = None
    required_constraint_matches: int | None = None
    reason: str = "not_applicable"
    coverage: list[EvidenceCoverage] = field(default_factory=list)


def _normalized_surface(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(_ALNUM_RE.findall(text.casefold()))


def _evidence_text(candidate: Candidate) -> str:
    """Only answer-bearing corpus evidence; never alias/dense retrieval text."""
    parts: list[str] = []
    for support in candidate.support:
        if not isinstance(support, dict):
            continue
        if support.get("source") not in _EVIDENCE_SOURCES:
            continue
        text = support.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return " ".join(parts)


def _constraint_surfaces(plan: Plan) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for target in getattr(plan, "targets", ()):
        if getattr(target, "role", None) != "constraint":
            continue
        text = str(getattr(target, "text", "") or "").strip()
        normalized = _normalized_surface(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append((str(getattr(target, "key", "") or text), normalized))
    return out


def _property_rank(candidate: Candidate) -> int | None:
    ranks = [
        signal.rank for signal in candidate.route_signals
        if signal.route == "property" and signal.group_key is None
    ]
    return min(ranks) if ranks else None


def _coverage(
    candidate: Candidate,
    *,
    position: int,
    constraints: list[tuple[str, str]],
    question_tokens: set[str],
) -> EvidenceCoverage:
    raw_text = _evidence_text(candidate)
    normalized_text = _normalized_surface(raw_text)
    padded_text = f" {normalized_text} "
    matched = tuple(
        key for key, surface in constraints
        if f" {surface} " in padded_text
    )
    evidence_tokens = set(tokenize(raw_text))
    return EvidenceCoverage(
        paper_id=candidate.paper_id,
        matched_constraints=matched,
        question_token_hits=len(question_tokens & evidence_tokens),
        numeric_token_count=len(_NUMBER_RE.findall(raw_text.casefold())),
        property_rank=_property_rank(candidate),
        candidate_position=position,
    )


def _strength(item: EvidenceCoverage) -> tuple:
    return (
        -len(item.matched_constraints),
        -item.question_token_hits,
        -item.numeric_token_count,
        item.property_rank if item.property_rank is not None else 10**9,
        item.candidate_position,
        item.paper_id,
    )


def select_answer_bearing(
    candidates: list[Candidate],
    plan: Plan,
    *,
    question: str,
) -> AnswerBearingSelectionResult:
    """Promote one strongly constraint-attested paper over the exact baseline."""
    baseline = select_target_coverage(candidates, plan)
    constraints = _constraint_surfaces(plan)
    target_count = sum(
        getattr(target, "role", None) == "target"
        for target in getattr(plan, "targets", ())
    )
    # The planner can mistake a list of requested TABLE ROWS (datasets,
    # benchmarks, settings) for a request for multiple papers.  When there are
    # no paper targets and one candidate passage uniquely contains the row-key
    # list, that passage is stronger cardinality evidence than the planner's
    # multiplicity label.  True multi-paper plans still carry target entities
    # and remain byte-for-byte unchanged.
    constraint_only_table = (
        plan.multiplicity == "multi"
        and target_count == 0
        and len(constraints) >= 2
        and (plan.criterion or {}).get("required_source_type") == "table"
    )
    if plan.multiplicity != "single" and not constraint_only_table:
        return AnswerBearingSelectionResult(
            paper_ids=list(baseline.paper_ids), baseline=baseline,
            reason="multi_paper_unchanged",
        )

    if len(constraints) < 2 or not baseline.paper_ids:
        return AnswerBearingSelectionResult(
            paper_ids=list(baseline.paper_ids), baseline=baseline,
            reason="insufficient_constraints",
        )

    question_tokens = set(tokenize(question))
    coverage = [
        _coverage(
            candidate,
            position=position,
            constraints=constraints,
            question_tokens=question_tokens,
        )
        for position, candidate in enumerate(candidates)
    ]
    coverage.sort(key=_strength)
    if not coverage:
        return AnswerBearingSelectionResult(
            paper_ids=list(baseline.paper_ids), baseline=baseline,
            reason="no_candidates",
        )

    by_paper = {item.paper_id: item for item in coverage}
    current_id = baseline.paper_ids[0]
    current = by_paper.get(current_id)
    current_matches = len(current.matched_constraints) if current else 0
    best = coverage[0]
    best_matches = len(best.matched_constraints)
    required = max(2, math.ceil(len(constraints) / 2))
    if best_matches < required:
        return AnswerBearingSelectionResult(
            paper_ids=list(baseline.paper_ids), baseline=baseline,
            required_constraint_matches=required,
            reason="coverage_below_threshold", coverage=coverage,
        )
    if best.paper_id == current_id:
        return AnswerBearingSelectionResult(
            paper_ids=list(baseline.paper_ids), baseline=baseline,
            required_constraint_matches=required,
            reason="baseline_already_best", coverage=coverage,
        )
    if best_matches < current_matches + 2:
        return AnswerBearingSelectionResult(
            paper_ids=list(baseline.paper_ids), baseline=baseline,
            required_constraint_matches=required,
            reason="coverage_margin_too_small", coverage=coverage,
        )

    reordered = (
        [best.paper_id]
        if constraint_only_table
        else [
            best.paper_id,
            *(
                paper_id for paper_id in baseline.paper_ids
                if paper_id != best.paper_id
            ),
        ]
    )
    return AnswerBearingSelectionResult(
        paper_ids=reordered,
        baseline=baseline,
        rescue_paper_id=best.paper_id,
        replaced_paper_id=current_id,
        required_constraint_matches=required,
        reason=(
            "constraint_only_table_rescue"
            if constraint_only_table
            else "constraint_evidence_rescue"
        ),
        coverage=coverage,
    )
