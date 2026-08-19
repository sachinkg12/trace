"""Group-aware, coverage-first paper ordering.

This selector is deliberately pure and opt-in. It consumes the structured
planner targets and group-local ``RouteSignal`` values, covers each requested
target with a distinct *target-property-corroborated* paper, then fills from the
proven property-first floor. A bare alias/global-property hit may aid ranking,
but cannot displace that floor. It never changes the legacy/property-first paths.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from littraceqa.paperset.cascade import Candidate, RouteSignal
from littraceqa.pipeline.planner import Plan, PlanTarget

_RRF_K = 60
_SET_COVER_MAX_PROPERTY_RANK = 2
_SHORT_TITLE_MAX_PROPERTY_RANK = 5
_SELF_DEFINITION_MAX_PROPERTY_RANK = 2


@dataclass(frozen=True)
class TargetAssignment:
    target_key: str
    paper_id: str
    group_rank: int
    target_property_rank: int
    fusion_score: float
    corroboration: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetSelectionResult:
    paper_ids: list[str]
    assignments: list[TargetAssignment] = field(default_factory=list)
    uncovered_targets: list[str] = field(default_factory=list)
    uncorroborated_target_counts: dict[str, int] = field(default_factory=dict)
    excluded_non_target_anchors: list[str] = field(default_factory=list)
    floor_paper_ids: list[str] = field(default_factory=list)


def _same_group(signal: RouteSignal, key: str) -> bool:
    return (
        isinstance(signal.group_key, str)
        and signal.group_key.strip().casefold() == key.strip().casefold()
        and signal.role == "target"
    )


def _target_signals(candidate: Candidate, key: str) -> list[RouteSignal]:
    return [s for s in candidate.route_signals if _same_group(s, key)]


def _target_property_signals(
    candidate: Candidate, key: str
) -> list[RouteSignal]:
    """Property evidence from the target+criterion subquery, not the global one."""
    return [
        signal for signal in _target_signals(candidate, key)
        if signal.route == "target_property"
    ]


def _has_grouped_name(candidate: Candidate, key: str) -> bool:
    """Whether the candidate was resolved through this target's name route."""
    return any(
        signal.route == "name" for signal in _target_signals(candidate, key)
    )


def _has_grouped_clause_owner(candidate: Candidate, key: str) -> bool:
    """Whether deterministic question-clause attribution nominated this paper."""
    return any(
        signal.route == "clause_owner"
        for signal in _target_signals(candidate, key)
    )


def _has_grouped_self_definition_owner(
    candidate: Candidate, key: str
) -> bool:
    """Whether full-pool metadata self-definition nominated this owner."""
    return any(
        signal.route == "self_definition_owner"
        for signal in _target_signals(candidate, key)
    )


def _identity_text(value: str) -> str:
    """Normalize only enough to compare an explicit target with a title alias."""
    return "".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _matching_title_items(
    candidate: Candidate, target: PlanTarget
) -> list[dict]:
    """Title supports belonging to this group and referenced by its target."""
    if not (
        _has_grouped_name(candidate, target.key)
        or _has_grouped_clause_owner(candidate, target.key)
        or _has_grouped_self_definition_owner(candidate, target.key)
    ):
        return []
    target_text = _identity_text(target.text)
    if not target_text:
        return []
    return [
        item for item in candidate.support
        if (
            isinstance(item, dict)
            and item.get("source") in {"title_surface", "metadata_definition"}
            and isinstance(item.get("alias"), str)
            and isinstance(item.get("group_key"), str)
            and item["group_key"].strip().casefold()
            == target.key.strip().casefold()
            and bool(alias_text := _identity_text(item["alias"]))
            and (
                item.get("question_source_owner") is True
                or item.get("question_clause_owner") is True
                or item.get("question_self_definition_owner") is True
                or alias_text in target_text
            )
        )
    ]


def _informative_title_alias(value: str) -> bool:
    """Long or multi-token aliases are safer than short acronym collisions."""
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    return len(tokens) >= 2 or len("".join(tokens)) >= 6


def _title_alias_identifies_target(item: dict, target_text: str) -> bool:
    alias = str(item.get("alias") or "")
    return (
        item.get("question_source_owner") is True
        or item.get("question_clause_owner") is True
        or item.get("question_self_definition_owner") is True
        or _identity_text(alias) == target_text
        or _informative_title_alias(alias)
    )


def _has_exact_title_identity(candidate: Candidate, target: PlanTarget) -> bool:
    """Direct title identity, guarded against acronym/title collisions.

    A short title acronym is ownership evidence only when the planner made that
    exact acronym the target. A longer or multi-token unique title is also a
    safe source identity inside a longer requested-fact description.
    """
    target_text = _identity_text(target.text)
    matching_items = _matching_title_items(candidate, target)
    if not target_text or not matching_items:
        return False
    return any(
        item.get("title_match_kind") == "exact"
        and item.get("title_match_count") == 1
        and _title_alias_identifies_target(item, target_text)
        for item in matching_items
    )


def _has_question_source_owner_identity(
    candidate: Candidate, target: PlanTarget
) -> bool:
    """Explicit question-named reporting source, stronger than row aliases."""
    return any(
        item.get("question_source_owner") is True
        and item.get("title_match_kind") == "exact"
        and item.get("title_match_count") == 1
        for item in _matching_title_items(candidate, target)
    )


def _has_question_clause_owner_identity(
    candidate: Candidate, target: PlanTarget
) -> bool:
    """Unique descriptive source with independent answer-bearing retrieval.

    Clause/title lexical overlap nominates an owner, but it is not sufficient
    by itself.  The same paper must also have target-specific property support
    and ordinary body-passage retrieval before this high-priority identity can
    affect selection.
    """
    return bool(_target_property_signals(candidate, target.key)) and (
        "property" in candidate.provenance
    ) and any(
        item.get("question_clause_owner") is True
        and item.get("title_match_kind") == "descriptive"
        and item.get("title_match_count") == 1
        for item in _matching_title_items(candidate, target)
    )


def _has_question_self_definition_owner_identity(
    candidate: Candidate, target: PlanTarget
) -> bool:
    """Unique named owner plus an independently strong target-property hit."""
    if "self_definition_owner" not in candidate.provenance:
        return False
    targeted = _target_property_signals(candidate, target.key)
    if not targeted or min(signal.rank for signal in targeted) > (
        _SELF_DEFINITION_MAX_PROPERTY_RANK
    ):
        return False
    return any(
        item.get("question_self_definition_owner") is True
        and item.get("role") == "target"
        and item.get("title_match_kind") == "self_definition"
        and item.get("title_match_count") == 1
        and item.get("owner_corpus_match_count") == 1
        and item.get("definition_kind") in {
            "exact_title", "abstract_self_definition",
        }
        for item in _matching_title_items(candidate, target)
    )


def _has_title_property_identity(
    candidate: Candidate, target: PlanTarget
) -> bool:
    """Unique informative title surface plus answer-bearing passage match."""
    if not _target_property_signals(candidate, target.key):
        return False
    target_text = _identity_text(target.text)
    property_rank = min(
        signal.rank for signal in _target_property_signals(
            candidate, target.key
        )
    )
    return any(
        (
            item.get("title_match_kind") == "exact"
            and item.get("title_match_count") == 1
        )
        or (
            item.get("title_match_kind") in {"leading", "interior"}
            and item.get("title_surface_match_count") == 1
        )
        for item in _matching_title_items(candidate, target)
        if _title_alias_identifies_target(item, target_text) or (
            item.get("title_match_kind") == "exact"
            and property_rank <= _SHORT_TITLE_MAX_PROPERTY_RANK
        )
    )


def _global_support_signals(candidate: Candidate) -> list[RouteSignal]:
    """Secondary global relevance signals; never assignment eligibility."""
    return [
        signal for signal in candidate.route_signals
        if signal.group_key is None and signal.route in {"property", "knn"}
    ]


def _candidate_strength(
    candidate: Candidate, target: PlanTarget, original_position: int
) -> tuple:
    grouped = _target_signals(candidate, target.key)
    targeted_property = _target_property_signals(candidate, target.key)
    corroborating = _global_support_signals(candidate)
    signals = [*grouped, *corroborating]
    grouped_routes = {signal.route for signal in grouped}
    fusion = sum(1.0 / (_RRF_K + signal.rank) for signal in signals)
    group_rank = min(signal.rank for signal in grouped)
    target_fusion = sum(1.0 / (_RRF_K + signal.rank) for signal in grouped)
    property_rank = min(
        (signal.rank for signal in targeted_property), default=10**9
    )
    property_score = max(
        (
            signal.score for signal in targeted_property
            if isinstance(signal.score, (int, float))
            and not isinstance(signal.score, bool)
        ),
        default=float("-inf"),
    )
    question_source_owner = _has_question_source_owner_identity(
        candidate, target
    )
    question_clause_owner = _has_question_clause_owner_identity(
        candidate, target
    )
    question_self_definition_owner = (
        _has_question_self_definition_owner_identity(candidate, target)
    )
    exact_title_identity = _has_exact_title_identity(candidate, target)
    has_grouped_name = _has_grouped_name(candidate, target.key)
    title_property_identity = _has_title_property_identity(
        candidate, target
    )
    # One rank of alias corroboration is useful; it must not outweigh a large
    # answer-bearing rank gap.  This preserves a method at property@1 over an
    # uncorroborated generic property@0 result while still rejecting an alias
    # origin at property@16 in favour of the true reporting paper at @0.
    adjusted_property_rank = property_rank - int(
        has_grouped_name and property_rank < 10**9
    )
    # Exact title identity is direct ownership. A title explicitly referenced
    # by a longer fact target is also strong ownership, but only when the
    # requested property was found in that paper. Otherwise target-specific
    # answer-bearing rank dominates a generic alias agreement.
    return (
        0 if question_source_owner
        else 1 if question_clause_owner
        else 2 if question_self_definition_owner
        else 3 if exact_title_identity
        else 4 if title_property_identity
        else 5,
        adjusted_property_rank,
        0 if has_grouped_name else 1,
        -property_score,
        -len(grouped_routes),
        -target_fusion,
        -fusion,
        group_rank,
        original_position,
    )


def _eligible_for_target(candidate: Candidate, target: PlanTarget) -> bool:
    return bool(
        _target_property_signals(candidate, target.key)
    ) or _has_question_clause_owner_identity(
        candidate, target
    ) or _has_question_self_definition_owner_identity(
        candidate, target
    ) or _has_exact_title_identity(
        candidate, target
    )


def _assignment(candidate: Candidate, target: PlanTarget) -> TargetAssignment:
    grouped = _target_signals(candidate, target.key)
    targeted_property = _target_property_signals(candidate, target.key)
    all_signals = [*grouped, *_global_support_signals(candidate)]
    routes = {signal.route for signal in grouped}
    if _has_question_source_owner_identity(candidate, target):
        corroboration = ("question_source_owner",)
        if targeted_property:
            corroboration += ("target_property",)
    elif _has_question_clause_owner_identity(candidate, target):
        corroboration = ("question_clause_owner",)
        if targeted_property:
            corroboration += ("target_property",)
    elif _has_question_self_definition_owner_identity(candidate, target):
        corroboration = ("question_self_definition_owner", "target_property")
    elif _has_exact_title_identity(
        candidate, target
    ) or _has_title_property_identity(candidate, target):
        corroboration = ("title_identity",)
        if targeted_property:
            corroboration += ("target_property",)
    else:
        corroboration = (
            ("target_property", "alias")
            if "name" in routes else ("target_property",)
        )
    return TargetAssignment(
        target_key=target.key,
        paper_id=candidate.paper_id,
        group_rank=min((signal.rank for signal in grouped), default=0),
        target_property_rank=min(
            (signal.rank for signal in targeted_property), default=-1
        ),
        fusion_score=sum(
            1.0 / (_RRF_K + signal.rank) for signal in all_signals
        ),
        corroboration=corroboration,
    )


def _strongly_covers(candidate: Candidate, target: PlanTarget) -> bool:
    """High-confidence membership for the defensive single-source set cover."""
    if _has_exact_title_identity(
        candidate, target
    ) or _has_question_clause_owner_identity(
        candidate, target
    ) or _has_question_self_definition_owner_identity(candidate, target):
        return True
    return any(
        signal.rank <= _SET_COVER_MAX_PROPERTY_RANK
        for signal in _target_property_signals(candidate, target.key)
    )


def _is_non_target_name_only(candidate: Candidate) -> bool:
    """Do not emit a named evidence anchor/dataset solely because it was named."""
    signals = candidate.route_signals
    return bool(signals) and all(
        signal.route == "name" and signal.role != "target" for signal in signals
    )


def _floor_order(candidates: list[Candidate], plan: Plan) -> list[Candidate]:
    """Preserve property-first generally; prefer citation for anchor queries."""
    has_anchor = any(target.role == "evidence_anchor" for target in plan.targets)
    if has_anchor:
        priorities = ("citation", "property")
    else:
        priorities = ("property",)
    out: list[Candidate] = []
    seen: set[int] = set()
    for route in priorities:
        for candidate in candidates:
            if id(candidate) not in seen and route in candidate.provenance:
                seen.add(id(candidate))
                out.append(candidate)
    out.extend(candidate for candidate in candidates if id(candidate) not in seen)
    return out


def _floor_candidates(
    candidates: list[Candidate], plan: Plan
) -> tuple[list[Candidate], list[str]]:
    """The exact no-assignment floor and any evidence-anchor aliases it drops."""
    floor: list[Candidate] = []
    excluded: list[str] = []
    for candidate in _floor_order(candidates, plan):
        if _is_non_target_name_only(candidate):
            excluded.append(candidate.paper_id)
        else:
            floor.append(candidate)
    return floor, excluded


def select_target_coverage(
    candidates: list[Candidate], plan: Plan
) -> TargetSelectionResult:
    """Order candidates by target coverage, preserving a deterministic floor."""
    positions = {id(candidate): i for i, candidate in enumerate(candidates)}
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    assignments: list[TargetAssignment] = []
    uncovered: list[str] = []
    uncorroborated_counts: dict[str, int] = {}

    floor, excluded = _floor_candidates(candidates, plan)

    targets = [target for target in plan.targets if target.role == "target"]

    # Defensive set-cover path for one reporting source.  Even with the improved
    # planner prompt, this prevents two requested facts from forcing two papers
    # when the structured source budget says they share one owner.
    if plan.desired_paper_count == 1 and len(targets) > 1:
        coverage = {
            id(candidate): [
                target for target in targets
                if _strongly_covers(candidate, target)
            ]
            for candidate in candidates
        }
        choices = [candidate for candidate in candidates if coverage[id(candidate)]]
        if choices:
            chosen = min(
                choices,
                key=lambda candidate: (
                    -len(coverage[id(candidate)]),
                    tuple(
                        _candidate_strength(
                            candidate, target, positions[id(candidate)]
                        )
                        for target in coverage[id(candidate)]
                    ),
                    positions[id(candidate)],
                ),
            )
            selected.append(chosen)
            selected_ids.add(chosen.paper_id)
            covered_keys = {target.key for target in coverage[id(chosen)]}
            assignments.extend(
                _assignment(chosen, target)
                for target in targets if target.key in covered_keys
            )
            uncovered.extend(
                target.key for target in targets if target.key not in covered_keys
            )

            for candidate in floor:
                if candidate.paper_id not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate.paper_id)

            return TargetSelectionResult(
                paper_ids=[candidate.paper_id for candidate in selected],
                assignments=assignments,
                uncovered_targets=uncovered,
                uncorroborated_target_counts=uncorroborated_counts,
                excluded_non_target_anchors=excluded,
                floor_paper_ids=[candidate.paper_id for candidate in floor],
            )

    for target in targets:
        grouped_choices = [
            candidate for candidate in candidates
            if candidate.paper_id not in selected_ids
            and _target_signals(candidate, target.key)
        ]
        uncorroborated = [
            candidate for candidate in grouped_choices
            if not _target_property_signals(candidate, target.key)
        ]
        if uncorroborated:
            uncorroborated_counts[target.key] = len(uncorroborated)
        choices = [
            candidate for candidate in grouped_choices
            if _eligible_for_target(candidate, target)
        ]
        if not choices:
            uncovered.append(target.key)
            continue
        chosen = min(
            choices,
            key=lambda candidate: _candidate_strength(
                candidate, target, positions[id(candidate)]
            ),
        )
        selected.append(chosen)
        selected_ids.add(chosen.paper_id)
        assignments.append(_assignment(chosen, target))

    for candidate in floor:
        if candidate.paper_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.paper_id)

    return TargetSelectionResult(
        paper_ids=[candidate.paper_id for candidate in selected],
        assignments=assignments,
        uncovered_targets=uncovered,
        uncorroborated_target_counts=uncorroborated_counts,
        excluded_non_target_anchors=excluded,
        floor_paper_ids=[candidate.paper_id for candidate in floor],
    )
