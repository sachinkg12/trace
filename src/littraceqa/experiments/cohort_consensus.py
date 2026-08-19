"""Gold-blind batch consensus for repeated question cohorts.

LitTraceQA inputs may contain several questions about the same paper cohort.
The per-question selector can find different members of that cohort while a
cardinality decision truncates each focused question to one paper.  This
module combines only independently repeated signals from a select-only batch:

* a component must be connected to a multi-target anchor question;
* records connect to an anchor through a named-method match or an emitted-paper
  overlap;
* the component must contain at least ``min_records`` records; and
* a paper must have been emitted by at least ``min_votes`` distinct records.

No gold answers, labels, locators, or paper IDs are consumed.  The conservative
defaults intentionally abstain on small or weakly supported components.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from littraceqa.experiments.driver import (
    DEFAULT_PER_RECORD_TIMEOUT,
    compute_records,
)
from littraceqa.experiments.runner import DevRunner
from littraceqa.pipeline.input import InputRecord

COHORT_CONSENSUS = "cohort_consensus"
SUPPORTED_BATCH_SELECTIONS = frozenset({COHORT_CONSENSUS})


@dataclass(frozen=True)
class CohortConsensusConfig:
    """Safety thresholds for a batch-consensus decision."""

    min_records: int = 6
    min_votes: int = 2
    max_papers: int = 5

    def __post_init__(self) -> None:
        for name, value in (
            ("min_records", self.min_records),
            ("min_votes", self.min_votes),
            ("max_papers", self.max_papers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CohortConsensusResult:
    """Derived overrides plus batch-level audit metadata."""

    overrides: dict[str, dict[str, Any]]
    summary: dict[str, Any]


@runtime_checkable
class PaperOverrideRunner(Protocol):
    """Opt-in runner seam used by the composition roots for a two-pass run."""

    def selection_pass_runner(self) -> DevRunner: ...

    def with_paper_overrides(
        self, overrides: Mapping[str, Mapping[str, Any]]
    ) -> DevRunner: ...


def _normalize_method(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _named_methods(trace: Mapping[str, Any]) -> set[str]:
    plan = trace.get("plan") or {}
    raw = plan.get("named_methods") or []
    return {
        normalized
        for value in raw
        if len(normalized := _normalize_method(value)) >= 3
    }


def _methods_match(left: str, right: str) -> bool:
    if left == right:
        return True
    # Substring matching is useful for names such as D-FINE / DEIM-D-FINE-X,
    # but short acronyms would create many accidental edges.
    return min(len(left), len(right)) >= 4 and (left in right or right in left)


def _is_anchor(trace: Mapping[str, Any]) -> bool:
    plan = trace.get("plan") or {}
    targets = [
        target
        for target in (plan.get("targets") or [])
        if isinstance(target, Mapping) and target.get("role") == "target"
    ]
    return (
        plan.get("multiplicity") == "multi"
        and len(targets) >= 2
        and len(trace.get("paper_ids") or []) >= 2
    )


def _connected_components(
    traces_by_id: Mapping[str, Mapping[str, Any]],
) -> list[list[str]]:
    """Build anchor-centred components without a semantic-model dependency."""
    anchors = {
        query_id
        for query_id, trace in traces_by_id.items()
        if _is_anchor(trace)
    }
    adjacency: dict[str, set[str]] = defaultdict(set)
    for anchor_id in anchors:
        anchor = traces_by_id[anchor_id]
        anchor_methods = _named_methods(anchor)
        anchor_papers = set(anchor.get("paper_ids") or [])
        for query_id, trace in traces_by_id.items():
            if query_id == anchor_id:
                continue
            method_hit = any(
                _methods_match(left, right)
                for left in anchor_methods
                for right in _named_methods(trace)
            )
            paper_hit = bool(anchor_papers & set(trace.get("paper_ids") or []))
            if method_hit or paper_hit:
                adjacency[anchor_id].add(query_id)
                adjacency[query_id].add(anchor_id)

    components: list[list[str]] = []
    seen: set[str] = set()
    for query_id in traces_by_id:
        if query_id in seen or not adjacency.get(query_id):
            continue
        stack = [query_id]
        seen.add(query_id)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def derive_cohort_consensus(
    traces: Sequence[Mapping[str, Any]],
    *,
    config: CohortConsensusConfig | None = None,
) -> CohortConsensusResult:
    """Derive paper overrides from select-only traces, never from gold data."""
    active = config or CohortConsensusConfig()
    traces_by_id: dict[str, Mapping[str, Any]] = {}
    for trace in traces:
        query_id = str(trace.get("query_id") or "").strip()
        if not query_id:
            raise ValueError("select-only trace is missing query_id")
        if query_id in traces_by_id:
            raise ValueError(f"duplicate select-only trace for {query_id}")
        traces_by_id[query_id] = trace

    overrides: dict[str, dict[str, Any]] = {}
    qualified_components: list[dict[str, Any]] = []
    components = _connected_components(traces_by_id)
    for component in components:
        if len(component) < active.min_records:
            continue

        votes: Counter[str] = Counter()
        best_position: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        seen_index = 0
        for query_id in component:
            # One record contributes at most one vote to a paper, even if a
            # malformed trace were to repeat that paper.
            record_seen: set[str] = set()
            for position, raw_paper_id in enumerate(
                traces_by_id[query_id].get("paper_ids") or []
            ):
                paper_id = str(raw_paper_id or "").strip()
                if not paper_id or paper_id in record_seen:
                    continue
                record_seen.add(paper_id)
                votes[paper_id] += 1
                best_position[paper_id] = min(
                    best_position.get(paper_id, position), position
                )
                if paper_id not in first_seen:
                    first_seen[paper_id] = seen_index
                    seen_index += 1

        ordered = sorted(
            (paper_id for paper_id, count in votes.items() if count >= active.min_votes),
            key=lambda paper_id: (
                -votes[paper_id],
                best_position[paper_id],
                first_seen[paper_id],
                paper_id,
            ),
        )
        if not (2 <= len(ordered) <= active.max_papers):
            continue

        support = {paper_id: votes[paper_id] for paper_id in ordered}
        component_metadata = {
            "query_ids": list(component),
            "paper_ids": list(ordered),
            "support_counts": support,
        }
        qualified_components.append(component_metadata)
        for query_id in component:
            overrides[query_id] = {
                "mode": COHORT_CONSENSUS,
                "cohort_query_ids": list(component),
                "paper_ids": list(ordered),
                "support_counts": dict(support),
                "min_records": active.min_records,
                "min_votes": active.min_votes,
                "max_papers": active.max_papers,
            }

    return CohortConsensusResult(
        overrides=overrides,
        summary={
            "mode": COHORT_CONSENSUS,
            "n_records": len(traces_by_id),
            "n_anchor_components": len(components),
            "n_qualified_components": len(qualified_components),
            "n_overridden_records": len(overrides),
            "components": qualified_components,
            "config": {
                "min_records": active.min_records,
                "min_votes": active.min_votes,
                "max_papers": active.max_papers,
            },
        },
    )


def prepare_batch_runner(
    runner: DevRunner,
    records: Sequence[InputRecord],
    *,
    batch_selection: str | None,
    min_records: int = 6,
    min_votes: int = 2,
    max_papers: int = 5,
    workers: int = 1,
    per_record_timeout: float = DEFAULT_PER_RECORD_TIMEOUT,
) -> tuple[DevRunner, dict[str, Any] | None]:
    """Run the gold-blind select-only pass and return an override-enabled clone."""
    if batch_selection in (None, "", False):
        return runner, None
    if batch_selection not in SUPPORTED_BATCH_SELECTIONS:
        raise ValueError(
            f"unknown batch_selection {batch_selection!r}; "
            f"valid: {sorted(SUPPORTED_BATCH_SELECTIONS)}"
        )
    if not isinstance(runner, PaperOverrideRunner):
        raise TypeError(
            f"batch_selection={batch_selection!r} requires a paper-override runner"
        )

    selection_runner = runner.selection_pass_runner()
    selected_records, computed = compute_records(
        selection_runner,
        list(records),
        workers=workers,
        per_record_timeout=per_record_timeout,
    )
    failures = [
        record.query_id
        for record, (_line, _trace, failure_reason) in zip(selected_records, computed)
        if failure_reason
    ]
    if failures:
        raise RuntimeError(
            "batch select-only pass failed closed for: " + ", ".join(failures)
        )

    result = derive_cohort_consensus(
        [copy.deepcopy(trace) for _line, trace, _failure in computed],
        config=CohortConsensusConfig(
            min_records=min_records,
            min_votes=min_votes,
            max_papers=max_papers,
        ),
    )
    return runner.with_paper_overrides(result.overrides), result.summary
