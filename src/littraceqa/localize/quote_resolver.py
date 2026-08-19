"""Resolve verbatim evidence anchors to immutable PDF source objects.

The resolver never trusts a proposed page number.  It searches the exact PDF
snapshot represented by :class:`SourceLedger`, enforces source-type and visible
object constraints, and abstains on ties.  This module produces decisions only;
it does not mutate predictions or select evidence under a global cap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Sequence

from littraceqa.evidence import make_evidence
from littraceqa.localize.source_ledger import (
    SourceBlock,
    SourceLedger,
    canonical_source_object_id,
    normalize_anchor_text,
)


_KNOWN_SOURCE_TYPES = {
    "text_span",
    "table",
    "figure",
    "equation_algorithm",
    "citation_context",
}
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?(?!\w)")
_CONTENT_RE = re.compile(r"[a-z]{3,}|[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?")


@dataclass(frozen=True)
class EvidenceQuoteTarget:
    target_id: str
    paper_id: str
    quote: str
    expected_source_type: str | None = None
    visible_id: str | None = None
    prefix: str = ""
    suffix: str = ""
    expected_pdf_sha256: str | None = None


@dataclass(frozen=True)
class QuoteResolution:
    target: EvidenceQuoteTarget
    status: str
    block: SourceBlock | None = None
    evidence: dict | None = None
    match_kind: str | None = None
    rank: tuple[int, int] | None = None
    candidate_block_ids: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


@dataclass(frozen=True)
class _BlockMatch:
    block: SourceBlock
    kind: str
    # Lower is better. The second component is negative context strength.
    rank: tuple[int, int]


def _numeric_atoms(value: str) -> tuple[str, ...]:
    return tuple(_NUMBER_RE.findall(normalize_anchor_text(value)))


def _content_tokens(value: str) -> set[str]:
    return set(_CONTENT_RE.findall(normalize_anchor_text(value)))


def _context_strength(target: EvidenceQuoteTarget, block: SourceBlock) -> int:
    score = 0
    prefix = normalize_anchor_text(target.prefix)
    suffix = normalize_anchor_text(target.suffix)
    if prefix and (
        prefix in block.prefix
        or block.prefix.endswith(prefix[-min(80, len(prefix)):])
    ):
        score += 1
    if suffix and (
        suffix in block.suffix
        or block.suffix.startswith(suffix[:min(80, len(suffix))])
    ):
        score += 1
    return score


def _quote_match(
    target: EvidenceQuoteTarget,
    block: SourceBlock,
    *,
    object_exact: bool,
) -> _BlockMatch | None:
    quote = normalize_anchor_text(target.quote)
    if not quote:
        return None
    context_strength = _context_strength(target, block)
    if quote == block.normalized_text:
        base = 1
        kind = "exact_block"
    elif len(quote) >= 4 and quote in block.normalized_text:
        base = 2
        kind = "exact_substring"
    else:
        quote_numbers = set(_numeric_atoms(quote))
        quote_tokens = _content_tokens(quote)
        block_tokens = _content_tokens(block.normalized_text)
        coverage = (
            len(quote_tokens & block_tokens) / len(quote_tokens)
            if quote_tokens else 0.0
        )
        if (
            context_strength < 1
            or not quote_numbers
            or not quote_numbers.issubset(set(_numeric_atoms(block.normalized_text)))
            or coverage < 0.8
        ):
            return None
        base = 3
        kind = "context_numeric"
    if object_exact:
        base = 0
        kind = f"visible_id_{kind}"
    return _BlockMatch(
        block=block,
        kind=kind,
        rank=(base, -context_strength),
    )


def _visible_kind(source_type: str) -> str:
    return {
        "table": "table",
        "figure": "figure",
        "equation_algorithm": "equation",
        "citation_context": "citation",
    }[source_type]


def _canonical_target_id(target: EvidenceQuoteTarget) -> str | None:
    raw = str(target.visible_id or "").strip()
    if not raw:
        return None
    source_type = target.expected_source_type
    if source_type not in {
        "table", "figure", "equation_algorithm", "citation_context"
    }:
        return None
    kind = _visible_kind(source_type)
    # Algorithm labels remain algorithm labels under the scorer even though
    # they share the equation_algorithm source type.
    if source_type == "equation_algorithm" and raw.casefold().startswith("alg"):
        kind = "algorithm"
    return canonical_source_object_id(raw, kind)


def _matches_without_constraints(
    target: EvidenceQuoteTarget, blocks: Sequence[SourceBlock]
) -> list[_BlockMatch]:
    return [
        match
        for block in blocks
        if (match := _quote_match(target, block, object_exact=False)) is not None
    ]


def resolve_quote_target(
    target: EvidenceQuoteTarget,
    ledger: SourceLedger,
) -> QuoteResolution:
    """Resolve one target or return a precise fail-closed status."""
    if target.paper_id != ledger.manifest.paper_id:
        return QuoteResolution(target, "snapshot_mismatch")
    expected_sha = str(target.expected_pdf_sha256 or "").strip().casefold()
    if expected_sha and expected_sha != ledger.manifest.sha256.casefold():
        return QuoteResolution(target, "snapshot_mismatch")
    source_type = target.expected_source_type
    if source_type is not None and source_type not in _KNOWN_SOURCE_TYPES:
        raise ValueError(f"unknown expected source type: {source_type!r}")
    if not normalize_anchor_text(target.quote):
        return QuoteResolution(target, "no_anchor")

    unconstrained = _matches_without_constraints(target, ledger.blocks)
    if not unconstrained:
        return QuoteResolution(target, "no_anchor")
    type_compatible = [
        match for match in unconstrained
        if source_type is None or match.block.source_type == source_type
    ]
    if not type_compatible:
        return QuoteResolution(
            target,
            "wrong_source_type",
            candidate_block_ids=tuple(sorted(
                match.block.block_id for match in unconstrained
            )),
        )

    canonical_id = _canonical_target_id(target)
    if target.visible_id and canonical_id is None:
        return QuoteResolution(target, "object_id_missing")
    constrained: list[_BlockMatch] = []
    for match in type_compatible:
        if canonical_id is not None and match.block.object_id != canonical_id:
            continue
        constrained_match = _quote_match(
            target,
            match.block,
            object_exact=canonical_id is not None,
        )
        if constrained_match is not None:
            constrained.append(constrained_match)
    if canonical_id is not None and not constrained:
        return QuoteResolution(
            target,
            "object_id_missing",
            candidate_block_ids=tuple(sorted(
                match.block.block_id for match in type_compatible
            )),
        )
    if not constrained:
        return QuoteResolution(target, "no_anchor")

    best_rank = min(match.rank for match in constrained)
    best = sorted(
        (match for match in constrained if match.rank == best_rank),
        key=lambda match: match.block.block_id,
    )
    if len(best) != 1:
        return QuoteResolution(
            target,
            "ambiguous_anchor",
            rank=best_rank,
            candidate_block_ids=tuple(match.block.block_id for match in best),
        )
    chosen = best[0]
    try:
        evidence = make_evidence(
            target.paper_id,
            chosen.block.source_type,
            chosen.block.physical_page,
            chosen.block.object_id,
        )
    except (KeyError, TypeError, ValueError):
        return QuoteResolution(
            target,
            "object_id_missing",
            rank=chosen.rank,
            candidate_block_ids=(chosen.block.block_id,),
        )
    return QuoteResolution(
        target=target,
        status="resolved",
        block=chosen.block,
        evidence=evidence,
        match_kind=chosen.kind,
        rank=chosen.rank,
        candidate_block_ids=(chosen.block.block_id,),
    )


def resolve_quote_targets(
    targets: Sequence[EvidenceQuoteTarget],
    ledgers: dict[str, SourceLedger],
) -> tuple[QuoteResolution, ...]:
    """Resolve targets independently, then enforce one source block per target.

    If two targets make equally strong claims on the same block, both abstain;
    an input-order tie-break would invent source ownership.  A strictly stronger
    claim keeps the block and the weaker claim abstains.
    """
    resolutions: list[QuoteResolution] = []
    for target in targets:
        ledger = ledgers.get(target.paper_id)
        resolutions.append(
            resolve_quote_target(target, ledger)
            if ledger is not None
            else QuoteResolution(target, "snapshot_mismatch")
        )

    by_block: dict[str, list[int]] = {}
    for position, resolution in enumerate(resolutions):
        if resolution.resolved and resolution.block is not None:
            by_block.setdefault(resolution.block.block_id, []).append(position)
    for positions in by_block.values():
        if len(positions) < 2:
            continue
        ranked = sorted(
            positions,
            key=lambda position: (
                resolutions[position].rank or (10**9, 0),
                resolutions[position].target.target_id,
            ),
        )
        best_rank = resolutions[ranked[0]].rank
        winners = [
            position
            for position in ranked
            if resolutions[position].rank == best_rank
        ]
        rejected = positions if len(winners) > 1 else [
            position for position in positions if position != winners[0]
        ]
        for position in rejected:
            resolution = resolutions[position]
            resolutions[position] = replace(
                resolution,
                status="ambiguous_anchor",
                block=None,
                evidence=None,
                match_kind=None,
            )
    return tuple(resolutions)
