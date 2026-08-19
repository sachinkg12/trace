"""Compact, read-only retrieval diagnostics for experiment traces.

The dossier is deliberately downstream of selection: it serializes the
candidate objects the runner already produced and never feeds a value back into
ranking.  This makes target/routing/support failures inspectable without
changing a submission artifact.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from littraceqa.paperset.cascade import Candidate, RouteSignal

_SUPPORT_LIMIT = 2
_TEXT_LIMIT = 320
_SUPPORT_FIELDS = (
    "source",
    "source_type",
    "route",
    "page",
    "object_id",
    "parser_visible_id",
    "object_key",
    "score",
    "alias",
    "anchor",
    "chunk_id",
    "paper_id",
    "group_key",
    "role",
    "section_kind",
    "title_match_kind",
    "title_match_count",
    "title_surface_match_count",
    "question_source_owner",
    "question_clause_owner",
    "question_self_definition_owner",
    "definition_kind",
    "owner_corpus_match_count",
    "owner_clause",
    "owner_score",
    "owner_runner_up_score",
)


def _short_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= _TEXT_LIMIT:
        return normalized
    return normalized[: _TEXT_LIMIT - 1].rstrip() + "…"


def _safe_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _support_dict(raw: Mapping) -> dict[str, Any]:
    item = {
        key: _safe_scalar(raw.get(key))
        for key in _SUPPORT_FIELDS
        if raw.get(key) is not None
    }
    text = _short_text(raw.get("text"))
    if text:
        item["text"] = text
    return item


def _raw_support(candidate: object) -> list[Mapping]:
    raw_support = getattr(candidate, "support", ())
    if not isinstance(raw_support, Sequence) or isinstance(raw_support, (str, bytes)):
        return []
    return [raw for raw in raw_support if isinstance(raw, Mapping)]


def _support_items(candidate: object) -> list[dict[str, Any]]:
    return [
        _support_dict(raw) for raw in _raw_support(candidate)[:_SUPPORT_LIMIT]
    ]


def _title_identity_items(candidate: object) -> list[dict[str, Any]]:
    """Losslessly preserve selector-critical owner metadata outside the cap."""
    return [
        _support_dict(raw)
        for raw in _raw_support(candidate)
        if raw.get("source") in {"title_surface", "metadata_definition"}
    ]


def _signal_dict(signal: object) -> dict[str, Any]:
    return {
        "route": str(getattr(signal, "route", "") or ""),
        "rank": int(getattr(signal, "rank", 0) or 0),
        "score": _safe_scalar(getattr(signal, "score", None)),
        "group_key": getattr(signal, "group_key", None),
        "role": str(getattr(signal, "role", "target") or "target"),
    }


def _grouped_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for signal in signals:
        group_key = signal.get("group_key")
        if not isinstance(group_key, str) or not group_key.strip():
            continue
        role = str(signal.get("role") or "target")
        grouped.setdefault((role, group_key), []).append({
            "route": signal["route"],
            "rank": signal["rank"],
            "score": signal["score"],
        })
    return [
        {"group_key": group_key, "role": role, "signals": values}
        for (role, group_key), values in grouped.items()
    ]


def _paper_metadata(pool: object, paper_id: str) -> dict[str, Any]:
    by_id = getattr(pool, "by_id", None)
    if not callable(by_id):
        return {"title": None, "venue": None, "year": None}
    try:
        paper = by_id(paper_id)
    except Exception:  # noqa: BLE001 -- diagnostics must never break a run
        paper = None
    return {
        "title": getattr(paper, "title", None) if paper is not None else None,
        "venue": getattr(paper, "venue", None) if paper is not None else None,
        "year": getattr(paper, "year", None) if paper is not None else None,
    }


def build_candidate_dossiers(
    candidates: Sequence[object],
    *,
    pool: object,
    ranked_paper_ids: Sequence[str],
    emitted_paper_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Serialize every candidate with routing, support, and selection context."""
    ranked_positions = {paper_id: i for i, paper_id in enumerate(ranked_paper_ids)}
    emitted_positions = {paper_id: i for i, paper_id in enumerate(emitted_paper_ids)}
    dossiers: list[dict[str, Any]] = []
    for candidate_position, candidate in enumerate(candidates):
        paper_id = str(getattr(candidate, "paper_id", "") or "")
        signals = [
            _signal_dict(signal)
            for signal in (getattr(candidate, "route_signals", ()) or ())
        ]
        supports = _support_items(candidate)
        try:
            best_support = _short_text(candidate.best_support_text())
        except Exception:  # noqa: BLE001 -- diagnostics must never break a run
            best_support = ""
        provenance = [
            str(route) for route in (getattr(candidate, "provenance", ()) or ())
        ]
        dossiers.append({
            "paper_id": paper_id,
            **_paper_metadata(pool, paper_id),
            "candidate_position": candidate_position,
            "ranked_position": ranked_positions.get(paper_id),
            "emitted_position": emitted_positions.get(paper_id),
            "provenance": provenance,
            "route_signals": signals,
            "target_groups": _grouped_signals(signals),
            "support": supports,
            # Selection depends on title match kind, uniqueness, and semantic
            # group. Generic support remains compact, but these records must
            # survive a frozen dossier round trip even when they occur after
            # the two-excerpt cap.
            "title_identity_support": _title_identity_items(candidate),
            "best_support_text": best_support or None,
            "support_text_present": bool(
                best_support or any(item.get("text") for item in supports)
            ),
            "relation_support_present": bool(
                getattr(candidate, "relation_support", None)
            ),
        })
    return dossiers


def candidate_from_dossier(dossier: Mapping[str, Any]) -> Candidate:
    """Reconstruct the selector-relevant Candidate state from one dossier."""
    support = [
        dict(item)
        for item in (dossier.get("support") or [])
        if isinstance(item, Mapping)
    ]
    for raw in dossier.get("title_identity_support") or []:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        if item not in support:
            support.append(item)
    signals = [
        RouteSignal(
            route=str(raw.get("route") or ""),
            rank=int(raw.get("rank") or 0),
            score=raw.get("score"),
            group_key=raw.get("group_key"),
            role=str(raw.get("role") or "target"),
        )
        for raw in (dossier.get("route_signals") or [])
        if isinstance(raw, Mapping)
    ]
    return Candidate(
        paper_id=str(dossier.get("paper_id") or ""),
        provenance=[str(item) for item in (dossier.get("provenance") or [])],
        support=support,
        route_signals=signals,
    )
