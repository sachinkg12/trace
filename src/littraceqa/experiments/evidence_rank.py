"""Independent, calibrated ranking of RAW localized evidence (FIX 2).

Evidence-F1 used to be coupled to the answer string: only the evidence the
answer strategy ATTESTED was submitted, so a correct localization was thrown
away whenever the answer strategy failed or fell below its confidence floor.
This module decouples the two -- it ranks the `LocatedEvidence` the localizer
produced (independent of any answer) and emits the top items directly.

RANKING (0-LLM, deterministic -- documented so the ladder can reason about it):
  1. PRIMARY -- the localizer's own `confidence` (descending). The localizer is
     the calibrated judge of "is this the evidence"; we trust its score first.
  2. SECONDARY -- BM25 relevance of the evidence TEXT (quote + object_id +
     source_type) to the QUESTION, descending. Reuses `retrieval/bm25.py`'s
     verified BM25 (a tiny per-question index over the candidate quotes), so
     when two items tie on confidence the one whose text better matches the
     question wins.
  3. TIEBREAK -- the original localization order (stable): earlier-localized
     (earlier selected paper / earlier item) wins, so the ordering is fully
     deterministic even when confidence AND BM25 tie exactly.

EMISSION: rank, then convert each item to a scorer-shaped evidence dict via
`littraceqa.evidence.make_evidence` (skipping any that don't validate), and keep
at most `cap` VALID dicts. The cap bounds evidence-precision damage from a long
localized tail while keeping the high-confidence head. `DEFAULT_EVIDENCE_EMIT_CAP`
is the default; the runner threads `params.evidence_emit_cap` to override it.
"""
from __future__ import annotations

import re

from littraceqa.answer.scorer_contract import coarse_evidence_key
from littraceqa.evidence import make_evidence
from littraceqa.localize.interfaces import LocatedEvidence
from littraceqa.retrieval.bm25 import BM25Index, tokenize

# Default number of localized-evidence items to emit per question. Gold evidence
# sets are small; emitting a handful of the highest-confidence, most on-topic
# locators trades a little recall for evidence precision (F1). Tunable per-config
# via `params.evidence_emit_cap`.
DEFAULT_EVIDENCE_EMIT_CAP = 5

_EXPLICIT_TABLE_RE = re.compile(r"\b(?:table|tables|tabulated)\b", re.IGNORECASE)


def _evidence_doc(ev: LocatedEvidence) -> str:
    """The text used to BM25-match an evidence item against the question."""
    return " ".join(
        part for part in (ev.quote, ev.object_id, ev.source_type) if part
    )


def rank_localized_evidence(
    question: str, evidence: list[LocatedEvidence]
) -> list[LocatedEvidence]:
    """Return `evidence` ordered by the calibrated ranker (confidence, then
    BM25 relevance to `question`, then stable original order). Pure sort -- no
    cap, no conversion. Empty in -> empty out."""
    if not evidence:
        return []
    # Tiny per-question BM25 index over the candidate evidence texts; score every
    # candidate against the question. Missing (zero-overlap) => score 0.0.
    index = BM25Index(
        [str(i) for i in range(len(evidence))],
        [tokenize(_evidence_doc(ev)) for ev in evidence],
    )
    scores = index.score_all(tokenize(question))
    order = sorted(
        range(len(evidence)),
        key=lambda i: (-float(evidence[i].confidence), -scores.get(i, 0.0), i),
    )
    return [evidence[i] for i in order]


def _dedup_by_coarse_key(dicts: list[dict]) -> list[dict]:
    """Drop evidence dicts that collapse to the SAME official evaluator key
    (`vendor.evaluate.coarse_evidence_key`), keeping the first (highest-ranked).
    The scorer converts predicted evidence to a SET of these keys, so a duplicate
    earns no extra credit -- but WITHOUT this it still consumes an emit-cap slot,
    evicting a distinct locator that would have scored (verified: submission had
    147 entries / 105 unique / 9 questions at the cap). Order-preserving."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for d in dicts:
        key = coarse_evidence_key(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _ranked_valid_pairs(
    question: str, evidence: list[LocatedEvidence]
) -> list[tuple[LocatedEvidence, dict]]:
    """Convert ranked candidates and deduplicate by the scorer's exact key."""
    seen: set[tuple] = set()
    out: list[tuple[LocatedEvidence, dict]] = []
    for ev in rank_localized_evidence(question, evidence):
        try:
            shaped = make_evidence(
                ev.paper_id, ev.source_type, ev.page, ev.object_id
            )
        except (ValueError, KeyError):
            continue
        key = coarse_evidence_key(shaped)
        if key in seen:
            continue
        seen.add(key)
        out.append((ev, shaped))
    return out


def _should_rebalance_explicit_table_pairs(
    question: str,
    pairs: list[tuple[LocatedEvidence, dict]],
    cap: int | None,
) -> bool:
    """Whether the normal cap is monopolized on an explicit table request.

    This intentionally excludes figures and implicit/table-answer-type guesses.
    It activates only when at least three papers have valid table candidates and
    the normal capped output contains table locators for strictly fewer than
    half of those papers. Non-table evidence from a paper does not satisfy an
    explicit table contract.
    """
    if cap is None or cap < 2 or not _EXPLICIT_TABLE_RE.search(question):
        return False
    table_papers = {
        ev.paper_id for ev, _ in pairs if ev.source_type == "table"
    }
    if len(table_papers) < 3:
        return False
    capped_table_papers = {
        ev.paper_id for ev, _ in pairs[:cap] if ev.source_type == "table"
    }
    return 2 * len(capped_table_papers) < len(table_papers)


def explicit_table_rebalance_triggered(
    question: str,
    evidence: list[LocatedEvidence],
    *,
    cap: int | None = DEFAULT_EVIDENCE_EMIT_CAP,
) -> bool:
    """Expose the narrow balancing decision for traces and offline gates."""
    return _should_rebalance_explicit_table_pairs(
        question, _ranked_valid_pairs(question, evidence), cap
    )


def _rebalance_explicit_table_pairs(
    pairs: list[tuple[LocatedEvidence, dict]],
) -> list[tuple[LocatedEvidence, dict]]:
    """Put each paper's best-ranked table first, then preserve the ranked tail."""
    selected: list[tuple[LocatedEvidence, dict]] = []
    selected_keys: set[tuple] = set()
    seen_papers: set[str] = set()
    for pair in pairs:
        ev, shaped = pair
        if ev.source_type != "table" or ev.paper_id in seen_papers:
            continue
        selected.append(pair)
        selected_keys.add(coarse_evidence_key(shaped))
        seen_papers.add(ev.paper_id)
    for pair in pairs:
        key = coarse_evidence_key(pair[1])
        if key not in selected_keys:
            selected.append(pair)
            selected_keys.add(key)
    return selected


def build_emitted_evidence(
    question: str,
    evidence: list[LocatedEvidence],
    *,
    cap: int | None = DEFAULT_EVIDENCE_EMIT_CAP,
    rebalance_explicit_table: bool = False,
) -> list[dict]:
    """Rank the localized evidence, convert each (in ranked order) to a
    validator-shaped scorer dict via `make_evidence` (skipping any that don't
    validate), DEDUP by the official coarse evaluator key BEFORE the cap so
    duplicates never evict a distinct locator, then keep at most `cap` VALID
    dicts (all when `cap` is None)."""
    pairs = _ranked_valid_pairs(question, evidence)
    if (
        rebalance_explicit_table
        and _should_rebalance_explicit_table_pairs(question, pairs, cap)
    ):
        pairs = _rebalance_explicit_table_pairs(pairs)
    deduped = [shaped for _, shaped in pairs]
    return deduped[:cap] if cap is not None else deduped
