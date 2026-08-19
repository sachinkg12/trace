"""The `path:"new"` dev-runner: planner -> router -> precision-cascade.

`NewPathRunner` implements the SAME `DevRunner` seam as `OldPathRunner`, so the
driver/factory are unchanged. It mirrors `Pipeline._attempt`'s evidence->answer
->build_line flow but SWAPS the retrieval front-end: instead of seed-finder +
paperset-selector over live-fetched PDFs, it runs

    planner.plan(record)                      # QuestionPlanner -> Plan (router)
    run_strategies(plan, ctx, ...)            # index-backed retrieval -> Candidates
    cascade.select(question, cands, crit)     # precision gate -> paper_ids
    <cap by multiplicity>
    EvidenceLocalizationService(...) + AnswerPipeline(...)   # reused verbatim

and reads PDFs from the CORPUS snapshot (`CorpusPdfFetcher`), never live.

`build_cascade(stage_names, ...)` is the ABLATION RUNG SELECTOR: one config's
`cascade_stages` list -> one ordered cascade -> one experiment rung. `[]` builds
an empty cascade; the composition root treats an empty stage list as BYPASS
(rung "new-raw"), passing `cascade=None` so paper_ids come straight from the
deduped candidates.

Never-drop / never-crash: `run_one` wraps the WHOLE flow so any failure still
returns a validator-safe fallback line for `record.query_id` with `failure_reason`
set (a dropped query_id scores 0 precision) -- exactly like `OldPathRunner`.
Trace numerics (`cost`, `latency_s`, `n_candidates`) are real floats/ints so the
driver's `_as_float` summation never trips.
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from littraceqa.experiments.config import RunConfig

from littraceqa.answer.interfaces import AnswerContext
from littraceqa.experiments.evidence_rank import (
    DEFAULT_EVIDENCE_EMIT_CAP,
    build_emitted_evidence,
    explicit_table_rebalance_triggered,
)
from littraceqa.experiments.retrieval_dossier import build_candidate_dossiers
from littraceqa.experiments.runner import RunOutcome, make_trace
from littraceqa.localize.interfaces import LocatedEvidence, ParsedPdf
from littraceqa.paperset.cascade import (
    CascadeResult,
    CrossEncoderRerankStage,
    DeterministicFilterStage,
    EvidenceAttestationStage,
    LLMValidatorStage,
    PrecisionCascade,
    Stage,
)
from littraceqa.pipeline.input import InputRecord
from littraceqa.retrieval.answer_bearing_selection import select_answer_bearing
from littraceqa.retrieval.cardinality import explicit_target_paper_count
from littraceqa.retrieval.selection_filter import filter_by_venue_year
from littraceqa.retrieval.table_container_selection import select_table_container
from littraceqa.retrieval.strategy import (
    DEFAULT_PROPERTY_SEARCH_K,
    RetrievalContext,
    run_strategies,
)
from littraceqa.retrieval.target_selection import select_target_coverage
from littraceqa.submission import (
    answer_fallback_types,
    build_fallback_line,
    build_record_line,
)

# Short cascade-stage NAMES (what a config's `cascade_stages` list uses) mapped
# to the Stage builders. Keeping the config vocabulary tiny + stable is what
# makes ONE config == ONE ablation rung. Adding a stage is a one-line entry here
# (OCP: the cascade itself is untouched).
_STAGE_BUILDERS = {
    "filter": lambda *, pool, reranker, llm, parsed_by_id, top_k: (
        DeterministicFilterStage(pool=pool)
    ),
    "rerank": lambda *, pool, reranker, llm, parsed_by_id, top_k: (
        CrossEncoderRerankStage(reranker=reranker, top_k=top_k)
    ),
    "attest": lambda *, pool, reranker, llm, parsed_by_id, top_k: (
        EvidenceAttestationStage(parsed_by_id=parsed_by_id)
    ),
    "llm": lambda *, pool, reranker, llm, parsed_by_id, top_k: (
        LLMValidatorStage(llm=llm)
    ),
}

CASCADE_STAGE_NAMES = tuple(_STAGE_BUILDERS)
DEFAULT_CASCADE_STAGES = ["filter", "rerank", "attest", "llm"]


def build_cascade(
    stage_names: list[str],
    *,
    pool=None,
    reranker=None,
    llm=None,
    parsed_by_id: dict | None = None,
    top_k: int | None = None,
) -> PrecisionCascade:
    """Map `["filter","rerank","attest","llm"]` -> a `PrecisionCascade` whose
    stages are those Stage instances IN THAT ORDER (the rung selector). `[]`
    yields a cascade with NO stages (bypass). An unknown name fails fast with a
    ValueError (a typo in a config must not silently drop a stage)."""
    stages: list[Stage] = []
    for name in stage_names:
        builder = _STAGE_BUILDERS.get(name)
        if builder is None:
            raise ValueError(
                f"unknown cascade stage {name!r}; valid: {sorted(_STAGE_BUILDERS)}"
            )
        stages.append(builder(
            pool=pool, reranker=reranker, llm=llm,
            parsed_by_id=parsed_by_id or {}, top_k=top_k,
        ))
    return PrecisionCascade(stages=stages)


# --------------------------------------------------------------------------- #
# SELECTION POLICY (OCP): how a CascadeResult becomes the pre-cap paper_id list.
# A config's `selection` picks ONE policy; adding a policy is a one-line registry
# entry (the runner + cascade are untouched). Signature: (res, has_gate) -> ids.
#
#   "gate"  (default, UNCHANGED): the accepted set (accept-gate cascade) or the
#           final survivors (no accept-gate) -- an HONEST precision result that is
#           empty when the gate rejects everything. This is exactly today's logic.
#   "rank"  (never-empty): accepted papers FIRST (ordered by descending route
#           confidence -- `res.paper_ids` is already in that order), then the
#           remaining `survivor_ids` (filter/rerank order) APPENDED, so the list
#           is never empty even when the gate accepted nothing. Order-preserving
#           dedup. The full ablation showed gate-only craters paperF1 to ~0.09 by
#           emitting [] when the gate over-rejects; "rank" ranks instead of gates.
# --------------------------------------------------------------------------- #
def _gate_selection(res: CascadeResult, has_gate: bool) -> list[str]:
    return list(res.paper_ids if has_gate else res.survivor_ids)


def _rank_selection(res: CascadeResult, has_gate: bool) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    # `res.paper_ids` = accepted, already ordered by descending route confidence
    # (the cascade sorts by `confidences`); then the residual survivors.
    for pid in list(res.paper_ids) + list(res.survivor_ids):
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


_SELECTION_POLICIES = {"gate": _gate_selection, "rank": _rank_selection}

#   "property_first"  (candidate-level, BYPASSES the cascade): order the RAW
#           `run_strategies` candidate list by provenance priority -- property-
#           provenance candidates FIRST (in their original merge order among
#           themselves), then every non-property candidate in merge order;
#           order-preserving dedup. This is NOT a CascadeResult->ids policy (it
#           never runs the cascade), so it lives OUTSIDE `_SELECTION_POLICIES`;
#           it delegates VERBATIM to `provenance_rerank.property_first`, the exact
#           offline ordering that scored 0.531 paper-F1 (beating the old 0.49).
PROPERTY_FIRST_SELECTION = "property_first"
TARGET_COVERAGE_SELECTION = "target_coverage"
ANSWER_BEARING_SELECTION = "answer_bearing"
SELECTION_POLICY_NAMES = tuple(_SELECTION_POLICIES) + (
    PROPERTY_FIRST_SELECTION, TARGET_COVERAGE_SELECTION,
    ANSWER_BEARING_SELECTION,
)
DEFAULT_SELECTION = "gate"


def _property_first_order(candidates) -> list[str]:
    """Order the RAW router candidates property-provenance-first, reproducing the
    offline `provenance_rerank.property_first` BYTE-FOR-BYTE (same tie-break, same
    order-preserving dedup). Reuses that function verbatim -- no re-implementation.

    Imported LAZILY: `provenance_rerank` transitively pulls the offline embedding/
    scorer stack (numpy), so a module-level import would couple `NewPathRunner`'s
    importability to that stack. Only a property_first run pays the import cost."""
    from littraceqa.experiments.provenance_rerank import property_first

    return property_first(
        [{"paper_id": c.paper_id, "provenance": list(c.provenance)} for c in candidates]
    )


def _dedup_paper_ids(candidates) -> list[str]:
    """Order-preserving dedup of candidate paper_ids (the bypass/new-raw path)."""
    out: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        pid = cand.paper_id
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _trace_plan(plan) -> dict[str, Any]:
    """Serialize the planner decision without coupling traces to one Plan version."""
    targets: list[dict[str, Any]] = []
    for target in getattr(plan, "targets", ()):
        if isinstance(target, dict):
            targets.append(dict(target))
        else:
            targets.append({
                "key": getattr(target, "key", None),
                "text": getattr(target, "text", None),
                "role": getattr(target, "role", None),
            })
    return {
        "criterion": dict(getattr(plan, "criterion", {}) or {}),
        "named_methods": list(getattr(plan, "named_methods", ()) or ()),
        "strategies": list(getattr(plan, "strategies", ()) or ()),
        "multiplicity": getattr(plan, "multiplicity", None),
        "targets": targets,
        "desired_paper_count": getattr(plan, "desired_paper_count", None),
    }


def build_dense_retriever(pool):
    """Build a `DenseRetriever` over the cached title+abstract embeddings, or
    DEGRADE to None on any build failure (embedding cache/model unavailable) so a
    run never crashes -- the "knn" strategy then cleanly no-ops.

    SHARED by both drivers (`run_dev` fresh-per-config, `run_ladder` cached
    load-once) so the degrade path can never drift between them.
    """
    try:
        from littraceqa.retrieval.dense import DenseRetriever
        from littraceqa.retrieval.embedder_local import LocalEmbedder

        return DenseRetriever(pool, LocalEmbedder())
    except Exception as exc:  # noqa: BLE001 -- degrade cleanly, never crash the run
        import logging

        logging.getLogger(__name__).warning(
            "dense retriever unavailable (%r); knn disabled", exc
        )
        return None


def _select_table_strategy_name(params: dict) -> str:
    """Map the `table_strategy` config value to the registered answer-strategy
    name. Absent or `"legacy"` -> `"table"`; `"planned"` selects the ordinary
    planner; `"planned_native"` selects its internal null-fill composition;
    `"planned_visual_fill"` selects the fixed three-read visual fill-only
    composition. An UNKNOWN value RAISES (correction #5): silently falling
    back to legacy on a typo could burn a limited submission on the wrong path.
    """
    value = params.get("table_strategy")
    if value in (None, "legacy"):
        return "table"
    if value == "planned":
        return "table_planned"
    if value == "planned_native":
        return "table_planned_native"
    if value == "planned_visual_fill":
        return "table_planned_visual_fill"
    raise ValueError(
        f"unknown table_strategy {value!r}; expected 'legacy', 'planned', "
        "'planned_native', or 'planned_visual_fill'"
    )


def _build_evidence_localizer(params: dict, indexes, llm):
    """Build the configured evidence localizer; default remains whole-paper LLM."""
    from littraceqa.localize.indexed import (
        HybridEvidenceLocalizer,
        IndexEvidenceLocalizer,
        UnionEvidenceLocalizer,
    )
    from littraceqa.localize.localizer import (
        DEFAULT_MAX_CHARS_PER_PAGE,
        LlmEvidenceLocalizer,
    )

    name = params.get("evidence_localizer", "llm")
    llm_localizer = LlmEvidenceLocalizer(
        llm,
        max_chars_per_page=params.get(
            "evidence_max_chars_per_page", DEFAULT_MAX_CHARS_PER_PAGE
        ),
        cohort_coverage=params.get("evidence_cohort_coverage", False),
    )
    if name == "llm":
        localizer = llm_localizer
    elif name == "focused_llm":
        from littraceqa.localize.focused import (
            DEFAULT_SHORTLIST_K,
            DEFAULT_SHORTLIST_MAX_PAGES,
            FocusedEvidenceLocalizer,
        )

        localizer = FocusedEvidenceLocalizer(
            llm_localizer,
            top_k=params.get("evidence_shortlist_k", DEFAULT_SHORTLIST_K),
            max_pages=params.get(
                "evidence_shortlist_max_pages", DEFAULT_SHORTLIST_MAX_PAGES
            ),
        )
    else:
        indexed = IndexEvidenceLocalizer(
            getattr(indexes, "passages", None),
            getattr(indexes, "objects", None),
            passage_k=params.get("evidence_passage_k", 3),
            object_k=params.get("evidence_object_k", 3),
        )
        if name == "index":
            localizer = indexed
        elif name == "hybrid":
            localizer = HybridEvidenceLocalizer(indexed, llm_localizer)
        elif name == "union":
            localizer = UnionEvidenceLocalizer(indexed, llm_localizer)
        else:
            raise ValueError(
                f"unknown evidence_localizer {name!r}; "
                "expected 'llm', 'focused_llm', 'index', 'hybrid', or 'union'"
            )

    if params.get("evidence_target_query", False):
        from littraceqa.localize.targeted import (
            DEFAULT_MAX_QUERY_CHARS,
            DEFAULT_UNRELATED_CONFIDENCE,
            TargetAwareEvidenceLocalizer,
        )

        localizer = TargetAwareEvidenceLocalizer(
            localizer,
            llm,
            max_query_chars=params.get(
                "evidence_target_query_max_chars", DEFAULT_MAX_QUERY_CHARS
            ),
            unrelated_confidence=params.get(
                "evidence_target_unrelated_confidence",
                DEFAULT_UNRELATED_CONFIDENCE,
            ),
        )

    if params.get("evidence_contract_normalization", False):
        from littraceqa.localize.evidence_contract import EvidenceContractLocalizer

        localizer = EvidenceContractLocalizer(localizer)
    return localizer


def assemble_new_runner(
    config: RunConfig,
    *,
    indexes,
    pool,
    llm,
    evidence_llm=None,
    vision_llm=None,
    reranker=None,
    dense=None,
) -> "NewPathRunner":
    """SINGLE SOURCE OF TRUTH for how one `path:"new"` config maps to a runner.

    Takes the ALREADY-RESOLVED shared resources (`indexes`/`pool`/`llm`/
    optional role-specific `evidence_llm`/`vision_llm`/`reranker`/`dense`) and
    does the pure, cheap per-config
    assembly: read `params.cascade_stages`/`selection`/`use_dense`/
    `evidence_confidence_floor`/corpus source, build the `CorpusPdfFetcher` +
    `EvidenceLocalizationService` + `AnswerPipeline` + cascade rung + planner,
    and thread `selection` and `dense` correctly into `NewPathRunner`.

    Both `run_dev._make_new_runner` (builds fresh resources) and
    `run_ladder.build_new_runner` (reuses load-once caches) delegate the
    assembly HERE, so the ladder can never drift from run_dev again -- adding a
    future `params.X` to the wiring is a ONE-place change.

    Dense/kNN is wired ONLY when `params.use_dense` (default True) is on; a
    disabled `use_dense` OR an unavailable retriever (`dense is None`) => the
    runner gets `dense=None` and the "knn" strategy cleanly no-ops.
    """
    from littraceqa.answer.pipeline import AnswerPipeline
    from littraceqa.localize.fetch_corpus import CorpusPdfFetcher
    from littraceqa.localize.parse_pymupdf import PyMuPdfParser
    from littraceqa.localize.service import EvidenceLocalizationService
    from littraceqa.pipeline.planner import QuestionPlanner

    params = config.params

    # kNN semantic recall is wired ONLY when use_dense is on (default). A
    # disabled use_dense forces dense=None even if a retriever was resolved.
    if not params.get("use_dense", True):
        dense = None

    top_k = params.get("rerank_top_k")

    # Corpus PDF source: local dir (tests / synced disk) OR GCS bucket (VM). No
    # live fetch -- every PDF is already in the snapshot.
    corpus_pdf_dir = params.get("corpus_pdf_dir")
    corpus_gcs_bucket = params.get("corpus_gcs_bucket")
    corpus_pdf_prefix = params.get("corpus_pdf_prefix", "pdfs")
    if corpus_pdf_dir:
        fetcher = CorpusPdfFetcher(pdf_dir=corpus_pdf_dir)
    elif corpus_gcs_bucket:
        from littraceqa.corpus.gcs_backend import GcsBackend

        backend = GcsBackend(corpus_gcs_bucket, prefix=corpus_pdf_prefix)
        fetcher = CorpusPdfFetcher(backend=backend, prefix=corpus_pdf_prefix)
    else:
        raise ValueError(
            "path='new' requires params.corpus_pdf_dir or params.corpus_gcs_bucket"
        )

    evidence_service = EvidenceLocalizationService(
        fetcher,
        PyMuPdfParser(),
        _build_evidence_localizer(
            params, indexes, evidence_llm if evidence_llm is not None else llm
        ),
    )
    # Table strategy selection (correction #5): the override MUST preserve
    # `evidence_confidence_floor` and keep freeform/MC on their defaults --
    # only the "table" slot swaps to the planned answerer.
    floor = params.get("evidence_confidence_floor", 0.5)
    table_name = _select_table_strategy_name(params)  # raises on an unknown value
    if table_name == "table":
        fact_params = sorted(
            key for key in params if key.startswith("table_fact_")
        )
        if fact_params:
            raise ValueError(
                "table fact parameters require table_strategy: planned; "
                f"got {fact_params}"
            )
        answer_pipeline = AnswerPipeline(evidence_confidence_floor=floor)
    else:
        if table_name == "table_planned_native":
            import littraceqa.answer.table_planned_native  # noqa: F401
        elif table_name == "table_planned_visual_fill":
            import littraceqa.answer.table_planned_visual_fill  # noqa: F401
        else:
            import littraceqa.answer.table_planned  # noqa: F401
        from littraceqa.answer.interfaces import build_strategy
        common_table_kwargs = {
            "retain_concise_missing_expected_rows": params.get(
                "table_retain_concise_missing_expected_rows", False
            ),
            "route_expected_rows_to_papers": params.get(
                "table_route_expected_rows_to_papers", False
            ),
            "open_ended_one_row_per_paper": params.get(
                "table_open_ended_one_row_per_paper", False
            ),
            "prefer_owned_cells": params.get(
                "table_prefer_owned_cells", False
            ),
            "retain_source_attested_expected_rows": params.get(
                "table_retain_source_attested_expected_rows", False
            ),
            "visual_extraction_mode": params.get(
                "table_visual_extraction_mode", "direct"
            ),
            "visual_retry_owned_rows": params.get(
                "table_visual_retry_owned_rows", False
            ),
            "visual_consensus_repeats": params.get(
                "table_visual_consensus_repeats", 1
            ),
            "extraction_sources": params.get(
                "table_extraction_sources", "both"
            ),
            "text_context_mode": params.get(
                "table_text_context_mode", "evidence"
            ),
            "text_page_k": params.get("table_text_page_k", 3),
            "fill_nulls_from_scalar_evidence": params.get(
                "table_fill_nulls_from_scalar_evidence", False
            ),
            "trace_reassembly_inputs": params.get(
                "table_trace_reassembly_inputs", False
            ),
        }
        if table_name == "table_planned_native":
            forbidden = sorted(
                key
                for key in params
                if (
                    key.startswith("table_fact_")
                    and key not in {
                        "table_fact_max_pages",
                        "table_fact_empty_vision_retries",
                    }
                )
                or key.startswith("table_native_verify_")
                or key in {
                    "table_review_frozen_deltas",
                    "table_verify_frozen_source_cells",
                    "table_contract_collapse_hedges",
                }
            )
            if forbidden:
                raise ValueError(
                    "planned_native uses a fixed internal null-fill policy; "
                    f"unsupported parameters: {forbidden}"
                )
        if table_name == "table_planned_visual_fill":
            forbidden = sorted(
                key
                for key in params
                if key.startswith("table_fact_")
                or key.startswith("table_native_verify_")
                or key in {
                    "table_review_frozen_deltas",
                    "table_verify_frozen_source_cells",
                    "table_contract_collapse_hedges",
                    "table_visual_consensus_repeats",
                }
            )
            if forbidden:
                raise ValueError(
                    "planned_visual_fill uses fixed one-read/three-read "
                    "fill-only policy; unsupported parameters: "
                    f"{forbidden}"
                )
        fact_producer = None
        fact_producer_name = params.get("table_fact_producer")
        if (
            table_name != "table_planned_native"
            and fact_producer_name is not None
        ):
            if (
                not isinstance(fact_producer_name, str)
                or not fact_producer_name.strip()
            ):
                raise ValueError("table_fact_producer must be a non-empty string")
            fact_producer_name = fact_producer_name.strip()
            from littraceqa.answer.table_fact_extract import (
                build_table_fact_producer,
            )

            fact_producer = build_table_fact_producer(
                fact_producer_name,
                max_pages=params.get("table_fact_max_pages", 2),
                empty_vision_retries=params.get(
                    "table_fact_empty_vision_retries", 0
                ),
            )
        strategies = {n: build_strategy(n) for n in ("freeform", "multiple_choice")}
        if table_name == "table_planned_native":
            strategies["table"] = build_strategy(
                table_name,
                **common_table_kwargs,
                native_max_pages=params.get("table_fact_max_pages", 2),
                native_empty_vision_retries=params.get(
                    "table_fact_empty_vision_retries", 0
                ),
            )
        elif table_name == "table_planned_visual_fill":
            visual_fill_kwargs = dict(common_table_kwargs)
            visual_fill_kwargs.pop("visual_consensus_repeats")
            strategies["table"] = build_strategy(
                table_name,
                **visual_fill_kwargs,
            )
        else:
            strategies["table"] = build_strategy(
                table_name,
                **common_table_kwargs,
                review_frozen_table_deltas=params.get(
                    "table_review_frozen_deltas", False
                ),
                verify_frozen_source_cells=params.get(
                    "table_verify_frozen_source_cells", False
                ),
                native_verify_max_pages=params.get(
                    "table_native_verify_max_pages", 2
                ),
                native_verify_add_rows=params.get(
                    "table_native_verify_add_rows", False
                ),
                native_verify_substitute_rows=params.get(
                    "table_native_verify_substitute_rows", False
                ),
                native_verify_cell_updates=params.get(
                    "table_native_verify_cell_updates", True
                ),
                native_verify_source_key_canonicalization=params.get(
                    "table_native_verify_source_key_canonicalization", False
                ),
                fact_producer=fact_producer,
                fact_allow_row_additions=params.get(
                    "table_fact_allow_row_additions", False
                ),
                fact_preserve_unmatched_frozen=params.get(
                    "table_fact_preserve_unmatched_frozen", True
                ),
                fact_allow_cell_replacements=params.get(
                    "table_fact_allow_cell_replacements", False
                ),
                fact_canonicalize_attestation_only_rows=params.get(
                    "table_fact_canonicalize_attestation_only_rows", False
                ),
                fact_cell_value_policy=params.get(
                    "table_fact_cell_value_policy", "source"
                ),
                contract_collapse_hedges=params.get(
                    "table_contract_collapse_hedges", False
                ),
            )
        answer_pipeline = AnswerPipeline(strategies=strategies,
                                         evidence_confidence_floor=floor)

    stage_names = params.get("cascade_stages", DEFAULT_CASCADE_STAGES)
    cascade = (
        build_cascade(stage_names, pool=pool, reranker=reranker, llm=llm,
                      parsed_by_id={}, top_k=top_k)
        if stage_names else None  # [] => bypass (rung "new-raw")
    )

    return NewPathRunner(
        planner=QuestionPlanner(
            llm,
            structured_targets=params.get("target_aware", False),
            max_attempts=params.get("planner_max_attempts", 1),
        ),
        indexes=indexes,
        pool=pool,
        evidence_service=evidence_service,
        answer_pipeline=answer_pipeline,
        llm=llm,
        cascade=cascade,
        dense=dense,
        vision_llm=vision_llm,
        selection=params.get("selection", DEFAULT_SELECTION),
        max_papers=params.get("max_papers"),
        select_only=params.get("select_only", False),
        property_search_k=params.get(
            "property_search_k", DEFAULT_PROPERTY_SEARCH_K
        ),
        # Opt-in answer-bearing object recall.  The production configs omit this
        # flag, so their strategy list and ordering remain byte-for-byte intact.
        use_object_route=params.get("use_object_route", False),
        table_container_selection=params.get(
            "table_container_selection", False
        ),
        # FIX 1: the SAME corpus fetcher feeds both localization and the vision
        # shared-PDF-source, so GCS-streamed bytes reach `render_evidence_png`.
        pdf_fetcher=fetcher,
        # FIX 2: independently-ranked localized evidence emission cap.
        evidence_emit_cap=params.get("evidence_emit_cap", DEFAULT_EVIDENCE_EMIT_CAP),
        evidence_rebalance_explicit_table=params.get(
            "evidence_rebalance_explicit_table", False
        ),
    )


class NewPathRunner:
    """Rung-2+ adapter: planner -> router -> cascade -> evidence -> answer.

    Deps are all injected Protocols (DIP): `planner.plan(record)`, `indexes`
    (a `LoadedIndexes`/`BuiltIndexes`), `pool` (`PoolIndex`), `evidence_service`
    (`EvidenceLocalizationService`), `answer_pipeline` (`AnswerPipeline`), the
    `llm` (threaded into the answer context), an optional `cascade`
    (`PrecisionCascade`; None => bypass/new-raw), an optional `dense` retriever
    (for a future knn strategy), and an optional `vision_llm`.
    """

    def __init__(self, *, planner, indexes, pool, evidence_service,
                 answer_pipeline, llm, cascade: PrecisionCascade | None = None,
                 dense=None, vision_llm=None, selection: str = DEFAULT_SELECTION,
                 max_papers: int | None = None, select_only: bool = False,
                 use_object_route: bool = False,
                 table_container_selection: bool = False,
                 property_search_k: int = DEFAULT_PROPERTY_SEARCH_K,
                 pdf_fetcher=None,
                 evidence_emit_cap: int | None = DEFAULT_EVIDENCE_EMIT_CAP,
                 evidence_rebalance_explicit_table: bool = False,
                 paper_overrides: Mapping[str, Mapping[str, Any]] | None = None):
        if selection not in SELECTION_POLICY_NAMES:
            raise ValueError(
                f"unknown selection policy {selection!r}; "
                f"valid: {sorted(SELECTION_POLICY_NAMES)}"
            )
        if (
            isinstance(property_search_k, bool)
            or not isinstance(property_search_k, int)
            or property_search_k < 1
        ):
            raise ValueError("property_search_k must be a positive integer")
        if not isinstance(table_container_selection, bool):
            raise ValueError("table_container_selection must be a boolean")
        self._planner = planner
        self._indexes = indexes
        self._pool = pool
        self._evidence_service = evidence_service
        self._answer_pipeline = answer_pipeline
        self._llm = llm
        self._cascade = cascade
        self._dense = dense
        self._vision_llm = vision_llm
        # Selection policy over the CascadeResult (default "gate" == today).
        # "property_first" instead bypasses the cascade and orders the RAW
        # router candidates via `provenance_rerank.property_first`.
        self._selection = selection
        # Optional hard cap on the multi-multiplicity list (None => keep all).
        self._max_papers = max_papers
        # SELECT-ONLY mode: stop after paper_ids -- skip evidence localization AND
        # answer generation. ~1 planner LLM call/question
        # (no PDF fetch, no vision, no answer) for a cheap paper-selection run.
        self._select_only = select_only
        # Opt-in retrieval over table/figure captions. It is activated only for
        # table-answer or explicitly object-grounded questions; the default False
        # preserves every existing production plan and selector input.
        self._use_object_route = bool(use_object_route)
        self._table_container_selection = table_container_selection
        # Opt-in depth probe for lexical property/target-property retrieval.
        # The default is the historical 30 chunks, preserving production.
        self._property_search_k = property_search_k
        # SHARED PDF SOURCE (FIX 1): the SAME fetcher the evidence service reads
        # from. The runner re-reads each selected paper's bytes into an in-memory
        # `pdf_bytes_by_id` map on the AnswerContext so the vision path renders
        # from streamed GCS bytes (the on-disk cache the streaming path never
        # fills). None => vision uses the on-disk cache only (old local path).
        self._pdf_fetcher = pdf_fetcher
        # EVIDENCE INDEPENDENCE (FIX 2): max localized-evidence items to emit.
        self._evidence_emit_cap = evidence_emit_cap
        self._evidence_rebalance_explicit_table = bool(
            evidence_rebalance_explicit_table
        )
        # Optional gold-blind batch paper decisions.  The ordinary per-record
        # selector remains the traced baseline; an override is applied only
        # after that selector and its cardinality policy have completed.
        self._paper_overrides = {
            str(query_id): dict(metadata)
            for query_id, metadata in (paper_overrides or {}).items()
        }

    def selection_pass_runner(self):
        """Return a shallow dependency-sharing clone that stops after papers."""
        clone = copy.copy(self)
        clone._select_only = True
        clone._paper_overrides = {}
        return clone

    def with_paper_overrides(
        self, overrides: Mapping[str, Mapping[str, Any]]
    ):
        """Return an immutable-style clone with audited per-query overrides."""
        clone = copy.copy(self)
        clone._paper_overrides = {
            str(query_id): dict(metadata)
            for query_id, metadata in overrides.items()
        }
        return clone

    def run_one(self, record: InputRecord) -> RunOutcome:
        start = time.perf_counter()
        try:
            return self._run(record, start)
        except Exception as exc:  # noqa: BLE001 -- never drop a query_id, never crash
            line = build_fallback_line(record)
            trace = make_trace(
                record.query_id, "new", time.perf_counter() - start,
                failure_reason=f"{type(exc).__name__}: {exc}",
                candidates=[], cost=0.0,
                extra={"n_candidates": 0, "cascade": None, "paper_ids": [],
                       "selection": self._selection,
                       "select_only": self._select_only, "ranked_paper_ids": [],
                       "candidate_order_before": [],
                       "candidate_order_after": [],
                       "localized_evidence_candidates": [],
                       "emitted_evidence": [], "attested_evidence": []},
            )
            return RunOutcome(line=line, trace=trace)

    def _run(self, record: InputRecord, start: float) -> RunOutcome:
        plan = self._planner.plan(record)

        # SEMANTIC-RECALL GUARANTEE: whenever a dense retriever is wired, the "knn"
        # route MUST run -- even if the planner (LLM) omitted it. The planner already
        # keeps "knn" as a fallback, but making dense PRESENCE alone sufficient here
        # means a stray/degraded plan can never silently drop semantic search. With
        # NO dense, the registered knn strategy is a no-op ([]), so appending it is
        # harmless when dense is off -- but we only append when dense is present.
        if self._dense is not None and "knn" not in plan.strategies:
            plan = replace(plan, strategies=[*plan.strategies, "knn"])

        # ANSWER-BEARING OBJECT RECALL (opt-in instrumentation): a correct table
        # answer may live in a paper that reports a requested method/value but did
        # not originate the method. In this first gate, run the object route into
        # a SEPARATE candidate list. It is traced but never passed to selection,
        # so candidate recall can be measured without changing emitted papers.
        required_source_type = (plan.criterion or {}).get("required_source_type")
        object_route_active = self._use_object_route and (
            "table" in record.answer_types
            or required_source_type in {"table", "figure"}
        )

        ctx = RetrievalContext.from_indexes(
            self._indexes, dense=self._dense, pool=self._pool,
            question=record.question,
            property_search_k=self._property_search_k,
        )
        cands = run_strategies(plan, ctx, question=record.question)
        object_cands = (
            run_strategies(
                replace(plan, strategies=["object"]),
                ctx,
                question=record.question,
            )
            if object_route_active else []
        )
        # AUDIT #1: deterministic venue/year hard-constraint filter, applied to the
        # raw candidates BEFORE any ranking policy (property_first bypasses the
        # cascade's DeterministicFilterStage, so the constraint would otherwise
        # never be enforced). Recall-safe: never empties the set.
        _n_before_filter = len(cands)
        cands = filter_by_venue_year(cands, record.question)
        venue_year_dropped = _n_before_filter - len(cands)
        _n_objects_before_filter = len(object_cands)
        object_cands = filter_by_venue_year(object_cands, record.question)
        object_venue_year_dropped = _n_objects_before_filter - len(object_cands)

        # Cascade -> precise paper_ids; bypass (None) -> deduped candidate ids.
        # Ablation semantics: with an accept-gate (attest/llm) the output is the
        # ACCEPTED set (a precision gate -- honestly [] when nothing is accepted,
        # no survivor fallback); with NO accept-gate (filter/rerank-only) it is
        # the final SURVIVORS (else `accepted` is empty and the rung yields []).
        # The SELECTION POLICY turns the CascadeResult into the pre-cap paper list.
        # "gate" (default) is today's behavior; "rank" is never-empty (accepted by
        # confidence, then survivors appended). Bypass (no cascade) is unaffected.
        output_mode: str | None = None
        # PROPERTY-FIRST records the candidate order BEFORE and AFTER the reorder.
        candidate_order_before: list[str] | None = None
        candidate_order_after: list[str] | None = None
        target_selection_trace: dict | None = None
        answer_bearing_trace: dict | None = None
        if self._selection in {
            TARGET_COVERAGE_SELECTION, ANSWER_BEARING_SELECTION,
        }:
            candidate_order_before = _dedup_paper_ids(cands)
            if self._selection == ANSWER_BEARING_SELECTION:
                answer_result = select_answer_bearing(
                    cands, plan, question=record.question
                )
                target_result = answer_result.baseline
                paper_ids = answer_result.paper_ids
                answer_bearing_trace = {
                    "reason": answer_result.reason,
                    "rescue_paper_id": answer_result.rescue_paper_id,
                    "replaced_paper_id": answer_result.replaced_paper_id,
                    "required_constraint_matches": (
                        answer_result.required_constraint_matches
                    ),
                    # Paired baseline from the same plan and candidate objects.
                    "baseline_paper_ids": list(target_result.paper_ids),
                    "coverage": [
                        {
                            "paper_id": item.paper_id,
                            "matched_constraints": list(
                                item.matched_constraints
                            ),
                            "question_token_hits": item.question_token_hits,
                            "numeric_token_count": item.numeric_token_count,
                            "property_rank": item.property_rank,
                            "candidate_position": item.candidate_position,
                        }
                        for item in answer_result.coverage[:10]
                    ],
                }
            else:
                target_result = select_target_coverage(cands, plan)
                paper_ids = target_result.paper_ids
            candidate_order_after = list(paper_ids)
            cascade_trace = None
            output_mode = self._selection
            target_selection_trace = {
                "assignments": [
                    {
                        "target_key": item.target_key,
                        "paper_id": item.paper_id,
                        "group_rank": item.group_rank,
                        "target_property_rank": item.target_property_rank,
                        "fusion_score": item.fusion_score,
                        "corroboration": list(item.corroboration),
                    }
                    for item in target_result.assignments
                ],
                "uncovered_targets": list(target_result.uncovered_targets),
                "uncorroborated_target_counts": dict(
                    target_result.uncorroborated_target_counts
                ),
                "excluded_non_target_anchors": list(
                    target_result.excluded_non_target_anchors
                ),
                # Same candidates, same plan, same deterministic floor. This is
                # the apples-to-apples baseline for the target selector; comparing
                # with a separately planned run confounds planner nondeterminism.
                "floor_paper_ids": list(target_result.floor_paper_ids),
            }
        elif self._selection == PROPERTY_FIRST_SELECTION:
            # BYPASS the cascade entirely: order the RAW `run_strategies`
            # candidates via the offline `provenance_rerank.property_first`
            # (property-provenance first in merge order, then the rest in merge
            # order; order-preserving dedup) -- the byte-identical reproduction of
            # the offline 0.531 paper-F1 ordering. The multiplicity cap below then
            # applies UNCHANGED (single -> the first property candidate; multi ->
            # all, honoring any `max_papers`).
            candidate_order_before = _dedup_paper_ids(cands)
            paper_ids = _property_first_order(cands)
            candidate_order_after = list(paper_ids)
            cascade_trace: dict | None = None
            output_mode = "property_first"
        elif self._cascade is not None:
            res = self._cascade.select(record.question, cands, criterion=plan.criterion)
            has_gate = self._cascade.has_accept_gate
            paper_ids = _SELECTION_POLICIES[self._selection](res, has_gate)
            output_mode = "accepted" if has_gate else "survivors"
            cascade_trace = {
                "reasons": res.reasons, "confidences": res.confidences,
            }
        else:
            paper_ids = _dedup_paper_ids(cands)
            cascade_trace = None

        table_container_trace: dict[str, Any] = {
            "enabled": self._table_container_selection,
            "applied": False,
            "reason": "disabled",
        }
        table_container_override = False
        if self._table_container_selection:
            table_container_baseline_ids = list(paper_ids)
            container = select_table_container(
                cands,
                plan,
                question=record.question,
                answer_types=record.answer_types,
                table_schema=record.table_schema,
                pool=self._pool,
                baseline_paper_ids=paper_ids,
            )
            table_container_trace = {
                "enabled": True,
                **container.to_trace(),
                "baseline_paper_ids": table_container_baseline_ids,
                "removed_paper_ids": (
                    [
                        paper_id for paper_id in table_container_baseline_ids
                        if paper_id != container.selected_paper_id
                    ]
                    if container.applied else []
                ),
            }
            if container.applied:
                paper_ids = [container.selected_paper_id]
                table_container_override = True

        # Pre-cap ranked list (recorded for analysis before multiplicity trimming).
        ranked_paper_ids = list(paper_ids)

        # Cardinality is independent of ranking policy.  An explicit count in
        # the question is strongest; otherwise honor a structured planner count
        # for EVERY selector (not only target_coverage).  max_papers remains an
        # operational safety ceiling, never a replacement for the requested
        # count.  When neither source has a count, preserve the legacy
        # single-vs-multi behavior exactly.
        explicit_count = explicit_target_paper_count(record.question)
        planner_count = getattr(plan, "desired_paper_count", None)
        requested_count = (
            explicit_count if explicit_count is not None else planner_count
        )
        if table_container_override:
            paper_ids = paper_ids[:1]
            count_source = "table_container_selection"
        elif requested_count is not None:
            target_cap = requested_count
            if self._max_papers is not None:
                target_cap = min(target_cap, self._max_papers)
            paper_ids = paper_ids[:target_cap]
            count_source = (
                "question_explicit" if explicit_count is not None else "planner"
            )
        elif plan.multiplicity == "single":
            paper_ids = paper_ids[:1]
            count_source = "multiplicity_single"
        elif self._max_papers is not None:
            paper_ids = paper_ids[: self._max_papers]
            count_source = "max_papers"
        else:
            count_source = "uncapped_multi"

        batch_selection_trace: dict[str, Any] | None = None
        batch_override = self._paper_overrides.get(record.query_id)
        if batch_override is not None:
            override_ids = [
                str(paper_id).strip()
                for paper_id in (batch_override.get("paper_ids") or [])
                if str(paper_id).strip()
            ]
            override_ids = list(dict.fromkeys(override_ids))
            if not override_ids:
                raise ValueError(
                    f"empty batch paper override for {record.query_id}"
                )
            if self._max_papers is not None and len(override_ids) > self._max_papers:
                raise ValueError(
                    f"batch paper override for {record.query_id} exceeds "
                    f"max_papers={self._max_papers}"
                )
            original_paper_ids = list(paper_ids)
            paper_ids = override_ids
            count_source = str(batch_override.get("mode") or "batch_override")
            batch_selection_trace = {
                **dict(batch_override),
                "original_paper_ids": original_paper_ids,
                "override_paper_ids": list(paper_ids),
            }

        paper_count_policy = {
            "explicit_count": explicit_count,
            "planner_count": planner_count,
            "requested_count": requested_count,
            "max_papers": self._max_papers,
            "source": count_source,
            "emitted_count": len(paper_ids),
        }

        if target_selection_trace is not None:
            # Record the floor under the EXACT effective cardinality used above
            # (including an explicit/planner requested count). This prevents a
            # target@4 result from being compared to a property-first@5 replay.
            target_selection_trace["effective_count"] = len(paper_ids)
            target_selection_trace["floor_paper_ids_at_cap"] = (
                target_selection_trace["floor_paper_ids"][:len(paper_ids)]
            )
        if answer_bearing_trace is not None:
            answer_bearing_trace["baseline_paper_ids_at_cap"] = (
                answer_bearing_trace["baseline_paper_ids"][:len(paper_ids)]
            )

        # ---- SELECT-ONLY short-circuit: stop after paper_ids. NO evidence
        # localization (no PDF fetch/parse), NO answer generation -- the
        # evidence service and answer pipeline are left entirely untouched, so a
        # select-only run costs ~1 planner LLM call/question. Emit empty
        # evidence and a contract placeholder answer for the paper-selection check.
        # RAW localized evidence, INDEPENDENT of what the answer strategy attests
        # (FIX 2): what the localizer found, ranked + capped + emitted directly.
        localized_evidence: list[LocatedEvidence] = []
        emitted_evidence: list[dict] = []
        attested_evidence: list[dict] = []
        if self._select_only:
            # Keep the diagnostic selection artifact contract-valid without
            # invoking an answer strategy.  An empty answer would be repaired
            # by the shared driver and falsely reported as a failed selection.
            answer = build_fallback_line(record)["answer"]
            component_confidences: dict[str, float] = {}
            component_diagnostics: dict[str, dict] = {}
        else:
            # ---- Evidence + answer: reused EXACTLY as Pipeline._attempt (corpus-read).
            parsed_by_id: dict[str, ParsedPdf] = {}
            pdf_bytes_by_id: dict[str, bytes] = {}
            for pid in paper_ids:
                paper = self._pool.by_id(pid)
                if paper is None:
                    continue
                try:
                    located = self._evidence_service.locate_located(record.question, paper)
                except Exception:  # noqa: BLE001 -- one bad paper's evidence degrades to []
                    located = []
                if located:
                    localized_evidence.extend(located)
                try:
                    parsed = self._evidence_service.parsed_for(paper)
                except Exception:  # noqa: BLE001 -- a raising parser degrades this paper only
                    parsed = None
                if parsed is not None:
                    parsed_by_id[paper.paper_id] = parsed
                # SHARED PDF SOURCE (FIX 1): retain this paper's bytes so vision
                # renders from them on the GCS/streaming path. Degrades to no-op
                # when no fetcher is wired or the fetch misses.
                raw_for = getattr(self._evidence_service, "raw_for", None)
                if callable(raw_for):
                    try:
                        raw = raw_for(paper)
                    except Exception:  # noqa: BLE001 -- a cache/fetch error is a miss
                        raw = None
                elif self._pdf_fetcher is not None:
                    # Compatibility for injected evidence-service doubles that
                    # predate ``raw_for``. The production service always reuses
                    # its cache and therefore never takes this branch.
                    try:
                        raw = self._pdf_fetcher.fetch(
                            pid, getattr(paper, "pdf_url", None)
                        )
                    except Exception:  # noqa: BLE001 -- a fetch error is a miss
                        raw = None
                else:
                    raw = None
                if raw:
                    pdf_bytes_by_id[pid] = raw

            paper_titles = {
                pid: p.title for pid in paper_ids if (p := self._pool.by_id(pid))
            }

            answer_ctx = AnswerContext(
                record.question, record.answer_types, paper_ids, localized_evidence,
                parsed_by_id, paper_titles, self._llm, mc_options=record.mc_options,
                table_schema=record.table_schema, vision_llm=self._vision_llm,
                pdf_bytes_by_id=pdf_bytes_by_id,
            )
            result = self._answer_pipeline.answer(answer_ctx)
            attested_evidence = result.attested_evidence
            answer = result.answer
            component_confidences = dict(result.component_confidences)
            component_diagnostics = dict(
                getattr(result, "component_diagnostics", {}) or {}
            )

            # EMIT the raw localized evidence, ranked by the calibrated ranker and
            # capped -- NOT gated on the answer strategy's attestation/confidence.
            emitted_evidence = build_emitted_evidence(
                record.question,
                localized_evidence,
                cap=self._evidence_emit_cap,
                rebalance_explicit_table=(
                    self._evidence_rebalance_explicit_table
                ),
            )

        fallback_types = [] if self._select_only else answer_fallback_types(
            record, answer, component_confidences
        )
        evidence_rebalance_triggered = (
            self._evidence_rebalance_explicit_table
            and explicit_table_rebalance_triggered(
                record.question,
                localized_evidence,
                cap=self._evidence_emit_cap,
            )
        )
        failure_reason = (
            f"answer fallback: {', '.join(fallback_types)}"
            if fallback_types else None
        )
        line = build_record_line(record, paper_ids, emitted_evidence, answer)

        trace = make_trace(
            record.query_id, "new", time.perf_counter() - start, cost=0.0,
            failure_reason=failure_reason,
            candidates=[
                {"paper_id": c.paper_id, "provenance": list(c.provenance)}
                for c in cands[:50]
            ],
            extra={
                "n_candidates": len(cands),
                "venue_year_dropped": venue_year_dropped,
                # AUDIT #6 (partial): per-route (rank, score) signals for the
                # ENTIRE candidate set -- UNTRUNCATED, unlike `candidates[:50]` --
                # so an offline route-aware ranking policy can be replayed against
                # the exact frozen snapshot. Ranks are the PRE-venue-filter
                # unique-paper ranks per route; the offline replay rebases ranks on
                # the post-filter survivors before RRF (see the replay script).
                "candidate_signals": [
                    {"paper_id": c.paper_id,
                     "route_signals": [
                         {"route": s.route, "rank": s.rank, "score": s.score,
                          "group_key": s.group_key, "role": s.role}
                         for s in c.route_signals]}
                    for c in cands
                ],
                # INSTRUMENTATION ONLY: compact metadata + support excerpts for
                # every post-filter candidate. Built after selection and never
                # consumed by ranking or submission construction.
                "candidate_dossiers": build_candidate_dossiers(
                    cands,
                    pool=self._pool,
                    ranked_paper_ids=ranked_paper_ids,
                    emitted_paper_ids=paper_ids,
                ),
                # Gate-only object candidates are intentionally separate from
                # ``cands`` and therefore cannot influence ranked/emitted papers.
                "n_object_candidates": len(object_cands),
                "object_venue_year_dropped": object_venue_year_dropped,
                "object_candidate_dossiers": build_candidate_dossiers(
                    object_cands,
                    pool=self._pool,
                    ranked_paper_ids=[],
                    emitted_paper_ids=[],
                ),
                "cascade": cascade_trace,
                "target_selection": target_selection_trace,
                "answer_bearing_selection": answer_bearing_trace,
                "paper_ids": list(paper_ids),
                "output_mode": output_mode,
                "selection": self._selection,
                "select_only": self._select_only,
                "object_route_enabled": self._use_object_route,
                "object_route_active": object_route_active,
                "table_container_selection": table_container_trace,
                "property_search_k": self._property_search_k,
                "plan": _trace_plan(plan),
                "ranked_paper_ids": ranked_paper_ids,
                "paper_count_policy": paper_count_policy,
                "batch_selection": batch_selection_trace,
                "candidate_order_before": candidate_order_before,
                "candidate_order_after": candidate_order_after,
                "answer_fallback_types": fallback_types,
                "answer_component_confidences": component_confidences,
                "answer_component_diagnostics": component_diagnostics,
                # EVIDENCE FORENSICS (FIX 2): the RAW localized candidates, what
                # was EMITTED (ranked+capped, independent of the answer), and what
                # the answer strategy ATTESTED -- so forensics can tell a localizer
                # miss from a ranking/emission drop (no more submitted==attested).
                "localized_evidence_candidates": [
                    {"paper_id": ev.paper_id, "source_type": ev.source_type,
                     "page": ev.page, "object_id": ev.object_id,
                     "quote": ev.quote,
                     "confidence": ev.confidence}
                    for ev in localized_evidence[:50]
                ],
                "evidence_rebalance_explicit_table_enabled": (
                    self._evidence_rebalance_explicit_table
                ),
                "evidence_rebalance_explicit_table_triggered": (
                    evidence_rebalance_triggered
                ),
                "emitted_evidence": emitted_evidence,
                "attested_evidence": attested_evidence,
            },
        )
        return RunOutcome(line=line, trace=trace)
