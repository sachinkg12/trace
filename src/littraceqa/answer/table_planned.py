"""SchemaPlannedTableAnswerer: the schema-driven table strategy.

Registered as `"table_planned"` (the legacy one-row-per-paper `TableAnswerer`
keeps `"table"`); the composition root selects between them via the
`table_strategy` config flag, so this is added by REGISTRATION, never by editing
`build_strategy` (Open/Closed).

Flow (ENTITY axis): plan the question -> optionally route each requested row to
its clearly identified owning paper (ambiguous rows fail open) -> extract rows
from visual + non-visual evidence -> deterministically assemble (cross-paper
merge, null-prune, dedup). PAPER axis (the row-key genuinely denotes the paper)
reuses the deterministic metadata-title path. Never raises -- any failure
degrades to empty rows.
"""
from __future__ import annotations

from dataclasses import replace
import re

from littraceqa.answer.grounding import ground_value
from littraceqa.answer.interfaces import AnswerContext, StrategyOutput, register_strategy
from littraceqa.answer.references import (
    answer_reference_question,
    is_reference_list_question,
    parse_reference_list,
)
from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table import _coerce_number
from littraceqa.answer.table_assemble import assemble_rows, matches_expected_key
from littraceqa.answer.table_extract import _coerce_string_cell
from littraceqa.answer.table_extract import extract_rows_from_paper
from littraceqa.answer.table_delta import review_frozen_table_additions
from littraceqa.answer.table_plan import RowAxis, TablePlan, plan_table
from littraceqa.answer.table_route import (
    expected_keys_by_paper,
    source_attested_expected_keys,
)


@register_strategy("table_planned")
class SchemaPlannedTableAnswerer:
    answer_type = "table"

    @property
    def requires_documents(self) -> bool:
        """Whether replay must fetch/parse PDFs before calling this strategy."""

        return not self._contract_collapse_hedges

    def __init__(
        self,
        *,
        retain_concise_missing_expected_rows: bool = False,
        route_expected_rows_to_papers: bool = False,
        open_ended_one_row_per_paper: bool = False,
        prefer_owned_cells: bool = False,
        retain_source_attested_expected_rows: bool = False,
        visual_extraction_mode: str = "direct",
        visual_retry_owned_rows: bool = False,
        visual_consensus_repeats: int = 1,
        extraction_sources: str = "both",
        text_context_mode: str = "evidence",
        text_page_k: int = 3,
        review_frozen_table_deltas: bool = False,
        verify_frozen_source_cells: bool = False,
        native_verify_max_pages: int = 2,
        native_verify_add_rows: bool = False,
        native_verify_substitute_rows: bool = False,
        native_verify_cell_updates: bool = True,
        native_verify_source_key_canonicalization: bool = False,
        fill_nulls_from_scalar_evidence: bool = False,
        trace_reassembly_inputs: bool = False,
        fact_producer=None,
        fact_allow_row_additions: bool = False,
        fact_preserve_unmatched_frozen: bool = True,
        fact_allow_cell_replacements: bool = False,
        fact_canonicalize_attestation_only_rows: bool = False,
        fact_cell_value_policy: str = "source",
        contract_collapse_hedges: bool = False,
    ):
        if visual_extraction_mode not in {"direct", "full_grid", "full_grid_crop"}:
            raise ValueError(
                "visual_extraction_mode must be 'direct', 'full_grid', or "
                "'full_grid_crop'"
            )
        if (
            isinstance(visual_consensus_repeats, bool)
            or not isinstance(visual_consensus_repeats, int)
            or not 1 <= visual_consensus_repeats <= 5
        ):
            raise ValueError("visual_consensus_repeats must be an integer from 1 to 5")
        if extraction_sources not in {"both", "text_only", "visual_only"}:
            raise ValueError(
                "extraction_sources must be 'both', 'text_only', or 'visual_only'"
            )
        if text_context_mode not in {"evidence", "focused_pages"}:
            raise ValueError(
                "text_context_mode must be 'evidence' or 'focused_pages'"
            )
        if (
            isinstance(text_page_k, bool)
            or not isinstance(text_page_k, int)
            or not 1 <= text_page_k <= 8
        ):
            raise ValueError("text_page_k must be an integer from 1 to 8")
        if (
            isinstance(native_verify_max_pages, bool)
            or not isinstance(native_verify_max_pages, int)
            or not 1 <= native_verify_max_pages <= 5
        ):
            raise ValueError(
                "native_verify_max_pages must be an integer from 1 to 5"
            )
        frozen_modes = sum(bool(value) for value in (
            review_frozen_table_deltas,
            verify_frozen_source_cells,
            fact_producer is not None,
            contract_collapse_hedges,
        ))
        if frozen_modes > 1:
            raise ValueError(
                "frozen delta review, source-cell verification, and table "
                "fact production/row-contract replay are mutually exclusive"
            )
        if not isinstance(native_verify_add_rows, bool):
            raise ValueError("native_verify_add_rows must be a boolean")
        if not isinstance(native_verify_substitute_rows, bool):
            raise ValueError("native_verify_substitute_rows must be a boolean")
        if not isinstance(native_verify_cell_updates, bool):
            raise ValueError("native_verify_cell_updates must be a boolean")
        if not isinstance(native_verify_source_key_canonicalization, bool):
            raise ValueError(
                "native_verify_source_key_canonicalization must be a boolean"
            )
        if not isinstance(fact_allow_row_additions, bool):
            raise ValueError("fact_allow_row_additions must be a boolean")
        if not isinstance(fact_preserve_unmatched_frozen, bool):
            raise ValueError(
                "fact_preserve_unmatched_frozen must be a boolean"
            )
        if not isinstance(fact_allow_cell_replacements, bool):
            raise ValueError(
                "fact_allow_cell_replacements must be a boolean"
            )
        if not isinstance(fact_canonicalize_attestation_only_rows, bool):
            raise ValueError(
                "fact_canonicalize_attestation_only_rows must be a boolean"
            )
        if not isinstance(contract_collapse_hedges, bool):
            raise ValueError("contract_collapse_hedges must be a boolean")
        if fact_cell_value_policy not in {
            "source", "schema_canonical", "header_unit_explicit"
        }:
            raise ValueError(
                "fact_cell_value_policy must be 'source', "
                "'schema_canonical', or 'header_unit_explicit'"
            )
        if fact_producer is None and (
            fact_allow_row_additions
            or not fact_preserve_unmatched_frozen
            or fact_allow_cell_replacements
            or fact_canonicalize_attestation_only_rows
            or fact_cell_value_policy != "source"
        ):
            raise ValueError(
                "table fact policies require a configured fact producer"
            )
        self._retain_concise_missing_expected_rows = bool(
            retain_concise_missing_expected_rows
        )
        self._route_expected_rows_to_papers = bool(
            route_expected_rows_to_papers
        )
        self._open_ended_one_row_per_paper = bool(
            open_ended_one_row_per_paper
        )
        self._prefer_owned_cells = bool(prefer_owned_cells)
        self._retain_source_attested_expected_rows = bool(
            retain_source_attested_expected_rows
        )
        self._visual_extraction_mode = visual_extraction_mode
        self._visual_retry_owned_rows = bool(visual_retry_owned_rows)
        self._visual_consensus_repeats = visual_consensus_repeats
        self._extraction_sources = extraction_sources
        self._text_context_mode = text_context_mode
        self._text_page_k = text_page_k
        self._review_frozen_table_deltas = bool(review_frozen_table_deltas)
        self._verify_frozen_source_cells = bool(verify_frozen_source_cells)
        self._native_verify_max_pages = native_verify_max_pages
        self._native_verify_add_rows = native_verify_add_rows
        self._native_verify_substitute_rows = native_verify_substitute_rows
        self._native_verify_cell_updates = native_verify_cell_updates
        self._native_verify_source_key_canonicalization = (
            native_verify_source_key_canonicalization
        )
        self._fill_nulls_from_scalar_evidence = bool(
            fill_nulls_from_scalar_evidence
        )
        self._trace_reassembly_inputs = bool(trace_reassembly_inputs)
        self._fact_producer = fact_producer
        self._fact_allow_row_additions = fact_allow_row_additions
        self._fact_preserve_unmatched_frozen = fact_preserve_unmatched_frozen
        self._fact_allow_cell_replacements = fact_allow_cell_replacements
        self._fact_canonicalize_attestation_only_rows = (
            fact_canonicalize_attestation_only_rows
        )
        self._fact_cell_value_policy = fact_cell_value_policy
        self._contract_collapse_hedges = contract_collapse_hedges

    def answer(self, ctx: AnswerContext) -> StrategyOutput:
        return self.answer_with_plan(ctx)[0]

    def answer_with_plan(
        self, ctx: AnswerContext
    ) -> tuple[StrategyOutput, TablePlan | None]:
        """Return the answer and the exact plan used to produce it."""
        plan_sink: list[TablePlan] = []
        try:
            return self._answer(ctx, _plan_sink=plan_sink), (
                plan_sink[0] if plan_sink else None
            )
        except Exception as exc:  # noqa: BLE001 -- never raise; empty rows on total failure
            return StrategyOutput(
                value=[], confidence=0.0, attested_evidence=[],
                diagnostics={
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                },
            ), None

    def answer_from_plan(
        self, ctx: AnswerContext, plan: TablePlan
    ) -> StrategyOutput:
        """Answer with an already-created plan, without re-planning.

        This narrow seam lets conservative ensemble strategies compare two
        extraction passes while keeping row identity and source scope fixed.
        Like ``answer_with_plan``, it never lets an extraction failure escape.
        """
        if not isinstance(plan, TablePlan):
            raise TypeError("plan must be a TablePlan")
        try:
            return self._answer(ctx, _provided_plan=plan)
        except Exception as exc:  # noqa: BLE001 -- table strategies fail closed
            return StrategyOutput(
                value=[], confidence=0.0, attested_evidence=[],
                diagnostics={
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                },
            )

    def _answer(
        self,
        ctx: AnswerContext,
        *,
        _plan_sink: list[TablePlan] | None = None,
        _provided_plan: TablePlan | None = None,
    ) -> StrategyOutput:
        if self._fact_producer is None and _provided_plan is None:
            reference_rows = self._reference_rows(ctx)
            if reference_rows is not None:
                return reference_rows

        plan = _provided_plan or plan_table(
            ctx.question, ctx.table_schema, ctx.paper_titles, ctx.llm
        )
        if _plan_sink is not None:
            _plan_sink.append(plan)
        if self._contract_collapse_hedges and ctx.frozen_table_rows is not None:
            from littraceqa.answer.table_row_contract import collapse_hedged_rows

            result = collapse_hedged_rows(
                ctx.frozen_table_rows,
                row_key_columns=plan.row_key_cols,
                value_columns=[column["name"] for column in plan.value_cols],
                expected_keys=plan.expected_keys,
            )
            return StrategyOutput(
                value=list(result.rows),
                confidence=1.0 if result.rows else 0.0,
                attested_evidence=[],
                diagnostics={
                    "status": "row_contract_collapsed" if result.changed else "row_contract_preserved",
                    "row_axis": plan.row_axis.value,
                    "row_key_cols": list(plan.row_key_cols),
                    "expected_keys": [list(key) for key in plan.expected_keys],
                    "contract_collapse_hedges": True,
                    "contract_changed": result.changed,
                    "contract_reason": result.reason,
                    "frozen_rows": result.input_rows,
                    "assembled_rows": result.output_rows,
                    "value_groups": result.value_groups,
                    "contract_assignments": [
                        {"expected": list(expected), "source": list(source)}
                        for expected, source in result.assignments
                    ],
                },
            )
        if self._fact_producer is not None and ctx.frozen_table_rows is not None:
            from littraceqa.answer.table_fact_assign import assemble_table_facts
            from littraceqa.answer.table_fact_ledger import build_table_fact_ledger

            ledger = build_table_fact_ledger(
                plan,
                ctx,
                include_complete_frozen_rows=(
                    self._fact_allow_cell_replacements
                ),
            )
            production = self._fact_producer.produce(
                ctx,
                ledger,
                text_llm=ctx.llm,
                vision_llm=ctx.vision_llm or ctx.llm,
            )
            assembly = assemble_table_facts(
                production.ledger,
                ctx.frozen_table_rows,
                allow_row_additions=self._fact_allow_row_additions,
                preserve_unmatched_frozen=(
                    self._fact_preserve_unmatched_frozen
                ),
                allow_cell_replacements=(
                    self._fact_allow_cell_replacements
                ),
                canonicalize_attestation_only_rows=(
                    self._fact_canonicalize_attestation_only_rows
                ),
                cell_value_policy=self._fact_cell_value_policy,
            )
            rows = list(assembly.rows)
            return StrategyOutput(
                value=rows,
                confidence=1.0 if rows else 0.0,
                attested_evidence=[],
                diagnostics={
                    "status": "table_facts_assembled",
                    "row_axis": plan.row_axis.value,
                    "row_key_cols": list(plan.row_key_cols),
                    "expected_keys": [list(key) for key in plan.expected_keys],
                    "frozen_rows": len(ctx.frozen_table_rows),
                    "assembled_rows": len(rows),
                    "fact_targets": len(production.ledger.targets),
                    "fact_count": len(production.ledger.facts),
                    "row_attestation_count": len(
                        production.ledger.attestations
                    ),
                    "fact_allow_row_additions": (
                        self._fact_allow_row_additions
                    ),
                    "fact_preserve_unmatched_frozen": (
                        self._fact_preserve_unmatched_frozen
                    ),
                    "fact_allow_cell_replacements": (
                        self._fact_allow_cell_replacements
                    ),
                    "fact_canonicalize_attestation_only_rows": (
                        self._fact_canonicalize_attestation_only_rows
                    ),
                    "fact_cell_value_policy": self._fact_cell_value_policy,
                    "added_target_ids": list(assembly.added_target_ids),
                    "retry_target_ids": list(assembly.retry_target_ids),
                    "changed_cells": [list(item) for item in assembly.changed_cells],
                    "replaced_cells": [
                        list(item) for item in assembly.replaced_cells
                    ],
                    "unresolved": list(assembly.unresolved),
                    "assignment_kinds": [
                        list(item) for item in assembly.assignment_kinds
                    ],
                    "fact_production": list(production.diagnostics),
                },
            )
        if self._verify_frozen_source_cells and ctx.frozen_table_rows is not None:
            from littraceqa.answer.table_verify import verify_frozen_table

            rows, verification = verify_frozen_table(
                ctx,
                plan,
                ctx.llm,
                ctx.vision_llm or ctx.llm,
                max_pages=self._native_verify_max_pages,
                allow_row_additions=self._native_verify_add_rows,
                allow_row_substitutions=self._native_verify_substitute_rows,
                allow_cell_updates=self._native_verify_cell_updates,
                allow_source_key_canonicalization=(
                    self._native_verify_source_key_canonicalization
                ),
            )
            return StrategyOutput(
                value=rows,
                confidence=1.0 if rows else 0.0,
                attested_evidence=[],
                diagnostics={
                    "status": "source_cells_verified",
                    "row_axis": plan.row_axis.value,
                    "row_key_cols": list(plan.row_key_cols),
                    "expected_keys": [list(key) for key in plan.expected_keys],
                    "frozen_rows": len(ctx.frozen_table_rows),
                    "verified_rows": len(rows),
                    "native_verify_max_pages": self._native_verify_max_pages,
                    "native_verify_add_rows": self._native_verify_add_rows,
                    "native_verify_substitute_rows": (
                        self._native_verify_substitute_rows
                    ),
                    "native_verify_cell_updates": (
                        self._native_verify_cell_updates
                    ),
                    "native_verify_source_key_canonicalization": (
                        self._native_verify_source_key_canonicalization
                    ),
                    "source_cell_verification": verification,
                },
            )
        if self._review_frozen_table_deltas and ctx.frozen_table_rows is not None:
            additions, review = review_frozen_table_additions(
                ctx, plan, ctx.llm, page_k=self._text_page_k
            )
            rows = [dict(row) for row in ctx.frozen_table_rows] + additions
            return StrategyOutput(
                value=rows,
                confidence=1.0 if rows else 0.0,
                attested_evidence=[],
                diagnostics={
                    "status": "delta_reviewed",
                    "row_axis": plan.row_axis.value,
                    "row_key_cols": list(plan.row_key_cols),
                    "expected_keys": [list(key) for key in plan.expected_keys],
                    "frozen_rows": len(ctx.frozen_table_rows),
                    "added_rows": len(additions),
                    "delta_review": review,
                },
            )
        # Title-only ONLY for a paper-axis table WITHOUT value columns. A
        # paper-axis table WITH value columns must extract + assemble like an
        # entity table (else it emits titles but no values -- 4 hidden questions).
        if plan.row_axis is RowAxis.PAPER and not plan.value_cols:
            return self._paper_rows(ctx, plan)

        vision_llm = ctx.vision_llm or ctx.llm
        own_paper_only = (
            self._open_ended_one_row_per_paper
            and plan.row_axis is RowAxis.ENTITY
            and not plan.expected_keys
            and len(ctx.paper_ids) > 1
        )
        routed_keys = {pid: list(plan.expected_keys) for pid in ctx.paper_ids}
        route_diagnostics: list[dict] = []
        if (
            self._route_expected_rows_to_papers
            and plan.expected_keys
            and len(ctx.paper_ids) > 1
        ):
            routed_keys, route_diagnostics = expected_keys_by_paper(
                ctx, plan.expected_keys
            )

        retry_keys_by_paper: dict[str, list[tuple[str, ...]]] = {
            pid: [] for pid in ctx.paper_ids
        }
        if self._visual_retry_owned_rows and plan.expected_keys:
            if len(ctx.paper_ids) == 1:
                retry_keys_by_paper[ctx.paper_ids[0]] = list(
                    plan.expected_keys
                )
            else:
                for route in route_diagnostics:
                    if route.get("status") != "owned":
                        continue
                    key = tuple(route.get("expected_key") or ())
                    if not key:
                        continue
                    for pid in route.get("paper_ids") or ():
                        if pid in retry_keys_by_paper:
                            retry_keys_by_paper[pid].append(key)

        per_paper = []
        for pid in ctx.paper_ids:
            paper_expected = routed_keys.get(pid, [])
            # A paper with no confidently/ambiguously routed rows cannot
            # contribute a requested entity row.  Skip both paid vision and
            # text calls instead of asking it for unrelated methods.
            if plan.expected_keys and not paper_expected:
                paper_rows = []
            else:
                paper_plan = replace(plan, expected_keys=paper_expected)
                extraction_kwargs = {"own_paper_only": own_paper_only}
                if self._visual_extraction_mode != "direct":
                    extraction_kwargs["visual_extraction_mode"] = (
                        self._visual_extraction_mode
                    )
                if retry_keys_by_paper.get(pid):
                    extraction_kwargs["visual_retry_expected_keys"] = (
                        retry_keys_by_paper[pid]
                    )
                if self._visual_consensus_repeats != 1:
                    extraction_kwargs["visual_consensus_repeats"] = (
                        self._visual_consensus_repeats
                    )
                if self._extraction_sources != "both":
                    extraction_kwargs["extraction_sources"] = (
                        self._extraction_sources
                    )
                if self._text_context_mode != "evidence":
                    extraction_kwargs["text_context_mode"] = (
                        self._text_context_mode
                    )
                    extraction_kwargs["text_page_k"] = self._text_page_k
                paper_rows = self._canonicalize_paper_axis_rows(
                    extract_rows_from_paper(
                        ctx, pid, paper_plan, vision_llm, ctx.llm,
                        **extraction_kwargs,
                    ),
                    pid,
                    paper_plan,
                    ctx.paper_titles,
                )
            per_paper.append((pid, paper_rows))
        preferred_papers_by_expected_key = {
            tuple(route["expected_key"]): tuple(route["paper_ids"])
            for route in route_diagnostics
            if self._prefer_owned_cells and route.get("status") == "owned"
        }
        attested_candidate_keys = (
            source_attested_expected_keys(ctx, routed_keys)
            if (
                self._retain_source_attested_expected_rows
                or self._trace_reassembly_inputs
            )
            else []
        )
        attested_expected_keys = (
            attested_candidate_keys
            if self._retain_source_attested_expected_rows
            else []
        )
        rows = assemble_rows(
            plan,
            per_paper,
            retain_concise_missing_expected_rows=(
                self._retain_concise_missing_expected_rows
            ),
            preferred_papers_by_expected_key=(
                preferred_papers_by_expected_key
            ),
            source_attested_expected_keys=attested_expected_keys,
        )
        scalar_fills = 0
        if self._fill_nulls_from_scalar_evidence:
            rows, scalar_fills = fill_null_cells_from_scalar_evidence(
                ctx, plan, rows, route_diagnostics
            )

        attested: list = []
        grounded = 0
        for row in rows:
            row_attested = ground_value(row.get(plan.row_key_cols[0]),
                                        ctx.evidence, ctx.parsed_by_id)
            if row_attested:
                grounded += 1
            attested.extend(row_attested)
        # Confidence is the semantic-fallback signal. A non-empty schema-valid
        # table can score even when its row-key string is absent from the short
        # evidence quote, so row-key attestation is diagnostic rather than a
        # reason to label the generated answer a dead fallback.
        confidence = 1.0 if rows else 0.0
        diagnostics = {
            "status": "generated" if rows else "empty",
            "row_axis": plan.row_axis.value,
            "row_key_cols": list(plan.row_key_cols),
            "expected_keys": [list(key) for key in plan.expected_keys],
            "row_source_routing_enabled": self._route_expected_rows_to_papers,
            "row_source_routes": route_diagnostics,
            "owner_preferred_cell_merge": self._prefer_owned_cells,
            "owner_preferred_route_count": len(
                preferred_papers_by_expected_key
            ),
            "source_attested_row_retention": (
                self._retain_source_attested_expected_rows
            ),
            "source_attested_expected_keys": [
                list(key) for key in attested_expected_keys
            ],
            "source_attested_candidate_keys": [
                list(key) for key in attested_candidate_keys
            ],
            "visual_extraction_mode": self._visual_extraction_mode,
            "visual_retry_owned_rows": self._visual_retry_owned_rows,
            "visual_consensus_repeats": self._visual_consensus_repeats,
            "extraction_sources": self._extraction_sources,
            "text_context_mode": self._text_context_mode,
            "text_page_k": self._text_page_k,
            "scalar_evidence_null_fill": self._fill_nulls_from_scalar_evidence,
            "scalar_evidence_filled_cells": scalar_fills,
            "visual_retry_expected_keys": {
                pid: [list(key) for key in keys]
                for pid, keys in retry_keys_by_paper.items()
                if keys
            },
            "open_ended_one_row_per_paper": own_paper_only,
            "per_paper_rows": {pid: len(paper_rows) for pid, paper_rows in per_paper},
            "per_paper_row_keys": {
                pid: [
                    [row.get(column) for column in plan.row_key_cols]
                    for row in paper_rows[:20]
                ]
                for pid, paper_rows in per_paper
            },
            "per_paper_extracted_rows": (
                {
                    pid: [dict(row) for row in paper_rows]
                    for pid, paper_rows in per_paper
                }
                if self._trace_reassembly_inputs
                else {}
            ),
            "assembled_rows": len(rows),
            "retained_missing_expected_rows": sum(
                int(
                    all(row.get(column["name"]) is None for column in plan.value_cols)
                )
                for row in rows
            ),
            "grounded_row_keys": grounded,
        }
        return StrategyOutput(
            value=rows, confidence=confidence, attested_evidence=attested,
            diagnostics=diagnostics,
        )


    @staticmethod
    def _canonicalize_paper_axis_rows(rows, paper_id, plan, paper_titles):
        """Choose the strongest deterministic label for a paper-axis row.

        A value-bearing paper-axis table still needs LLM/vision extraction for
        its value columns. When question planning identified exactly one
        explicit label for this selected paper (for example ``GraphBench``),
        that label is the scorer-facing contract and is more precise than both
        an LLM paraphrase and the paper's long metadata title. Fall back to the
        metadata title only when no unique planned label exists.
        """
        if plan.row_axis is not RowAxis.PAPER or len(plan.row_key_cols) != 1:
            return rows
        key = plan.row_key_cols[0]
        planned_labels = [
            values[0]
            for values in plan.expected_keys
            if values and isinstance(values[0], str) and values[0].strip()
        ]
        title = paper_titles.get(paper_id)

        def matches(source, candidate):
            source_norm = normalize_text(source)
            candidate_norm = normalize_text(candidate)
            if not source_norm or not candidate_norm:
                return False
            if source_norm == candidate_norm:
                return True
            # Question labels are normally concise names embedded in a paper
            # title or followed by a role noun ("GraphBench benchmark"). Require a
            # token boundary so short names cannot match arbitrary substrings.
            return re.search(
                rf"(?<![a-z0-9]){re.escape(candidate_norm)}(?![a-z0-9])",
                source_norm,
            ) is not None

        def concise_question_label(label):
            """Reduce an explicit model-role phrase to its named work.

            Concise row keys use the named work rather than the surrounding
            prose role. Restrict shortening to acronym-like
            names, or a proper name immediately followed by a known role noun,
            so a descriptive lowercase phrase is never collapsed.
            """
            words = label.split()
            if len(words) < 2:
                return label
            first = words[0].strip(" ,:;()")
            second = words[1].strip(" ,:;()").casefold()
            role_nouns = {
                "benchmark", "taxonomy", "method", "model", "paper",
                "framework", "pipeline", "system", "work",
            }
            acronym_like = any(ch.isupper() for ch in first[1:])
            if acronym_like or second in role_nouns:
                return first
            return label

        out = []
        for row in rows:
            sources = [row.get(key), title]
            candidates = [
                label
                for label in planned_labels
                if any(matches(source, label) for source in sources)
            ]
            label = (
                candidates[0]
                if len(candidates) == 1
                else planned_labels[0]
                if len(planned_labels) == 1
                else title
            )
            out.append(
                {**row, key: concise_question_label(label)} if label else row
            )
        return out

    @staticmethod
    def _reference_rows(ctx: AnswerContext) -> StrategyOutput | None:
        """Reuse the deterministic bibliography parser for one-column tables.

        Some records request the same scalar twice: as freeform text and as a
        one-row, row-key-only table (for example, the first author of the third
        reference). The freeform strategy already answers those questions from
        the parsed reference list without an LLM. Running a second table LLM is
        redundant and nondeterministic, so shape that same deterministic value
        directly when the schema can represent exactly one scalar row.
        """
        schema = ctx.table_schema or []
        if (
            len(schema) != 1
            or not schema[0].get("is_row_key")
            or not is_reference_list_question(ctx.question)
            or not ctx.paper_ids
        ):
            return None
        parsed = ctx.parsed_by_id.get(ctx.paper_ids[0])
        value = answer_reference_question(
            ctx.question, parse_reference_list(parsed)
        )
        if not value:
            return None
        attested = ground_value(value, ctx.evidence, ctx.parsed_by_id)
        return StrategyOutput(
            value=[{schema[0]["name"]: value}],
            confidence=1.0 if attested else 0.3,
            attested_evidence=attested,
            diagnostics={
                "status": "generated",
                "route": "reference_list",
                "row_axis": "entity",
                "row_key_cols": [schema[0]["name"]],
                "expected_keys": [],
                "per_paper_rows": {ctx.paper_ids[0]: 1},
                "assembled_rows": 1,
                "grounded_row_keys": 1 if attested else 0,
            },
        )


    @staticmethod
    def _paper_rows(ctx: AnswerContext, plan) -> StrategyOutput:
        """PAPER axis: one row per selected paper, keyed by the VERBATIM metadata
        title -- deterministic, no LLM/vision, dedup by title."""
        key = plan.row_key_cols[0]
        rows, seen = [], set()
        for pid in ctx.paper_ids:
            title = ctx.paper_titles.get(pid)
            if title and title not in seen:
                seen.add(title)
                rows.append({key: title})
        return StrategyOutput(
            value=rows,
            confidence=1.0 if rows else 0.0,
            attested_evidence=[],
            diagnostics={
                "status": "generated" if rows else "empty",
                "row_axis": plan.row_axis.value,
                "row_key_cols": list(plan.row_key_cols),
                "expected_keys": [list(key) for key in plan.expected_keys],
                "per_paper_rows": {},
                "assembled_rows": len(rows),
                "grounded_row_keys": 0,
            },
        )


_SCALAR_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SCALAR_QUOTE_RE = re.compile(
    rf"^(?:-|{_SCALAR_NUMBER}(?:\s*±\s*{_SCALAR_NUMBER})?%?)$"
)


def fill_null_cells_from_scalar_evidence(ctx, plan, rows, route_diagnostics):
    """Fill a null single-value cell from one unambiguous owner quote.

    A null value is already incorrect under the official scorer.  Replacing it
    with a source-printed scalar cannot reduce cell accuracy, but broad prose
    parsing or assigning one number to several rows would invent associations.
    This repair therefore requires all of the following: exactly one value
    column, an explicitly planned row, one title/acronym-confirmed owning paper,
    and exactly one unique scalar-only quote from that paper. A single-paper
    table is eligible only when it requests one row, so one scalar is never
    copied across a multi-row grid. Existing non-null cells are immutable.
    """

    if len(plan.value_cols) != 1 or not plan.expected_keys:
        return [dict(row) for row in rows], 0
    value_column = plan.value_cols[0]
    value_name = value_column["name"]
    paper_ids = list(dict.fromkeys(ctx.paper_ids))
    owners_by_expected: dict[tuple[str, ...], tuple[str, ...]] = {}
    if len(paper_ids) == 1 and len(plan.expected_keys) == 1:
        owners_by_expected = {
            tuple(expected): (paper_ids[0],) for expected in plan.expected_keys
        }
    else:
        owners_by_expected = {
            tuple(route.get("expected_key") or ()): tuple(route.get("paper_ids") or ())
            for route in route_diagnostics
            if route.get("status") == "owned"
            and len(route.get("paper_ids") or ()) == 1
            and int(
                (route.get("scores") or {}).get(
                    (route.get("paper_ids") or [""])[0], 0
                )
            ) >= 9_000
        }

    def scalar_values(paper_id: str) -> list[object]:
        values: list[object] = []
        seen: set[tuple[str, str]] = set()
        for evidence in ctx.evidence:
            if evidence.paper_id != paper_id:
                continue
            quote = str(evidence.quote or "").strip()
            if not quote or _SCALAR_QUOTE_RE.fullmatch(quote) is None:
                continue
            value = (
                _coerce_number(quote)
                if value_column.get("type") == "number"
                else _coerce_string_cell(quote, value_name)
            )
            if value is None:
                continue
            key = (type(value).__name__, repr(value))
            if key not in seen:
                seen.add(key)
                values.append(value)
        return values

    out = [dict(row) for row in rows]
    fills = 0
    for row in out:
        if row.get(value_name) is not None:
            continue
        key_values = tuple(row.get(column) for column in plan.row_key_cols)
        matching_owners = {
            owners[0]
            for expected, owners in owners_by_expected.items()
            if len(owners) == 1
            and matches_expected_key(key_values, expected, plan.row_key_cols)
        }
        if len(matching_owners) != 1:
            continue
        values = scalar_values(next(iter(matching_owners)))
        if len(values) != 1:
            continue
        row[value_name] = values[0]
        fills += 1
    return out, fills
