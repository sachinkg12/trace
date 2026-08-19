"""Pipeline orchestrator: threads seed(#4) -> evidence(#5) -> paper-set(#6)
-> answer(#7) into one submission line, per query.

`_attempt` runs the full seed->evidence->paperset->answer thread at a given
seed breadth (`top_n`). `run` performs one base attempt at `top_n_seeds`
and, on a grounding failure (low `AnswerResult.confidence`), re-attempts
with a WIDENED seed breadth to pull in different (more) evidence -- see
`_run` below and the module docstring on `run` for the trigger-gated
re-localization loop. The answer is ALWAYS emitted: re-localization only
ever swaps in a better-grounded attempt, it never blanks one.

Never-drop, never-crash contract: the WHOLE body of `run` is wrapped so any
unexpected failure still returns a valid, placeholder-bearing submission line
for `record.query_id` (a dropped query_id scores 0 precision in the official
scorer -- see `littraceqa.submission.write_submission`). Each external call
(seed finding, per-seed evidence localization, anchor extraction) is ALSO
wrapped individually so one flaky component degrades to an empty result
rather than aborting the whole query.
"""
from __future__ import annotations

from littraceqa.answer.interfaces import AnswerContext, AnswerResult
from littraceqa.llm.interfaces import LLMClient
from littraceqa.localize.interfaces import LocatedEvidence, ParsedPdf
from littraceqa.pipeline.input import InputRecord
from littraceqa.seed.anchor import extract_anchor
from littraceqa.submission import build_fallback_line, build_record_line


class Pipeline:
    def __init__(self, *, seed_finder, evidence_service, paperset_selector,
                 answer_pipeline, pool, llm: LLMClient, top_n_seeds: int = 3,
                 relocalize: bool = True, relocalize_floor: float = 0.5,
                 max_relocalize: int = 1, relocalize_widen: int = 3,
                 vision_llm: LLMClient | None = None):
        self._seed_finder = seed_finder
        self._evidence_service = evidence_service
        self._paperset_selector = paperset_selector
        self._answer_pipeline = answer_pipeline
        self._pool = pool
        self._llm = llm
        self._vision_llm = vision_llm
        self._top_n_seeds = top_n_seeds
        self._relocalize = relocalize
        self._relocalize_floor = relocalize_floor
        self._max_relocalize = max_relocalize
        self._relocalize_widen = relocalize_widen

    def run(self, record: InputRecord) -> dict:
        try:
            return self._run(record)
        except Exception:  # noqa: BLE001 -- never drop a query_id, never crash
            return build_fallback_line(record)

    def _run(self, record: InputRecord) -> dict:
        result, line = self._attempt(record, self._top_n_seeds)

        if self._relocalize:
            attempt_index = 1
            while (result.confidence < self._relocalize_floor
                   and attempt_index <= self._max_relocalize):
                top_n = self._top_n_seeds + attempt_index * self._relocalize_widen
                try:
                    candidate_result, candidate_line = self._attempt(record, top_n)
                except Exception:  # noqa: BLE001 -- a failed widened retry keeps the
                    # incumbent (base) attempt; re-localization must never BLANK an
                    # already-computed answer, only ever swap in a better-grounded one.
                    attempt_index += 1
                    continue
                if candidate_result.confidence > result.confidence:
                    result, line = candidate_result, candidate_line
                attempt_index += 1

        return line

    def _attempt(self, record: InputRecord, top_n: int) -> tuple[AnswerResult, dict]:
        try:
            seeds = self._seed_finder.find(record.question, top_n=top_n)
        except Exception:  # noqa: BLE001 -- one bad seed-finder call degrades, not fatal
            seeds = []

        evidence: list[LocatedEvidence] = []
        parsed_by_id: dict[str, ParsedPdf] = {}
        evidence_paper_ids: list[str] = []
        for seed in seeds:
            paper = self._pool.by_id(seed.paper_id)
            if paper is None:
                continue
            try:
                located = self._evidence_service.locate_located(record.question, paper)
            except Exception:  # noqa: BLE001 -- one bad seed's evidence degrades to []
                located = []
            if located:
                evidence.extend(located)
                if seed.paper_id not in evidence_paper_ids:
                    evidence_paper_ids.append(seed.paper_id)
            try:
                parsed = self._evidence_service.parsed_for(paper)
            except Exception:  # noqa: BLE001 -- a raising parser degrades this seed only
                parsed = None
            if parsed is not None:
                parsed_by_id[paper.paper_id] = parsed

        try:
            asks_multiple = extract_anchor(self._llm, record.question).asks_multiple
        except Exception:  # noqa: BLE001 -- anchor extraction degrades to False
            asks_multiple = False

        paper_ids = self._paperset_selector.select(
            seeds, evidence_paper_ids, asks_multiple=asks_multiple
        )
        paper_titles = {
            pid: p.title for pid in paper_ids if (p := self._pool.by_id(pid))
        }

        ctx = AnswerContext(
            record.question, record.answer_types, paper_ids, evidence, parsed_by_id,
            paper_titles, self._llm, mc_options=record.mc_options,
            table_schema=record.table_schema, vision_llm=self._vision_llm,
        )
        result = self._answer_pipeline.answer(ctx)

        line = build_record_line(
            record, paper_ids, result.attested_evidence, result.answer
        )
        return result, line
