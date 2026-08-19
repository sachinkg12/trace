"""AnswerPipeline: the answer-pipeline orchestrator.

Dispatches each requested `answer_types` entry to its registered strategy
(via `build_strategy`, unless a `strategies` dict is injected for testing),
merges every strategy's `answers.serialize`d payload into the scorer-shaped
`answer` dict, and assembles the `AnswerResult` the scorer/dev-eval loop
consumes.

Always-answer, never-fatal: a strategy that raises is SKIPPED for that
answer_type only (never propagates), so one bad type never blanks the
whole response; the emitted `answer` dict simply omits that key.

Confidence -> precision gate (fork 2): `attested_evidence` is the DEDUPED
union of every strategy's attested `LocatedEvidence`, converted to scorer
dicts via `littraceqa.evidence.make_evidence`. If the best confidence across
all requested types is below `evidence_confidence_floor`, the evidence is
withheld entirely (`[]`) -- the answer value itself is NEVER withheld, only
its evidence attribution, which is the precision lever `evidence_confidence_floor`
tunes (dev-eval loop, later).
"""
from __future__ import annotations

from littraceqa import answers
from littraceqa.answer.interfaces import AnswerContext, AnswerResult, AnswerStrategy, build_strategy
from littraceqa.evidence import make_evidence
from littraceqa.localize.interfaces import LocatedEvidence

_DEFAULT_TYPES = ("freeform", "multiple_choice", "table")


class AnswerPipeline:
    def __init__(self, *, strategies: dict[str, AnswerStrategy] | None = None,
                 evidence_confidence_floor: float = 0.5):
        self._strategies = (strategies if strategies is not None
                            else {name: build_strategy(name) for name in _DEFAULT_TYPES})
        self._evidence_confidence_floor = evidence_confidence_floor

    def answer(self, ctx: AnswerContext) -> AnswerResult:
        answer_dict: dict = {}
        attested: list[LocatedEvidence] = []
        seen: set[tuple] = set()
        best_confidence = 0.0
        component_confidences: dict[str, float] = {}
        component_diagnostics: dict[str, dict] = {}

        for answer_type in ctx.answer_types:
            strategy = self._strategies.get(answer_type)
            if strategy is None:
                continue
            # serialize+merge live INSIDE the try alongside strategy.answer:
            # a strategy raise OR a serializer failure (e.g. a value the
            # serializer can't shape) skips that answer_type only, never
            # propagating out of AnswerPipeline.answer.
            try:
                out = strategy.answer(ctx)
                serialized = answers.serialize(answer_type, out.value)
            except Exception:  # noqa: BLE001 -- one bad type is skipped, never fatal
                continue

            answer_dict.update(serialized)
            best_confidence = max(best_confidence, out.confidence)
            component_confidences[answer_type] = float(out.confidence)
            component_diagnostics[answer_type] = dict(out.diagnostics or {})
            for ev in out.attested_evidence:
                key = (ev.paper_id, ev.source_type, ev.page, ev.object_id)
                if key in seen:
                    continue
                seen.add(key)
                attested.append(ev)

        attested_dicts: list[dict] = []
        if best_confidence >= self._evidence_confidence_floor:
            for ev in attested:
                try:
                    attested_dicts.append(make_evidence(ev.paper_id, ev.source_type, ev.page, ev.object_id))
                except (ValueError, KeyError):
                    continue

        return AnswerResult(answer=answer_dict, attested_evidence=attested_dicts,
                            confidence=best_confidence,
                            component_confidences=component_confidences,
                            component_diagnostics=component_diagnostics)
