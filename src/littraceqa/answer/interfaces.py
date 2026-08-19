"""Answer-pipeline contracts: data model, Protocol, and strategy registry.

Dispatch is OCP (mirrors littraceqa.localize.interfaces /
littraceqa.retrieval.interfaces): each answer strategy (freeform / MC /
table) registers itself; `build_strategy` looks it up by name and never
changes when a new strategy is added. `AnswerStrategy` is a Protocol (DIP)
so the answer service depends on the interface, not concrete strategy
classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Callable

from littraceqa.localize.interfaces import LocatedEvidence, ParsedPdf
from littraceqa.llm.interfaces import LLMClient


@dataclass(frozen=True)
class AnswerContext:
    """Everything a strategy needs to produce an answer: the question, the
    answer-type hints and scope, the located evidence and parsed pages to
    ground against, paper titles (for citation/prompting), and the LLM
    client. `mc_options` / `table_schema` are populated only for the
    corresponding answer types."""
    question: str
    answer_types: list[str]
    paper_ids: list[str]
    evidence: list[LocatedEvidence]
    parsed_by_id: dict[str, ParsedPdf]
    paper_titles: dict[str, str]
    llm: LLMClient
    mc_options: dict[str, str] | None = None
    table_schema: list[dict] | None = None
    # Optional stronger multimodal client for the figure/table VISION path
    # (e.g. gemini-2.5-pro, which reads dense figures far better than flash).
    # Falls back to `llm` when None.
    vision_llm: LLMClient | None = None
    # SHARED PDF SOURCE (FIX 1): the raw PDF bytes retained by the runner when it
    # fetched each selected paper for evidence localization, keyed by paper_id.
    # The vision path renders the cited page from THESE bytes so the GCS/streaming
    # corpus path (which never fills the on-disk cache) can still read figures/
    # tables. Empty by default -> vision falls back to the on-disk cache (the old
    # local-cache path is untouched).
    pdf_bytes_by_id: dict[str, bytes] = field(default_factory=dict)
    # Optional immutable draft used only by table-replay repair strategies.
    # Ordinary full runs leave this unset. A delta reviewer can inspect the
    # frozen validator-approved rows without granting it permission to rewrite
    # papers, evidence, MC answers, or existing table cells.
    frozen_table_rows: list[dict] | None = None


@dataclass(frozen=True)
class StrategyOutput:
    """A strategy's raw result, before serialization into the scored
    `answer` payload: `value` (str for freeform, letter str for MC,
    `list[dict]` rows for table), a confidence score, and the evidence
    items that ground `value`."""
    value: object
    confidence: float
    attested_evidence: list[LocatedEvidence]
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerResult:
    """The final, scorer-facing payload assembled from a `StrategyOutput`."""
    answer: dict
    attested_evidence: list[dict]
    confidence: float
    component_confidences: dict[str, float] = field(default_factory=dict)
    component_diagnostics: dict[str, dict] = field(default_factory=dict)


@runtime_checkable
class AnswerStrategy(Protocol):
    answer_type: str

    def answer(self, ctx: AnswerContext) -> StrategyOutput: ...


_STRATEGIES: dict[str, Callable[..., AnswerStrategy]] = {}


def register_strategy(name: str):
    """Decorator: register `cls` as the answer strategy for `name`.

    This is the sole extension point — adding a new answer strategy never
    requires touching `build_strategy`."""
    def deco(cls):
        if name in _STRATEGIES:
            raise ValueError(f"strategy {name!r} already registered")
        _STRATEGIES[name] = cls
        return cls
    return deco


def build_strategy(name: str, **kwargs) -> AnswerStrategy:
    """Build the answer strategy registered as `name`, dispatched through
    the `register_strategy` registry. Closed for modification: a new
    strategy is added by registration, not by editing this function."""
    if name not in _STRATEGIES:
        raise KeyError(f"no strategy registered as {name!r}; have {sorted(_STRATEGIES)}")
    return _STRATEGIES[name](**kwargs)
