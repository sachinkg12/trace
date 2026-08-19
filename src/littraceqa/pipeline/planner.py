"""QuestionPlanner: one LLM call that reads a question (+ its answer scaffold)
and emits a structured `Plan` -- the ROUTER for the retrieval + precision-
cascade pipeline.

The plan carries four things downstream:
  * `criterion` -- the machine-checkable constraint the precision cascade's
    `DeterministicFilterStage` enforces. Its keys are IDENTICAL to what that
    stage consumes (`venue`, `year`, `required_source_type`) plus a
    natural-language `description` -- the key the cascade's `_criterion_text`
    actually reads to build the reranker query + adversarial-validator prompt.
    A hard venue/year constraint is emitted ONLY when the question states one;
    absent constraints are None so the cascade does not over-filter.
    (`required_source_type` is a SOFT hint, not a hard filter -- see cascade.)
  * `named_methods` -- method/model/paper names the question explicitly names
    (the seeds to resolve). A superset of anchor.py's `method_acronyms`.
  * `strategies` -- which retrieval routes to run, chosen by QUESTION SHAPE
    (named method -> "name"; "which papers cite X" -> "citation"; "papers that
    do X" with no name -> "property"; ...). "knn" is always kept as a fallback.
  * `multiplicity` -- "single" vs "multi", replacing the brittle keyword
    `asks_multiple` gate: "multi" when the answer is inherently a SET of papers.

Defensive by design, mirroring `seed/anchor.py`: strips code fences, `json.loads`,
validates/coerces every field, and on ANY failure (LLM raises, empty/blocked
response, invalid JSON, wrong shape) DEGRADES to a safe default plan carrying a
cheap keyword multiplicity heuristic. `plan()` NEVER raises. Every call runs at
`temperature=0.0` for reproducibility.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from littraceqa.llm.interfaces import LLMClient
from littraceqa.pipeline.input import InputRecord

_log = logging.getLogger(__name__)

# Retrieval strategies the router may select from. "knn" is the always-available
# dense fallback; the rest are shape-specific. Unknown strategies are dropped.
_ALLOWED_STRATEGIES = ("name", "baseline", "citation", "property", "knn")
_FALLBACK_STRATEGY = "knn"
# The safe default strategy set when the LLM plan is unusable.
_DEFAULT_STRATEGIES = ["name", "knn"]

# Only these two evidence modalities are hard-enforceable by the cascade.
_SOURCE_TYPES = ("table", "figure")

_MULTIPLICITY_VALUES = ("single", "multi")
_TARGET_ROLES = ("target", "evidence_anchor", "constraint")
_MAX_DESIRED_PAPERS = 20


@dataclass(frozen=True)
class PlanTarget:
    """One entity mentioned in the question and its selection semantics.

    ``target`` entities should receive paper coverage, ``evidence_anchor``
    entities help retrieve citing/comparing papers but must not themselves be
    emitted merely because they were named, and ``constraint`` entities (for
    example a dataset) refine retrieval without consuming a paper slot.
    """

    key: str
    text: str
    role: str = "target"


@dataclass(frozen=True)
class Plan:
    """The structured routing plan for one question.

    `criterion` mirrors the cascade `DeterministicFilterStage` keys EXACTLY:
      * `venue`  -- short venue name (e.g. "CVPR") or None if unconstrained.
      * `year`   -- int (e.g. 2025) or None if unconstrained.
      * `required_source_type` -- "table" | "figure" | None (a demanded
        evidence modality; None if the question does not require one). SOFT --
        the cascade uses it as a rerank/attestation hint, not a hard drop.
      * `description` -- a concise natural-language statement of what the answer
        paper(s) must satisfy. This is the exact key the cascade's
        `_criterion_text` reads for the reranker query + LLM validator prompt.
    """

    criterion: dict
    named_methods: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    multiplicity: str = "single"
    targets: list[PlanTarget] = field(default_factory=list)
    desired_paper_count: int | None = None


_SYSTEM_PROMPT = (
    "You are a ROUTER for a scientific-paper question-answering pipeline. Read "
    "the question (and any note that a multiple-choice or table answer scaffold "
    "exists) and produce a PLAN. Respond with STRICT minified JSON only -- no "
    "prose, no markdown, no code fences. The JSON object must have EXACTLY these "
    "four keys:\n"
    '"criterion": an object with EXACTLY these keys:\n'
    '  - "venue": the SHORT venue name (e.g. "CVPR", "NAACL", "ICML", "NeurIPS") '
    "IF the question states a hard venue constraint (e.g. \"CVPR 2025 papers\"), "
    "else null. Use the short acronym, not the full name.\n"
    '  - "year": the year as an INTEGER (e.g. 2025) IF the question states a hard '
    "year constraint, else null.\n"
    '  - "required_source_type": "table" or "figure" IF the answer MUST come from '
    "that object (e.g. \"in Table 3\", \"which figure shows...\"), else null. "
    "Never use any other value.\n"
    '  - "description": a concise natural-language statement of what the answer '
    "paper(s) must satisfy (used later to validate candidates). Never empty.\n"
    '"named_methods": a list of strings -- every method/model/paper name the '
    "question EXPLICITLY mentions (the seeds to resolve). Empty list if none. Do "
    "not invent names.\n"
    '"strategies": a list, a subset of ["name","baseline","citation","property",'
    '"knn"]. Choose by question shape: a named method -> "name" (add "baseline" '
    "if it asks about that paper's comparisons/baselines); a 'which papers cite "
    "X' question -> 'citation'; a 'papers that do X' question with NO name -> "
    "'property'. Always include \"knn\" as a fallback.\n"
    '"multiplicity": "multi" if the answer is inherently a SET of papers '
    "(comparison, enumeration, \"which papers...\"), else \"single\"."
)

_STRUCTURED_SYSTEM_PROMPT = (
    "You are a ROUTER for a scientific-paper question-answering pipeline. Read "
    "the complete question and answer scaffold. Respond with STRICT minified "
    "JSON only -- no prose, markdown, or code fences. The JSON object must have "
    "EXACTLY these six keys:\n"
    '"criterion": an object with exactly "venue", "year", '
    '"required_source_type", and "description". Emit a short venue and integer '
    "year only for hard question-scope constraints; required_source_type is "
    '"table", "figure", or null; description is never empty.\n'
    '"named_methods": every explicitly named method/model/paper, as strings.\n'
    '"strategies": a subset of ["name","baseline","citation","property","knn"]. '
    'Use "citation" when the requested papers cite or compare against an anchor. '
    'Always include "knn".\n'
    '"multiplicity": "multi" only when the answer requires facts or evidence '
    'from more than one distinct reporting paper; otherwise "single". Count '
    'source papers, never answer values, table rows, methods, models, datasets, '
    'metrics, scenes, or baseline rows/method mentions inside another study.\n'
    '"targets": a list of objects with exactly "key", "text", and "role". '
    'role is "target" for each distinct REPORTING PAPER/SOURCE the answer must cover, '
    '"evidence_anchor" for a cited/baseline paper used to find other papers, or '
    '"constraint" for a dataset/benchmark/condition that refines retrieval but '
    "must not consume a paper slot. Methods, models, baselines, answer rows, "
    "metrics, datasets, and settings inside one reporting study are constraints, "
    "not separate paper targets. Several values, rows, methods, or baselines from one study, "
    "table, evaluation, or framework require ONE target. Do not create one paper target per requested value. "
    "A contrasted paper is only a constraint when it is supplied as a clue but "
    "the requested answer/evidence comes from the other paper. Create two targets "
    "only when the answer actually requires facts or evidence from both sources. "
    "key must be a "
    "short stable identifier and text must preserve the question wording.\n"
    '"desired_paper_count": the inferred number of DISTINCT reporting source '
    "papers as an integer whenever the source grouping is clear, even if the "
    "question does not state a literal number; otherwise null. Do not count "
    "evidence_anchor or constraint objects as answer papers.\n"
    "Silently perform this source-decomposition before emitting JSON: "
    "(1) list the requested answer facts, (2) identify the reporting-paper "
    "clause that owns each fact, (3) merge facts with the same owner/study/table, "
    "(4) count the remaining source groups, and (5) emit one target per group. "
    "Do not output this reasoning. Example: two model scores requested from the "
    "same named study are one target and desired_paper_count=1. Efficiency "
    "requested from one study and training cost requested from a different "
    "study are two targets and desired_paper_count=2."
)

# Optional ```json / ``` fences -> stripped to the bare JSON body before parsing.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# Cheap keyword heuristic for the DEGRADED path: the answer is a SET of papers
# when the question enumerates over papers ("which ... papers", "papers that/
# cite/which", "compare ...", "across all ... papers").
_MULTI_RE = re.compile(
    r"\bwhich\b[^.?!]*\bpapers\b"
    r"|\bpapers\b[^.?!]*\b(?:that|cite|which)\b"
    r"|\ball\b[^.?!]*\bpapers\b"
    r"|\bcompare\b",
    re.IGNORECASE,
)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _as_str_list(value: object) -> list[str]:
    """Coerce a parsed JSON value into `list[str]`, dropping non-strings.

    Mirrors `anchor._as_str_list`: a bare string is treated as absent (never
    exploded into characters), and non-string items (a stray object/number) are
    dropped rather than `str(...)`-stringified into corrupt seeds.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _coerce_year(value: object) -> int | None:
    """Accept a JSON int, or a purely-numeric string (e.g. "2025"); else None.
    `bool` is rejected (JSON `true`/`false` are not years)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _coerce_venue(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_source_type(value: object) -> str | None:
    if isinstance(value, str) and value.strip().casefold() in _SOURCE_TYPES:
        return value.strip().casefold()
    return None


def _coerce_strategies(value: object) -> list[str]:
    """Filter to allowed strategies (dedup, order-preserving) and GUARANTEE the
    "knn" fallback is present. An empty/invalid list degrades to the default."""
    ordered: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item in _ALLOWED_STRATEGIES and item not in ordered:
                ordered.append(item)
    if not ordered:
        return list(_DEFAULT_STRATEGIES)
    if _FALLBACK_STRATEGY not in ordered:
        ordered.append(_FALLBACK_STRATEGY)
    return ordered


def _coerce_targets(value: object) -> list[PlanTarget]:
    """Validate structured target objects without inventing missing entities."""
    if not isinstance(value, list):
        return []
    out: list[PlanTarget] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        raw_role = item.get("role")
        role = raw_role.strip().casefold() if isinstance(raw_role, str) else None
        key = item.get("key")
        if not (isinstance(text, str) and text.strip()):
            continue
        if role not in _TARGET_ROLES:
            continue
        if not (isinstance(key, str) and key.strip()):
            key = text
        identity = key.strip().casefold()
        if identity in seen:
            continue
        seen.add(identity)
        out.append(PlanTarget(key=key.strip(), text=text.strip(), role=role))
    return out


def _coerce_desired_paper_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int):
        return None
    return value if 1 <= value <= _MAX_DESIRED_PAPERS else None


def _keyword_multiplicity(question: str) -> str:
    """Cheap fallback: "multi" if the question enumerates over a SET of papers,
    else "single". Used only on the degraded path (no usable LLM plan)."""
    return "multi" if _MULTI_RE.search(question or "") else "single"


def _default_plan(question: str) -> Plan:
    """The safe plan used whenever the LLM response is unusable. Never filters
    (all hard constraints None), resolves nothing by name, runs name+knn, and
    guesses multiplicity from the cheap keyword heuristic."""
    return Plan(
        criterion={
            "venue": None,
            "year": None,
            "required_source_type": None,
            "description": question,
        },
        named_methods=[],
        strategies=list(_DEFAULT_STRATEGIES),
        multiplicity=_keyword_multiplicity(question),
    )


class QuestionPlanner:
    """Routes a question to a structured `Plan` via one temperature-0 LLM call.

    Depends on the `LLMClient` Protocol (DIP) -- tests inject `FakeLLM`, the real
    pipeline injects Gemini. `plan()` is TOTAL: it never raises, degrading to a
    safe default plan on any LLM/parse failure.

    Degradation is OBSERVABLE: every fall-back to the default plan increments the
    public `degrade_count` counter AND logs a WARNING, so a silent no-op (the LLM
    quietly failing on every question, leaving the pipeline unrouted) surfaces
    instead of hiding. A well-formed plan never touches the counter.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        structured_targets: bool = False,
        max_attempts: int = 1,
    ):
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise TypeError("max_attempts must be an integer")
        if max_attempts < 1 or max_attempts > 3:
            raise ValueError("max_attempts must be between 1 and 3")
        self._llm = llm
        self._structured_targets = bool(structured_targets)
        self._max_attempts = max_attempts
        self.degrade_count = 0

    def _degrade(self, question: str, reason: str) -> Plan:
        """Fall back to the safe default plan, observably: bump the counter and
        emit a WARNING naming why this question could not be routed by the LLM."""
        self.degrade_count += 1
        _log.warning("QuestionPlanner degraded to default plan (%s); "
                     "degrade_count=%d", reason, self.degrade_count)
        return _default_plan(question)

    def plan(self, record: InputRecord) -> Plan:
        question = record.question or ""
        prompt = self._build_prompt(record)
        system = (
            _STRUCTURED_SYSTEM_PROMPT if self._structured_targets
            else _SYSTEM_PROMPT
        )
        last_reason = "unknown planner failure"
        for attempt in range(self._max_attempts):
            try:
                response = self._llm.complete(
                    prompt,
                    system=system,
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001 -- bounded retry, then degrade
                last_reason = f"llm error: {exc!r}"
                continue

            cleaned = _strip_code_fences(response or "")
            if not cleaned:
                last_reason = "empty response"
                continue

            try:
                parsed = json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                last_reason = "non-JSON response"
                continue

            if not isinstance(parsed, dict):
                last_reason = "non-dict JSON"
                continue

            return self._build_plan(parsed, record)

        return self._degrade(
            question,
            f"{last_reason} after {self._max_attempts} attempt(s)",
        )

    def _build_prompt(self, record: InputRecord) -> str:
        lines = [f"Question: {record.question or ''}"]
        if record.answer_types:
            lines.append(f"Answer types: {', '.join(map(str, record.answer_types))}")
        if record.mc_options:
            if self._structured_targets:
                lines.append(
                    "Multiple-choice options: "
                    + json.dumps(record.mc_options, ensure_ascii=False, sort_keys=True)
                )
            else:
                lines.append("A multiple-choice answer scaffold is provided.")
        if record.table_schema:
            if self._structured_targets:
                lines.append(
                    "Table schema: "
                    + json.dumps(record.table_schema, ensure_ascii=False, sort_keys=True)
                )
            else:
                lines.append("A table answer scaffold is provided.")
        lines.append("Respond with the JSON object only.")
        return "\n".join(lines)

    def _build_plan(self, parsed: dict, record: InputRecord) -> Plan:
        question = record.question or ""
        raw_crit = parsed.get("criterion")
        if not isinstance(raw_crit, dict):
            raw_crit = {}

        desc = raw_crit.get("description")
        if not (isinstance(desc, str) and desc.strip()):
            desc = question

        criterion = {
            "venue": _coerce_venue(raw_crit.get("venue")),
            "year": _coerce_year(raw_crit.get("year")),
            "required_source_type": _coerce_source_type(raw_crit.get("required_source_type")),
            "description": desc,
        }

        multiplicity = parsed.get("multiplicity")
        if multiplicity not in _MULTIPLICITY_VALUES:
            multiplicity = _keyword_multiplicity(question)

        targets = (
            _coerce_targets(parsed.get("targets"))
            if self._structured_targets else []
        )
        desired_paper_count = (
            _coerce_desired_paper_count(parsed.get("desired_paper_count"))
            if self._structured_targets else None
        )

        strategies = _coerce_strategies(parsed.get("strategies"))
        # Structured source descriptions often identify a paper through an
        # interior result, equation, figure, or table rather than its title.
        # Name+dense-only plans created all of the measured candidate holes in
        # that cohort.  Always retain full-text property retrieval when the
        # structured source-unit contract is active; the downstream selector
        # still decides which candidates satisfy the source description.
        if self._structured_targets and "property" not in strategies:
            strategies.insert(strategies.index(_FALLBACK_STRATEGY), "property")

        # The source budget is the authoritative meaning of multiplicity.  A
        # model can otherwise emit count=1 together with multiplicity="multi",
        # which falls through to the operational max-papers cap later.
        if desired_paper_count is not None:
            multiplicity = "single" if desired_paper_count == 1 else "multi"

        return Plan(
            criterion=criterion,
            named_methods=_as_str_list(parsed.get("named_methods")),
            strategies=strategies,
            multiplicity=multiplicity,
            targets=targets,
            desired_paper_count=desired_paper_count,
        )
