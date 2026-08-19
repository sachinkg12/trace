"""Precision cascade for paper-set selection: staged filters that turn a wide,
junky candidate set into a PRECISE final set on the F1 metric.

WHY A CASCADE (and why the LLM is LAST). An earlier ungated design put an LLM in
the primary-filter position; it over-accepted and regressed precision (and burned
cost/reproducibility budget on every candidate). This reverses that: cheap,
DETERMINISTIC stages do the bulk of the work and the LLM adversarial validator is
the FINAL stage, seeing ONLY the residual candidates that survived the cheap
stages yet still lack strong attestation. That "LLM-call budget" -- the LLM never
touches a candidate a cheap stage already decided -- is the load-bearing property.

STAGE ORDER (each stage is a small pluggable object; OCP via injection OR the
`register_stage` registry -- adding/reordering stages never edits the cascade):

  1. DeterministicFilterStage  (cheap, first)  -- DROP a candidate that violates
     a HARD constraint from `criterion`: wrong venue / wrong year (looked up in
     PoolIndex), or NO support AND NO provenance at all. Absent constraints are
     NOT enforced (an unset venue never filters). `required_source_type` is a
     SOFT hint only (NOT a hard drop) -- the localizer's dominant modality is
     `text_span`, so hard-dropping on a missing table/figure zeroed recall; it
     now informs rerank/attestation via `_criterion_text` instead.

  2. CrossEncoderRerankStage   (optional, small sets) -- score
     (question + criterion + candidate's best RELATION/evidence text) and keep
     top-K survivors. The reranker is an INJECTED `Reranker` seam (tests pass a
     stub); the real `BgeReranker` lazy-imports sentence_transformers and is NEVER
     downloaded in tests. A reranker error DEGRADES to "skip rerank, keep all".

  3. EvidenceAttestationStage  -- TWO ROUTES TO ACCEPTANCE, no LLM:
       (a) direct answer-evidence: `ground_value` attests the criterion's answer
           value VERBATIM in the candidate's support text; OR
       (b) an explicit resolved `compares_against` relation edge (`relation_support`).
     A candidate with NEITHER is NOT rejected here -- it passes THROUGH as a
     residual to stage 4.

  4. LLMValidatorStage (LAST, residual only) -- the ONLY candidates the LLM sees
     are the residuals from stage 3. Prompted ADVERSARIALLY ("answer NO unless the
     evidence explicitly supports it"); DEFAULT-REJECT on unparseable / empty /
     error; and the returned `support_quote` must be GROUNDED in the candidate's
     own support (anti-invention) -- the LLM may not conjure evidence.

TOTAL DEGRADATION: no stage ever raises out of the cascade. A reranker error
skips rerank; an LLM error default-rejects that one candidate; a bad pool lookup
just can't confirm a violation (so it doesn't drop). Empty candidates -> empty.

Returns the accepted `paper_ids` (deduped, ordered by a simple per-route
confidence) plus a per-paper `reason` naming the stage/route that accepted it.
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from littraceqa.answer.grounding import ground_value
from littraceqa.localize.interfaces import LocatedEvidence

# --------------------------------------------------------------------------- #
# Per-route confidences (documentary; used only to ORDER the accepted set --
# strongest evidence first). Deterministic routes outrank the LLM.
# --------------------------------------------------------------------------- #
_CONF_RELATION = 0.95      # explicit resolved compares_against edge
_CONF_ATTESTED = 0.90      # answer value grounded verbatim in support
_CONF_LLM = 0.60           # residual, validated only by the adversarial LLM


# --------------------------------------------------------------------------- #
# Data model.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RouteSignal:
    """One retrieval route's evidence for a paper: the route name, this paper's
    RANK within that route's output (0-based, lower is stronger), and the route's
    best relevance SCORE for the paper (property/target-property BM25 or knn
    cosine; None for rank-only routes like name/baseline). ``group_key`` makes
    the rank local to a requested method/clause instead of a concatenated global
    rank; ``role`` says
    whether the group is an answer target, evidence anchor, or constraint.
    Exactly one signal is retained per (paper, route, group).

    The legacy selectors ignore these fields. The opt-in target-coverage selector
    reads them, and traces retain them for frozen offline replay."""
    route: str
    rank: int
    score: float | None = None
    group_key: str | None = None
    role: str = "target"


@dataclass(frozen=True)
class Candidate:
    """A paper proposed for the final set, with WHY it was proposed.

    `provenance` lists the retrieval routes that surfaced it (e.g. `dense`,
    `seed-knn`). `support` is a list of evidence dicts
    `{page, source_type, text, object_id?}` (the same shape the localizer /
    relations index emit). `relation_support`, when present, is a resolved
    `compares_against` edge (any truthy object carrying at least a `text`) --
    the second, non-LLM route to acceptance. `route_signals` is the per-route
    (rank, score, group, role) evidence preserved through the merge. Legacy
    ordering ignores it; the opt-in target-coverage selector consumes it.
    """
    paper_id: str
    provenance: list[str] = field(default_factory=list)
    support: list[dict] = field(default_factory=list)
    relation_support: object | None = None
    route_signals: list[RouteSignal] = field(default_factory=list)

    def best_support_text(self) -> str:
        """The text a reranker should score on: the RELATION/evidence text, NOT
        the abstract. Prefer the resolved relation edge, else the first support."""
        rel = _relation_text(self.relation_support)
        if rel:
            return rel
        for s in self.support:
            if s.get("text"):
                return str(s.get("text"))
        return ""

    def support_evidence(self) -> list[LocatedEvidence]:
        """Adapt `support` dicts into `LocatedEvidence` so the attestation gate
        can reuse `ground_value` unchanged (its quote IS the support text)."""
        out: list[LocatedEvidence] = []
        for s in self.support:
            out.append(LocatedEvidence(
                paper_id=self.paper_id,
                source_type=str(s.get("source_type") or ""),
                page=int(s.get("page") or 0),
                object_id=s.get("object_id"),
                quote=str(s.get("text") or ""),
                confidence=1.0,
            ))
        return out


@dataclass
class StageContext:
    """Read-only context every stage receives: the question and the parsed
    `criterion` (hard constraints + the answer value to attest)."""
    question: str
    criterion: dict


@dataclass
class StageOutcome:
    """A stage's verdict on the candidates it received.

    `accepted` are decided (paper, reason, confidence) and are removed from the
    working set -- CRUCIALLY they never reach a later stage (this is what keeps
    the LLM off already-decided candidates). `survivors` pass to the next stage.
    Anything in neither list is DROPPED.
    """
    accepted: list[tuple[Candidate, str, float]] = field(default_factory=list)
    survivors: list[Candidate] = field(default_factory=list)


@dataclass(frozen=True)
class CascadeResult:
    """Accepted `paper_ids` (deduped, ordered by descending route confidence)
    plus a per-paper `reason` naming the accepting stage/route, for debugging.

    `survivor_ids` are the FINAL survivors after every stage ran (deduped, in
    survivor order). They are what a NO-accept-gate cascade (filter/rerank-only
    ablation rung) must output -- those stages populate `survivors`, never
    `accepted`, so `paper_ids` would be empty for them."""
    paper_ids: list[str]
    reasons: dict[str, str]
    confidences: dict[str, float] = field(default_factory=dict)
    survivor_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stage protocol + registry (OCP -- a new stage registers/injects, never edits
# the cascade; mirrors the retrieval/llm/answer registries in this codebase).
# --------------------------------------------------------------------------- #
@runtime_checkable
class Stage(Protocol):
    name: str
    # ACCEPT-GATE flag: True for a stage that populates `accepted` (a precision
    # gate -- attestation/LLM); False for a survivor-only filter (filter/rerank).
    # Read via `getattr(stage, "accepts", False)` so custom stages default False.
    accepts: bool

    def process(self, candidates: list[Candidate], ctx: StageContext) -> StageOutcome: ...


_STAGES: dict[str, Callable[..., Stage]] = {}


def register_stage(name: str):
    def deco(cls):
        if name in _STAGES:
            raise ValueError(f"stage {name!r} already registered")
        _STAGES[name] = cls
        return cls
    return deco


def build_stage(name: str, **kwargs) -> Stage:
    if name not in _STAGES:
        raise KeyError(f"no stage registered as {name!r}; have {sorted(_STAGES)}")
    return _STAGES[name](**kwargs)


# --------------------------------------------------------------------------- #
# Reranker seam + a lazy, optional real implementation.
# --------------------------------------------------------------------------- #
@runtime_checkable
class Reranker(Protocol):
    """Scores each doc's relevance to `query`; higher = more relevant."""

    def score(self, query: str, docs: list[str]) -> list[float]: ...


_RERANKERS: dict[str, Callable[..., Reranker]] = {}


def register_reranker(name: str):
    def deco(cls):
        if name in _RERANKERS:
            raise ValueError(f"reranker {name!r} already registered")
        _RERANKERS[name] = cls
        return cls
    return deco


def build_reranker(name: str, **kwargs) -> Reranker:
    if name not in _RERANKERS:
        raise KeyError(f"no reranker registered as {name!r}; have {sorted(_RERANKERS)}")
    return _RERANKERS[name](**kwargs)


@register_reranker("bge")
class BgeReranker:
    """Real cross-encoder (`BAAI/bge-reranker-base`). The heavy dependency
    (`sentence_transformers`) and the model download are LAZY -- deferred to the
    first `score` call -- so importing this module (and the whole fast test
    suite) never touches the network. Tests inject a stub instead."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        model = self._ensure()
        return [float(s) for s in model.predict([(query, d) for d in docs])]


# --------------------------------------------------------------------------- #
# Stage 1: deterministic filters.
# --------------------------------------------------------------------------- #
@register_stage("deterministic-filter")
class DeterministicFilterStage:
    """DROP candidates violating a hard `criterion` constraint (venue / year /
    nothing-to-stand-on). Absent constraints are not enforced; an unverifiable
    one (no pool entry / no venue on record) does NOT drop -- we only drop on a
    DEFINITE violation. `required_source_type` is intentionally NOT hard-enforced
    here (soft signal; see module docstring)."""

    name = "deterministic-filter"
    accepts = False   # survivor-only filter, never an accept-gate

    def __init__(self, pool=None):
        self._pool = pool

    def process(self, candidates, ctx):
        crit = ctx.criterion or {}
        survivors = [c for c in candidates if not self._violates(c, crit)]
        return StageOutcome(survivors=survivors)

    def _violates(self, cand: Candidate, crit: dict) -> bool:
        # No support AND no provenance AND no RESOLVED relation edge -> nothing to
        # stand on (junk relation_support does not rescue a bare candidate).
        if (not cand.support and not cand.provenance
                and not _is_valid_relation_support(cand.relation_support)):
            return True

        paper = self._pool.by_id(cand.paper_id) if self._pool is not None else None

        want_venue = crit.get("venue")
        if want_venue and paper is not None and paper.venue:
            if _norm(str(paper.venue)) != _norm(str(want_venue)):
                return True

        want_year = crit.get("year")
        if want_year is not None and paper is not None and paper.year is not None:
            if str(paper.year).strip() != str(want_year).strip():
                return True

        # NOTE: `required_source_type` is DELIBERATELY NOT a hard-reject. The
        # localizer's dominant modality is `text_span`, so hard-dropping any
        # candidate lacking literal table/figure support silently zeroed recall
        # on gold whose evidence is a text span. It remains a SOFT signal: it is
        # still carried in the criterion and read by `_criterion_text` (rerank
        # query + adversarial-validator prompt), so the demanded modality still
        # informs ranking/attestation -- it just no longer eliminates candidates.
        # venue/year above ARE correct hard filters and are preserved.
        return False


# --------------------------------------------------------------------------- #
# Stage 2: cross-encoder rerank (optional).
# --------------------------------------------------------------------------- #
@register_stage("rerank")
class CrossEncoderRerankStage:
    """Score (question + criterion + candidate best RELATION/evidence text) and
    keep top-K. Degrades to a no-op (keep all) on any reranker error."""

    name = "rerank"
    accepts = False   # survivor-only reranker, never an accept-gate

    def __init__(self, reranker: Reranker | None = None, top_k: int | None = None):
        self._reranker = reranker
        self._top_k = top_k
        # The reranker is the ONE shared local (torch) inference component: under
        # `run_experiment(workers>1)` many question threads reach the SAME stage
        # instance and call `.score` on the SAME cross-encoder model concurrently.
        # A CrossEncoder.predict forward pass over a shared model is not guaranteed
        # thread-safe (shared tensors / non-reentrant state), so serialize it. The
        # call is fast next to the per-question Gemini I/O, so this lock costs
        # almost nothing while making outputs deterministic. Uncontended (single
        # thread) it is a no-op.
        self._score_lock = threading.Lock()

    def process(self, candidates, ctx):
        if self._reranker is None or len(candidates) <= 1:
            return StageOutcome(survivors=list(candidates))
        query = _rerank_query(ctx)
        docs = [c.best_support_text() for c in candidates]
        try:
            with self._score_lock:
                scores = self._reranker.score(query, docs)
            if len(scores) != len(candidates):
                raise ValueError("reranker returned wrong number of scores")
        except Exception:  # noqa: BLE001 -- degrade: skip rerank, keep all
            return StageOutcome(survivors=list(candidates))
        ranked = [c for _s, _i, c in sorted(
            ((s, i, c) for i, (s, c) in enumerate(zip(scores, candidates))),
            key=lambda t: (-t[0], t[1]),   # score desc, stable by original index
        )]
        keep = ranked if self._top_k is None else ranked[: self._top_k]
        return StageOutcome(survivors=keep)


# --------------------------------------------------------------------------- #
# Stage 3: evidence-attestation gate (two routes, no LLM).
# --------------------------------------------------------------------------- #
@register_stage("attestation")
class EvidenceAttestationStage:
    """Accept a candidate WITHOUT the LLM via either route; else pass it through
    as a residual for the LLM stage. Never rejects here."""

    name = "attestation"
    accepts = True   # precision accept-gate (populates `accepted`)

    def __init__(self, parsed_by_id: dict | None = None):
        self._parsed_by_id = parsed_by_id or {}

    def process(self, candidates, ctx):
        accepted: list[tuple[Candidate, str, float]] = []
        survivors: list[Candidate] = []
        answer_value = (ctx.criterion or {}).get("answer_value")
        for cand in candidates:
            # Route 2: an explicit RESOLVED compares_against edge (shape-checked;
            # bare-truthy / wrong-shape relation_support is NOT a resolved edge).
            if _is_valid_relation_support(cand.relation_support):
                accepted.append((cand, "attestation:relation-support", _CONF_RELATION))
                continue
            # Route 1: the answer value is grounded VERBATIM in the support text.
            if answer_value:
                hits = ground_value(str(answer_value), cand.support_evidence(),
                                    self._parsed_by_id)
                if hits:
                    accepted.append((cand, "attestation:answer-evidence", _CONF_ATTESTED))
                    continue
            survivors.append(cand)   # residual -> stage 4
        return StageOutcome(accepted=accepted, survivors=survivors)


# --------------------------------------------------------------------------- #
# Stage 4: LLM adversarial validator (LAST, residual only).
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@register_stage("llm-validator")
class LLMValidatorStage:
    """The ONLY candidates here are residuals from the attestation gate. Prompt
    the LLM adversarially and DEFAULT-REJECT on unparseable/empty/error. The
    returned `support_quote` must be GROUNDED in the candidate's own support
    (anti-invention): the LLM may not conjure evidence. No LLM => ACCEPT nothing
    (we cannot validate residuals).

    TERMINAL-STAGE RESIDUAL PASS-THROUGH: the candidates NOT accepted are returned
    as `survivors` (mirroring `EvidenceAttestationStage`). Nothing downstream
    consumes them -- this is the last stage -- but exposing them lets
    `CascadeResult.survivor_ids` carry the filtered-but-not-accepted candidates
    that the runner's "rank" selection policy needs for its never-empty tail. The
    `accepted` set (and thus `paper_ids` and the "gate" policy) is unchanged --
    honestly empty when the LLM accepts nothing."""

    name = "llm-validator"
    accepts = True   # precision accept-gate (populates `accepted`)

    def __init__(self, llm=None):
        self._llm = llm

    def process(self, candidates, ctx):
        accepted: list[tuple[Candidate, str, float]] = []
        if self._llm is None:
            # No validator: cannot ACCEPT any residual, but still expose them as
            # survivors (mirrors EvidenceAttestationStage) so a NO-accept downstream
            # policy -- e.g. "rank" -- can rank them. `accepted` stays empty, so the
            # honest "gate" result over paper_ids is unchanged.
            return StageOutcome(survivors=list(candidates))
        survivors: list[Candidate] = []
        for cand in candidates:
            if self._validate(cand, ctx):
                accepted.append((cand, "llm:validated", _CONF_LLM))
            else:
                survivors.append(cand)   # residual pass-through (LLM did NOT accept)
        # This is a TERMINAL stage, so no later stage consumes `survivors`; exposing
        # the non-accepted residual lets `CascadeResult.survivor_ids` carry the
        # filtered-but-not-accepted candidates the "rank" policy needs for its
        # never-empty tail. `accepted` (and thus `paper_ids`/"gate") is unchanged.
        return StageOutcome(accepted=accepted, survivors=survivors)

    def _validate(self, cand: Candidate, ctx: StageContext) -> bool:
        prompt = _build_validator_prompt(cand, ctx)
        try:
            raw = self._llm.complete(prompt, temperature=0.0)
        except Exception:  # noqa: BLE001 -- default-reject on any LLM error
            return False
        obj = _parse_json(raw)
        if not obj or obj.get("supported") is not True:
            return False
        quote = str(obj.get("support_quote") or "").strip()
        if not quote:
            return False                       # empty quote -> default-reject
        return _quote_grounded(quote, cand)    # reject invented evidence


# --------------------------------------------------------------------------- #
# The cascade.
# --------------------------------------------------------------------------- #
class PrecisionCascade:
    """Ordered pipeline of pluggable stages. Inject a custom `stages` list for
    full control, or pass deps (`pool`/`reranker`/`llm`/...) to build the default
    precision-first order. Never raises out of `select`."""

    def __init__(self, *, pool=None, reranker: Reranker | None = None,
                 llm=None, parsed_by_id: dict | None = None,
                 top_k: int | None = None, stages: list[Stage] | None = None):
        if stages is not None:
            self._stages = list(stages)
        else:
            self._stages = [
                DeterministicFilterStage(pool=pool),
                CrossEncoderRerankStage(reranker=reranker, top_k=top_k),
                EvidenceAttestationStage(parsed_by_id=parsed_by_id),
                LLMValidatorStage(llm=llm),
            ]

    @property
    def has_accept_gate(self) -> bool:
        """True if ANY configured stage is an accept-gate (populates `accepted`).
        Filter/rerank-only cascades are False; attestation/LLM cascades are True.
        Custom stages default to False unless they declare `accepts = True`."""
        return any(getattr(s, "accepts", False) for s in self._stages)

    def select(self, question: str, candidates: list[Candidate],
               criterion: dict | None = None) -> CascadeResult:
        ctx = StageContext(question=question, criterion=criterion or {})
        survivors = list(candidates or [])
        # paper_id -> (reason, confidence); keep the strongest route per paper.
        accepted: dict[str, tuple[str, float]] = {}
        for stage in self._stages:
            try:
                outcome = stage.process(survivors, ctx)
            except Exception:  # noqa: BLE001 -- a broken stage degrades to no-op
                continue
            for cand, reason, conf in outcome.accepted:
                prev = accepted.get(cand.paper_id)
                if prev is None or conf > prev[1]:
                    accepted[cand.paper_id] = (reason, conf)
            survivors = list(outcome.survivors)

        # FINAL survivors -> paper_ids, deduped in survivor order (what a
        # NO-accept-gate ablation rung outputs; ADDITIVE, independent of accepted).
        survivor_ids: list[str] = []
        seen: set[str] = set()
        for cand in survivors:
            if cand.paper_id not in seen:
                seen.add(cand.paper_id)
                survivor_ids.append(cand.paper_id)

        # Order by descending confidence, tie-break by paper_id for determinism.
        ordered = sorted(accepted.items(), key=lambda kv: (-kv[1][1], kv[0]))
        return CascadeResult(
            paper_ids=[pid for pid, _ in ordered],
            reasons={pid: r for pid, (r, _c) in ordered},
            confidences={pid: c for pid, (_r, c) in ordered},
            survivor_ids=survivor_ids,
        )


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip().casefold()


def _is_valid_relation_support(rs: object) -> bool:
    """A relation_support object is a RESOLVED-edge shape (not bare truthiness).

    Accept ONLY if `rs` is a Mapping AND it carries either:
      * a non-empty string in `text` / `support` / `support_text`, OR
      * BOTH a truthy `relation_type` AND a truthy `target` / `target_paper_id`.

    A bare `True`, `1`, a list, `{"foo": "bar"}`, or `{}` are NOT resolved edges
    -> False (the candidate is not accepted via the relation route)."""
    if not isinstance(rs, Mapping):
        return False
    for k in ("text", "support", "support_text"):
        v = rs.get(k)
        if isinstance(v, str) and v.strip():
            return True
    if rs.get("relation_type") and (rs.get("target") or rs.get("target_paper_id")):
        return True
    return False


def _relation_text(rel: object | None) -> str:
    """Best text to describe a resolved relation edge, for reranking/prompting.
    Accepts a dict (`{text|mention_text|target_paper_id}`) or an object with
    `.text` / `.mention_text` attributes; falls back to str()."""
    if not rel:
        return ""
    if isinstance(rel, dict):
        for k in ("text", "mention_text", "support_text"):
            if rel.get(k):
                return str(rel[k])
        # A bare edge with only a target still deserves a describable doc.
        tgt = rel.get("target_paper_id")
        return f"compares against {tgt}" if tgt else ""
    for k in ("text", "mention_text", "support_text"):
        v = getattr(rel, k, None)
        if v:
            return str(v)
    return str(rel)


def _criterion_text(crit: dict) -> str:
    if not crit:
        return ""
    parts = []
    for k in ("description", "venue", "year", "required_source_type", "answer_value"):
        v = crit.get(k)
        if v is not None and v != "":
            parts.append(f"{k}={v}")
    return "; ".join(parts)


def _rerank_query(ctx: StageContext) -> str:
    crit = _criterion_text(ctx.criterion)
    return f"{ctx.question} {crit}".strip()


def _parse_json(raw: str | None) -> dict | None:
    if not raw or not raw.strip():
        return None
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


_MIN_QUOTE_CHARS = 10   # ~12-char floor; the two-alpha-token rule carries the rest
_ALPHA_TOKEN_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def _is_substantive_quote(nq: str) -> bool:
    """A grounded quote must also be SUBSTANTIVE, not a trivial fragment: at least
    `_MIN_QUOTE_CHARS` chars AND >= 2 alphabetic tokens (words of >= 2 letters).
    Blocks a real-but-trivial substring like "e" from grounding acceptance."""
    if len(nq) < _MIN_QUOTE_CHARS:
        return False
    return len(_ALPHA_TOKEN_RE.findall(nq)) >= 2


def _quote_grounded(quote: str, cand: Candidate) -> bool:
    """The LLM's `support_quote` must (a) actually appear in the candidate's own
    support text (normalized, whitespace-collapsed containment -- anti-invention)
    AND (b) be SUBSTANTIVE (not a trivial fragment). No support text at all, an
    empty quote, or a non-substantive quote -> not grounded -> reject."""
    nq = _norm(re.sub(r"\s+", " ", quote))
    if not nq or not _is_substantive_quote(nq):
        return False
    texts = [str(s.get("text") or "") for s in cand.support]
    rel = _relation_text(cand.relation_support)
    if rel:
        texts.append(rel)
    for t in texts:
        if nq in _norm(re.sub(r"\s+", " ", t)):
            return True
    return False


def _build_validator_prompt(cand: Candidate, ctx: StageContext) -> str:
    lines = [
        "You are an ADVERSARIAL validator checking a candidate paper for a "
        "precision-critical answer set.",
        f"Question: {ctx.question}",
        f"Criterion the paper must REALLY satisfy: {_criterion_text(ctx.criterion) or '(none given)'}",
        f"Candidate paper_id: {cand.paper_id}",
        "",
        "Evidence (the ONLY evidence you may use -- do NOT invent any):",
    ]
    if cand.support:
        for s in cand.support:
            src = s.get("source_type") or "?"
            page = s.get("page")
            lines.append(f"- [{src} p.{page}] {s.get('text') or ''}")
    else:
        lines.append("- (no direct support)")
    rel = _relation_text(cand.relation_support)
    if rel:
        lines.append(f"- [relation] {rel}")
    lines += [
        "",
        "Answer NO unless the evidence EXPLICITLY supports that this paper "
        "satisfies the criterion. Do not invent evidence; the support_quote MUST "
        "be copied verbatim from the evidence above.",
        'Return ONLY JSON: {"supported": bool, "reason": string, "support_quote": string}',
    ]
    return "\n".join(lines)
