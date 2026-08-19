"""Retrieval-strategy REGISTRY + index-backed strategies + a merging router.

This is the seam between the `QuestionPlanner` and the `PrecisionCascade`: it
turns a `Plan` into a deduped list of `Candidate`s drawn from the full-corpus
Level-2 indexes (passages / objects / aliases / relations). The Planner decides
WHICH routes to run (`plan.strategies`); this module RUNS them and merges their
output into exactly the `list[Candidate]` the cascade's `select` consumes.

OCP by the house pattern (mirrors `retrieval.interfaces` + `paperset.cascade`):
each strategy registers itself under a name via `@register_strategy`; the router
dispatches through `RETRIEVAL_STRATEGIES` and never edits when a new strategy is
added. Concrete strategies are named ONLY at the composition root (build.py); the
router is closed for modification. A requested-but-UNREGISTERED strategy is
SKIPPED with a log, never an error, so an experimental plan can name a route this
build does not yet implement.

DEGRADATION: a strategy whose backing index is absent returns `[]` (BaselineStrategy
with `relations=None`; a strategy needing `passages`/`aliases` when they are None;
DenseKnnStrategy with `dense=None`). No strategy raises out of the router; a broken
strategy is skipped + logged.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from littraceqa.paperset.cascade import Candidate, RouteSignal
from littraceqa.pipeline.planner import Plan, PlanTarget
from littraceqa.retrieval.exact import squash_title

_log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Bounds (documented caps that keep a wide-net strategy from amplifying junk --
# the F1 lesson from the reverted baseline-expander build).
# --------------------------------------------------------------------------- #
# NameStrategy: cap alias resolves PER METHOD. `aliases.resolve` is already ranked
# (paper_specific + more-distinctive first), so the top-N are the highest-precision
# links; the tail is where generic-term collisions live. 20 is generous headroom
# for a genuinely multi-paper coinage while bounding a generic acronym's blast.
_NAME_MAX_RESOLVES_PER_METHOD = 20
# NameStrategy support fallback: how many BM25 hits to scan for a passage on the
# resolved paper when the alias record carries no usable evidence sentence.
_NAME_SUPPORT_SEARCH_K = 10

# PropertyStrategy: BM25 breadth over the full-text KB, and the max chunks kept as
# support per surfaced paper (best-scoring first; more than a few adds no signal).
DEFAULT_PROPERTY_SEARCH_K = 30
_PROPERTY_SEARCH_K = DEFAULT_PROPERTY_SEARCH_K  # backward-compatible test seam
_PROPERTY_MAX_CHUNKS_PER_PAPER = 3

# CitationStrategy: search anchor mentions broadly, then group the best passages
# by the citing paper. It is activated only by structured evidence-anchor targets.
_CITATION_SEARCH_K = 60
_CITATION_MAX_CHUNKS_PER_PAPER = 3

# ObjectStrategy: search the table/figure-caption index broadly enough to surface
# answer-bearing papers that title/abstract and passage retrieval can miss.  The
# per-paper cap keeps repeated captions from one paper from crowding the support
# carried into ranking and diagnostics.
_OBJECT_SEARCH_K = 60
_OBJECT_MAX_CAPTIONS_PER_PAPER = 3

# DenseKnnStrategy: default top-K papers pulled by the semantic ("search by
# meaning") kNN route. 30 mirrors PropertyStrategy's BM25 breadth -- generous
# recall while the precision cascade downstream does the gating (the F1 lesson:
# wide net is safe ONLY behind a precision gate).
_KNN_K = 30


@dataclass(frozen=True)
class _TitleSurfaceMatch:
    paper_id: str
    surface: str
    kind: str
    exact_match_count: int
    surface_match_count: int


# --------------------------------------------------------------------------- #
# Context every strategy reads (the indexes + optional dense/pool + question).
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalContext:
    """Read-only bundle a strategy searches over.

    `passages`/`objects`/`aliases`/`relations` come straight from a `LoadedIndexes`
    (`relations` may be None on a pool-less build). `dense`/`pool` are OPTIONAL --
    strategies that need them (a future knn route) degrade to `[]` when absent.
    `question` is the raw question text, threaded in by the router so a strategy
    can fall back to it when the plan carries no usable description.
    """

    passages: object | None = None
    objects: object | None = None
    aliases: object | None = None
    relations: object | None = None
    dense: object | None = None
    pool: object | None = None
    question: str = ""
    property_search_k: int = DEFAULT_PROPERTY_SEARCH_K

    @classmethod
    def from_indexes(
        cls,
        indexes,
        *,
        dense=None,
        pool=None,
        question: str = "",
        property_search_k: int = DEFAULT_PROPERTY_SEARCH_K,
    ) -> "RetrievalContext":
        """Build from a `LoadedIndexes` (or any object exposing the four index
        attributes), keeping the composition root free of field-by-field wiring."""
        return cls(
            passages=getattr(indexes, "passages", None),
            objects=getattr(indexes, "objects", None),
            aliases=getattr(indexes, "aliases", None),
            relations=getattr(indexes, "relations", None),
            dense=dense,
            pool=pool,
            question=question,
            property_search_k=property_search_k,
        )


# --------------------------------------------------------------------------- #
# Strategy Protocol + registry (OCP -- add a strategy by registering, never edit).
# --------------------------------------------------------------------------- #
@runtime_checkable
class RetrievalStrategy(Protocol):
    name: str

    def search(self, plan: Plan, ctx: RetrievalContext) -> list[Candidate]: ...


# name -> a stateless strategy INSTANCE (singleton). Strategies hold no per-run
# state; the context carries everything, so one instance is reused per name.
RETRIEVAL_STRATEGIES: dict[str, RetrievalStrategy] = {}


def register_strategy(name: str):
    """Decorator: instantiate `cls` (no args) and register the singleton under
    `name`. The class is returned unchanged so it stays importable/testable."""

    def deco(cls):
        if name in RETRIEVAL_STRATEGIES:
            raise ValueError(f"strategy {name!r} already registered")
        RETRIEVAL_STRATEGIES[name] = cls()
        return cls

    return deco


def get_strategy(name: str) -> RetrievalStrategy:
    if name not in RETRIEVAL_STRATEGIES:
        raise KeyError(
            f"no strategy registered as {name!r}; have {sorted(RETRIEVAL_STRATEGIES)}"
        )
    return RETRIEVAL_STRATEGIES[name]


# --------------------------------------------------------------------------- #
# Strategy: "name" -- resolve each named method to its paper(s) via the aliases
# index (corpus-mined acronyms / coined names).
# --------------------------------------------------------------------------- #
@register_strategy("name")
class NameStrategy:
    """For every method the question NAMES, resolve its alias to paper_id(s) via
    `aliases.resolve` (already ranked: paper_specific + distinctive first). Cap at
    `_NAME_MAX_RESOLVES_PER_METHOD` per method to bound a generic acronym's junk.

    A structured answer target also gets a deterministic whole-surface title
    lookup before the mined alias index.  This covers paper names such as
    ``ARCHWAY`` that appear literally in the title but are absent from (or ranked
    poorly by) the body-derived alias index.  Title matches are identity evidence,
    so they must not lose to a nearby paper with a slightly higher property-BM25
    score.  Constraint/evidence-anchor names deliberately remain alias-only."""

    name = "name"

    def search(self, plan: Plan, ctx: RetrievalContext) -> list[Candidate]:
        if ctx.aliases is None and ctx.pool is None:
            return []
        out: list[Candidate] = _question_source_title_candidates(plan, ctx)
        for method in plan.named_methods:
            if not (isinstance(method, str) and method.strip()):
                continue
            group_key, role = _target_group(plan, method)
            title_matches: list[_TitleSurfaceMatch] = []
            seen_title_pids: set[str] = set()
            if role == "target":
                for variant_index, surface in enumerate(
                    _title_surface_variants(method)
                ):
                    surface_matches = _title_surface_matches(surface, ctx.pool)
                    # A shortened planner alias ("NAME benchmark" -> "NAME")
                    # is useful only as a unique title identity.  Broad suffix
                    # removal must never flood the shared 20-paper name cap.
                    if variant_index and len(surface_matches) != 1:
                        continue
                    for match in surface_matches:
                        if match.paper_id in seen_title_pids:
                            continue
                        seen_title_pids.add(match.paper_id)
                        title_matches.append(match)
            strong_title_pids = [
                match.paper_id for match in title_matches
                if (
                    match.kind == "exact" and match.exact_match_count == 1
                ) or match.surface_match_count == 1
            ]
            strong_title_pid_set = set(strong_title_pids)
            weak_title_pids = [
                match.paper_id for match in title_matches
                if match.paper_id not in strong_title_pid_set
            ]
            title_match_by_pid = {
                match.paper_id: match for match in title_matches
            }
            resolve = getattr(ctx.aliases, "resolve", None)
            alias_pids = list(resolve(method)) if callable(resolve) else []
            # Strong/unique title identities lead. Alias-index owners are
            # reserved ahead of ambiguous interior-title mentions, which can
            # otherwise consume the entire per-method cap.
            pids = list(dict.fromkeys([
                *strong_title_pids, *alias_pids, *weak_title_pids,
            ]))[:_NAME_MAX_RESOLVES_PER_METHOD]
            title_pids = [match.paper_id for match in title_matches]
            title_pid_set = set(title_pids)
            for local_rank, pid in enumerate(pids):
                # A name candidate MUST carry a real evidence `text`: the cascade's
                # attestation / adversarial-validator stages read `support[].text`,
                # and an empty one makes `best_support_text()` "" -> the candidate
                # can never ground and the precision gate silently drops any paper
                # found ONLY by name (this zeroed body-coinage recall). Fill it.
                text, page = _name_support(method, pid, ctx)
                source = "alias"
                if pid in title_pid_set:
                    paper = ctx.pool.by_id(pid)
                    text = paper.title
                    page = None
                    source = "title_surface"
                title_match = title_match_by_pid.get(pid)
                support = {
                    "source": source,
                    "alias": title_match.surface if title_match else method,
                    "paper_id": pid,
                    "text": text, "page": page,
                }
                if title_match is not None:
                    support.update({
                        "group_key": group_key,
                        "role": role,
                        "title_match_kind": title_match.kind,
                        "title_match_count": title_match.exact_match_count,
                        "title_surface_match_count": (
                            title_match.surface_match_count
                        ),
                    })
                out.append(Candidate(
                    paper_id=pid,
                    provenance=["name"],
                    support=[support],
                    route_signals=[RouteSignal(
                        route="name", rank=local_rank, score=None,
                        group_key=group_key, role=role,
                    )],
                ))
        return out


def _target_group(plan: Plan, text: str) -> tuple[str, str]:
    """Map a named entity to its planner target, preserving legacy fallback."""
    needle = text.strip().casefold()
    for target in getattr(plan, "targets", ()):
        target_text = str(getattr(target, "text", "") or "").strip().casefold()
        target_key = str(getattr(target, "key", "") or "").strip()
        # Planner surface forms are not always byte-identical: named_methods may
        # contain both "MCTS" and "Monte Carlo Tree Search" while the structured
        # target is "MCTS (Monte Carlo Tree Search)". Containment maps both to the
        # same semantic group; exact key equality remains the strongest case.
        if needle and (
            needle == target_key.casefold()
            or needle == target_text
            or _contains_surface(target_text, needle)
            or _contains_surface(needle, target_text)
        ):
            return target_key or text.strip(), getattr(target, "role", "target")
    return text.strip(), "target"


def _contains_surface(text: str, surface: str) -> bool:
    """Whole surface-form containment; avoids ``ATT`` matching a longer token."""
    if not text or not surface:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])", text
    ) is not None


_SOURCE_OWNER_SUFFIXES = ("evaluation", "paper", "study", "work")
_SOURCE_OWNER_NEGATIVE_PREFIX_RE = re.compile(
    r"(?:\bcites?\s+(?:the\s+)?|\bciting\s+(?:the\s+)?|"
    r"\bcitation\s+(?:to\s+)?(?:the\s+)?|\bbaseline\s+(?:the\s+)?|"
    r"\b(?:compares?|compared?)\s+(?:with|against|to)\s+(?:the\s+)?|"
    r"\bcomparison\s+(?:with|against|to)\s+(?:the\s+)?|"
    r"\bagainst\s+(?:the\s+)?|"
    r"\b(?:using|uses|based\s+on|extends?)\s+(?:the\s+)?|"
    r"\b(?:prior|previous)\s+(?:the\s+)?|"
    r"\b(?:versus|vs\.?)\s+(?:the\s+)?)$",
    re.IGNORECASE,
)
_GENERIC_SOURCE_OWNER_SURFACES = frozenset({
    "analysis", "benchmark", "evaluation", "experiment", "framework",
    "method", "model", "paper", "study", "system", "work",
})

# Descriptive reporting-source recovery is deliberately separate from generic
# property retrieval.  It reads only compact source phrases that the question
# itself presents as a paper/study/work/framework/analysis/evaluation/pipeline,
# then asks whether one full-pool title+abstract is a decisive lexical owner.
# The winning paper must already have been retrieved by an ordinary strategy;
# this block can annotate/reorder candidates but can never expand the set.
_CLAUSE_OWNER_SPLIT_RE = re.compile(
    r"[,;?]|\b(?:while|whereas)\b|\band\s+(?=(?:in|from|the|an?)\b)",
    re.IGNORECASE,
)
_CLAUSE_OWNER_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CLAUSE_OWNER_STOPWORDS = frozenset({
    "a", "an", "another", "according", "analysis", "approach", "at", "by",
    "baseline", "comparing", "depicts", "describes", "evaluation", "for", "framework",
    "from", "in", "introduces", "method", "model", "of", "on", "one", "paper",
    "pipeline", "previous", "prior", "related", "reported", "reporting", "reports", "source", "stated", "study",
    "system", "the", "these", "this", "those", "to", "using", "whose", "with",
    "work",
})
_CLAUSE_OWNER_NEGATIVE_RE = re.compile(
    r"\b(?:cites?|cited|citing|citation)\b|"
    r"\b(?:unlike|versus|against)\b|"
    r"\b(?:compared?|comparison)\s+(?:to|with|against)\b|"
    r"\b(?:in\s+contrast|relative)\s+to\b|"
    r"\b(?:using|with)\b|"
    r"\b(?:building|based|derived)\s+(?:on|from)\b|"
    r"\b(?:extends?|improves?)\s+(?:on|upon)\b|"
    r"\b(?:baseline|prior|previous|related)\s+"
    r"(?:analysis|evaluation|framework|paper|pipeline|study|work)\b",
    re.IGNORECASE,
)
_CLAUSE_OWNER_MIN_TERMS = 4
_CLAUSE_OWNER_MAX_TERMS = 8


@dataclass(frozen=True)
class _ClauseOwnerPhrase:
    text: str
    clause: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class _ClauseOwnerDoc:
    paper_id: str
    title: str
    abstract: str
    title_sequence: str
    abstract_sequence: str
    title_terms: frozenset[str]
    document_terms: frozenset[str]


@dataclass(frozen=True)
class _ClauseOwnerMatch:
    paper_id: str
    target_key: str
    phrase: _ClauseOwnerPhrase
    score: int
    runner_up_score: int


_CLAUSE_OWNER_POOL_CACHE: WeakKeyDictionary = WeakKeyDictionary()


def _clause_owner_term(value: str) -> str:
    """Lightly normalize lexical variants without fuzzy semantic expansion."""
    token = value.casefold()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _clause_owner_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    for raw in _CLAUSE_OWNER_TOKEN_RE.findall(str(value or "").casefold()):
        term = _clause_owner_term(raw)
        if (
            raw in _CLAUSE_OWNER_STOPWORDS
            or term in _CLAUSE_OWNER_STOPWORDS
            or term.isdigit()
            or len(term) < 2
        ):
            continue
        if term not in terms:
            terms.append(term)
    return tuple(terms)


def _question_clause_owner_phrases(question: str) -> list[_ClauseOwnerPhrase]:
    """Extract compact, singular reporting-container phrases from clauses."""
    phrases: list[_ClauseOwnerPhrase] = []
    for raw_clause in _CLAUSE_OWNER_SPLIT_RE.split(str(question or "")):
        clause = " ".join(raw_clause.split()).strip(" .:-")
        if not clause or _CLAUSE_OWNER_NEGATIVE_RE.search(clause):
            continue
        matches = list(re.finditer(
            r"\b(?:analysis|evaluation|framework|paper|pipeline|study|work)\b",
            clause,
            flags=re.IGNORECASE,
        ))
        for match in matches:
            prefix = clause[:match.start()]
            raw_terms = _clause_owner_terms(prefix)
            terms = raw_terms[-_CLAUSE_OWNER_MAX_TERMS:]
            if len(terms) < _CLAUSE_OWNER_MIN_TERMS:
                continue
            phrase_start = max(0, match.start() - 120)
            phrase_text = clause[phrase_start:match.end()].strip()
            phrases.append(_ClauseOwnerPhrase(
                text=phrase_text,
                clause=clause,
                terms=terms,
            ))
    return phrases


def _clause_owner_pool_docs(pool: object) -> tuple[_ClauseOwnerDoc, ...]:
    """Cache immutable title+abstract lexical features for one pool object."""
    try:
        cached = _CLAUSE_OWNER_POOL_CACHE.get(pool)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    ids = getattr(pool, "ids", None)
    by_id = getattr(pool, "by_id", None)
    if not isinstance(ids, list) or not callable(by_id):
        return ()
    docs: list[_ClauseOwnerDoc] = []
    for paper_id in ids:
        paper = by_id(paper_id)
        if paper is None:
            continue
        title = str(getattr(paper, "title", "") or "").strip()
        abstract = str(getattr(paper, "abstract", "") or "").strip()
        title_terms = frozenset(_clause_owner_terms(title))
        docs.append(_ClauseOwnerDoc(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            title_sequence=" ".join(
                _CLAUSE_OWNER_TOKEN_RE.findall(title.casefold())
            ),
            abstract_sequence=" ".join(
                _CLAUSE_OWNER_TOKEN_RE.findall(abstract.casefold())
            ),
            title_terms=title_terms,
            document_terms=frozenset({
                *title_terms, *_clause_owner_terms(abstract),
            }),
        ))
    result = tuple(docs)
    try:
        _CLAUSE_OWNER_POOL_CACHE[pool] = result
    except TypeError:
        pass
    return result


# Exact named-source recovery complements the descriptive clause-owner route.
# It covers target names whose canonical paper title uses an expansion (for
# example an acronym defined only in the abstract).  Every gate is deliberately
# local and deterministic: the surface comes from the question/plan, ownership
# is unique over the full pool, and an ordinary target-property query must have
# already retrieved the paper near the head.  This route can annotate but never
# add a candidate.
_SELF_DEFINITION_OWNER_SUFFIXES = frozenset({
    "algorithm", "approach", "benchmark", "evaluation", "framework",
    "method", "model", "paper", "pipeline", "study", "system", "work",
})
_SELF_DEFINITION_OWNER_GENERIC = frozenset({
    "adam", "adamw", "algorithm", "approach", "benchmark", "evaluation",
    "framework", "general", "improved", "lora", "method", "model", "new",
    "novel", "paper", "pipeline", "rmsprop", "sgd", "study", "system",
    "work",
})
_SELF_DEFINITION_OWNER_NEGATIVE_CLAUSE_RE = re.compile(
    r"\b(?:cites?|cited|citing|citation|baseline|prior|previous)\b|"
    r"\b(?:using|uses|based\s+on|extends?|outperforms?)\b|"
    r"\b(?:builds?|building)\s+(?:on|upon)\b|"
    r"\b(?:compares?|compared?|comparison)\s+(?:with|against|to)\b|"
    r"\b(?:unlike|versus|against|contrast)\b",
    re.IGNORECASE,
)
_SELF_DEFINITION_OWNER_FOLLOWING_RELATION_RE = re.compile(
    r"\b(?:cites?|cited|citing|outperforms?|extends?)\s+"
    r"(?:it|this|that|the\s+(?:approach|framework|method|model|paper|pipeline|"
    r"study|system|work))\b|"
    r"\b(?:builds?|building)\s+(?:on|upon)\s+"
    r"(?:it|this|that|the\s+(?:approach|framework|method|model|paper|pipeline|"
    r"study|system|work))\b|"
    r"\b(?:unlike|versus)\s+"
    r"(?:it|this|that|the\s+(?:approach|framework|method|model|paper|pipeline|"
    r"study|system|work))\b|"
    r"\b(?:in\s+contrast|compared?)\s+to\s+"
    r"(?:it|this|that|the\s+(?:approach|framework|method|model|paper|pipeline|"
    r"study|system|work))\b",
    re.IGNORECASE,
)
_SELF_DEFINITION_TITLE_NEGATIVE_RE = re.compile(
    r"\busing\b|\b(?:a\s+)?comparison\s+with\b|"
    r"\b(?:standard\s+)?baseline\b|\brevisit(?:ing|ed|s)?\b|"
    r"\b(?:based|building)\s+(?:on|upon)\b|\bextends?\b",
    re.IGNORECASE,
)
_SELF_DEFINITION_OWNER_CUE_RE = re.compile(
    r"\b(?:we|this\s+paper)\s+(?:first\s+|further\s+)?"
    r"(?:introduce|propose|present|develop)\b",
    re.IGNORECASE,
)
_SELF_DEFINITION_OWNER_MAX_PROPERTY_RANK = 2


def _sequence_text(value: str) -> str:
    return " ".join(
        _CLAUSE_OWNER_TOKEN_RE.findall(str(value or "").casefold())
    )


def _contains_sequence(text: str, surface: str) -> bool:
    return bool(surface) and f" {surface} " in f" {text} "


def _self_definition_surface(method: str) -> str:
    """Normalize a named method and strip at most one generic head noun."""
    tokens = _sequence_text(method).split()
    if len(tokens) > 1 and tokens[-1] in _SELF_DEFINITION_OWNER_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _self_definition_target(
    method: str, surface: str, targets: list[PlanTarget]
) -> PlanTarget | None:
    """Map an explicit named surface to exactly one structured target."""
    method_sequence = _sequence_text(method)
    winners: list[PlanTarget] = []
    for target in targets:
        target_text = _sequence_text(str(getattr(target, "text", "") or ""))
        target_key = _sequence_text(str(getattr(target, "key", "") or ""))
        if any(
            candidate and (
                candidate == target_key
                or candidate == target_text
                or _contains_sequence(target_text, candidate)
            )
            for candidate in (method_sequence, surface)
        ):
            winners.append(target)
    return winners[0] if len(winners) == 1 else None


def _question_allows_self_definition_owner(
    question: str, surface: str
) -> bool:
    """Require positive ownership syntax and reject dependency/contrast clauses."""
    question_text = str(question or "")
    if not question_text or not surface:
        return False
    surface_pattern = r"\W+".join(
        re.escape(token) for token in surface.split()
    )
    matches = list(re.finditer(
        rf"(?<![A-Za-z0-9]){surface_pattern}(?![A-Za-z0-9])",
        question_text,
        flags=re.IGNORECASE,
    ))
    if not matches:
        return False
    if _SELF_DEFINITION_OWNER_FOLLOWING_RELATION_RE.search(question_text):
        return False
    positive = False
    for match in matches:
        clause_start = max(
            question_text.rfind(mark, 0, match.start()) for mark in ",;?"
        ) + 1
        clause_ends = [
            position for mark in ",;?"
            if (position := question_text.find(mark, match.end())) >= 0
        ]
        clause_end = min(clause_ends, default=len(question_text))
        clause = question_text[clause_start:clause_end]
        if _SELF_DEFINITION_OWNER_NEGATIVE_CLAUSE_RE.search(clause):
            return False
        after = question_text[match.end():match.end() + 48]
        if re.match(
            r"(?:['’]s\b|\s+(?:algorithm|approach|benchmark|evaluation|"
            r"framework|method|model|paper|pipeline|study|system|work)\b)",
            after,
            flags=re.IGNORECASE,
        ):
            positive = True
    return positive


def _self_definition_kind(
    doc: _ClauseOwnerDoc, surface: str
) -> str | None:
    """Return the paper's explicit ownership form, or None for mere mentions."""
    if _contains_sequence(doc.title_sequence, surface):
        if _SELF_DEFINITION_TITLE_NEGATIVE_RE.search(doc.title):
            return None
        return "exact_title"
    surface_pattern = r"\W+".join(
        re.escape(token) for token in surface.split()
    )
    if re.search(
        _SELF_DEFINITION_OWNER_CUE_RE.pattern
        + rf"[^.;:!?]{{0,160}}(?<![A-Za-z0-9]){surface_pattern}"
        + r"(?![A-Za-z0-9])",
        doc.abstract,
        flags=re.IGNORECASE,
    ):
        return "abstract_self_definition"
    return None


def _definition_support_text(
    doc: _ClauseOwnerDoc, surface: str, definition_kind: str
) -> str:
    if definition_kind == "exact_title" or not doc.abstract:
        return doc.title
    # Preserve readable source text while bounding dossier/prompt size.
    normalized_surface = surface.replace(" ", r"\W+")
    match = re.search(normalized_surface, doc.abstract, flags=re.IGNORECASE)
    if match is None:
        return doc.abstract[:320]
    start = max(0, match.start() - 180)
    end = min(len(doc.abstract), match.end() + 180)
    return doc.abstract[start:end].strip()


def _question_self_definition_owner_candidates(
    plan: Plan,
    ctx: RetrievalContext,
    existing_signals: dict[str, tuple[RouteSignal, ...]],
) -> list[Candidate]:
    """Annotate globally unique named owners with independent local support."""
    targets = [target for target in plan.targets if target.role == "target"]
    if (
        not targets
        or plan.desired_paper_count != len(targets)
        or not existing_signals
        or ctx.pool is None
        or not str(ctx.question or "").strip()
    ):
        return []
    docs = _clause_owner_pool_docs(ctx.pool)
    if not docs:
        return []

    matches_by_target: dict[str, list[tuple[_ClauseOwnerDoc, str, str]]] = {}
    for method in plan.named_methods:
        if not isinstance(method, str) or not method.strip():
            continue
        surface = _self_definition_surface(method)
        tokens = surface.split()
        informative_tokens = [
            token for token in tokens
            if (
                token not in _SELF_DEFINITION_OWNER_GENERIC
                and not token.isdigit()
                and len(token) >= 2
            )
        ]
        if (
            len(tokens) < 2
            or len(informative_tokens) < 2
            or len("".join(tokens)) < 5
            or not _question_allows_self_definition_owner(ctx.question, surface)
        ):
            continue
        target = _self_definition_target(method, surface, targets)
        if target is None:
            continue
        owners = [
            doc for doc in docs
            if _contains_sequence(doc.title_sequence, surface)
            or _contains_sequence(doc.abstract_sequence, surface)
        ]
        if len(owners) != 1:
            continue
        owner = owners[0]
        definition_kind = _self_definition_kind(owner, surface)
        if definition_kind is None:
            continue
        target_signals = [
            signal for signal in existing_signals.get(owner.paper_id, ())
            if (
                signal.route == "target_property"
                and signal.role == "target"
                and isinstance(signal.group_key, str)
                and signal.group_key.strip().casefold()
                == target.key.strip().casefold()
                and signal.rank <= _SELF_DEFINITION_OWNER_MAX_PROPERTY_RANK
            )
        ]
        if not target_signals:
            continue
        matches_by_target.setdefault(target.key.casefold(), []).append(
            (owner, surface, definition_kind)
        )

    agreed: list[tuple[str, _ClauseOwnerDoc, str, str]] = []
    for target_key, target_matches in matches_by_target.items():
        if len({owner.paper_id for owner, _surface, _kind in target_matches}) != 1:
            continue
        # Prefer the longest exact surface when two planner names identify the
        # same owner; this affects trace readability only, not selection.
        owner, surface, definition_kind = max(
            target_matches, key=lambda item: len(item[1])
        )
        agreed.append((target_key, owner, surface, definition_kind))
    if len({owner.paper_id for _key, owner, _surface, _kind in agreed}) != len(agreed):
        return []

    return [
        Candidate(
            paper_id=owner.paper_id,
            provenance=["self_definition_owner"],
            support=[{
                "source": "metadata_definition",
                "alias": surface,
                "paper_id": owner.paper_id,
                "text": _definition_support_text(owner, surface, definition_kind),
                "page": None,
                "group_key": target_key,
                "role": "target",
                "title_match_kind": "self_definition",
                "title_match_count": 1,
                "title_surface_match_count": 1,
                "question_self_definition_owner": True,
                "definition_kind": definition_kind,
                "owner_corpus_match_count": 1,
            }],
            route_signals=[RouteSignal(
                route="self_definition_owner", rank=0, score=None,
                group_key=target_key, role="target",
            )],
        )
        for target_key, owner, surface, definition_kind in agreed
    ]


def _clause_owner_target(
    phrase: _ClauseOwnerPhrase, targets: list[PlanTarget]
) -> PlanTarget | None:
    """Map a source phrase to exactly one target; ties fail closed."""
    if len(targets) == 1:
        return targets[0]
    phrase_terms = set(phrase.terms)
    scored = [
        (len(phrase_terms.intersection(_clause_owner_terms(
            str(getattr(target, "text", "") or "")
        ))), target)
        for target in targets
    ]
    best = max((score for score, _target in scored), default=0)
    winners = [target for score, target in scored if score == best]
    if best < 3 or len(winners) != 1:
        return None
    return winners[0]


def _question_clause_owner_candidates(
    plan: Plan,
    ctx: RetrievalContext,
    existing_paper_ids: set[str],
) -> list[Candidate]:
    """Annotate decisive descriptive owners already present in candidates.

    Matching is global over the pool, not merely relative to the retrieved
    subset. A match needs strong title/abstract coverage, a strict score margin,
    unique ownership per target, and an already-retrieved winner. Ambiguity or a
    missing winner returns no signal instead of allowing pool/candidate order to
    decide. Candidate cardinality and order are therefore invariant.
    """
    targets = [target for target in plan.targets if target.role == "target"]
    if (
        not targets
        or not existing_paper_ids
        or ctx.pool is None
        or not str(ctx.question or "").strip()
    ):
        return []
    docs = _clause_owner_pool_docs(ctx.pool)
    if not docs:
        return []

    matches: list[_ClauseOwnerMatch] = []
    for phrase in _question_clause_owner_phrases(ctx.question):
        target = _clause_owner_target(phrase, targets)
        if target is None:
            continue
        phrase_terms = set(phrase.terms)
        scored: list[tuple[int, int, int, _ClauseOwnerDoc]] = []
        for doc in docs:
            title_hits = len(phrase_terms.intersection(doc.title_terms))
            document_hits = len(phrase_terms.intersection(doc.document_terms))
            score = 4 * title_hits + (document_hits - title_hits)
            scored.append((score, title_hits, document_hits, doc))
        scored.sort(key=lambda item: (-item[0], item[3].paper_id))
        best_score, title_hits, document_hits, best_doc = scored[0]
        runner_up_score = scored[1][0] if len(scored) > 1 else 0
        required_hits = max(3, math.ceil(len(phrase_terms) * 0.65))
        if (
            document_hits < required_hits
            or (title_hits < 2 and document_hits < 4)
            or best_score - runner_up_score < 2
            or best_doc.paper_id not in existing_paper_ids
        ):
            continue
        matches.append(_ClauseOwnerMatch(
            paper_id=best_doc.paper_id,
            target_key=str(getattr(target, "key", "") or ""),
            phrase=phrase,
            score=best_score,
            runner_up_score=runner_up_score,
        ))

    # Multiple phrases may describe the same target. They are usable only when
    # all decisive matches agree on its owner. For multi-paper plans, one paper
    # also cannot silently consume two nominally distinct source targets.
    by_target: dict[str, list[_ClauseOwnerMatch]] = {}
    for match in matches:
        by_target.setdefault(match.target_key.casefold(), []).append(match)
    agreed: list[_ClauseOwnerMatch] = []
    for target_matches in by_target.values():
        if len({match.paper_id for match in target_matches}) != 1:
            continue
        agreed.append(max(target_matches, key=lambda match: match.score))
    if plan.desired_paper_count != 1:
        paper_counts: dict[str, int] = {}
        for match in agreed:
            paper_counts[match.paper_id] = paper_counts.get(match.paper_id, 0) + 1
        agreed = [
            match for match in agreed if paper_counts[match.paper_id] == 1
        ]

    doc_by_id = {doc.paper_id: doc for doc in docs}
    return [
        Candidate(
            paper_id=match.paper_id,
            provenance=["clause_owner"],
            support=[{
                "source": "title_surface",
                "alias": match.phrase.text,
                "paper_id": match.paper_id,
                "text": doc_by_id[match.paper_id].title,
                "page": None,
                "group_key": match.target_key,
                "role": "target",
                "title_match_kind": "descriptive",
                "title_match_count": 1,
                "title_surface_match_count": 1,
                "question_clause_owner": True,
                "owner_clause": match.phrase.clause,
                "owner_score": match.score,
                "owner_runner_up_score": match.runner_up_score,
            }],
            route_signals=[RouteSignal(
                route="clause_owner", rank=0, score=None,
                group_key=match.target_key, role="target",
            )],
        )
        for match in agreed
    ]


def _question_marks_source_owner(question: str, surface: str) -> bool:
    """Whether the question explicitly presents ``surface`` as its source.

    The detector intentionally requires a reporting-container cue. Merely
    mentioning a method or saying that another paper cites it is not ownership.
    This keeps raw-question title recovery narrower than general alias lookup.
    """
    question_text = str(question or "")
    candidate_surface = str(surface or "").strip()
    if not question_text or not candidate_surface:
        return False
    escaped = re.escape(candidate_surface)
    token = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    suffix = "|".join(re.escape(item) for item in _SOURCE_OWNER_SUFFIXES)
    for match in re.finditer(token, question_text, flags=re.IGNORECASE):
        before = question_text[max(0, match.start() - 64):match.start()]
        if _SOURCE_OWNER_NEGATIVE_PREFIX_RE.search(before):
            continue
        after = question_text[match.end():match.end() + 40]
        if re.match(
            rf"(?:['’]s)?\s+(?:{suffix})\b", after, flags=re.IGNORECASE
        ):
            return True
        if re.search(
            r"(?:according\s+to|reported\s+(?:in|by)|"
            r"presented\s+(?:in|by)|results?\s+from)\s+(?:the\s+)?$",
            before,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _question_source_title_candidates(
    plan: Plan, ctx: RetrievalContext
) -> list[Candidate]:
    """Recover a uniquely named reporting study directly from the question.

    Planner ``named_methods`` often contains the baseline rows whose values are
    requested but omits the study/table that owns those rows. For a one-source
    plan only, scan deterministic title heads (the segment before ``:``) and
    full titles for an explicit reporting cue such as ``NAME evaluation``.
    Ambiguous surfaces and cited/baseline contexts are rejected.
    """
    targets = [target for target in plan.targets if target.role == "target"]
    if (
        plan.desired_paper_count != 1
        or len(targets) != 1
        or ctx.pool is None
        or not str(ctx.question or "").strip()
    ):
        return []
    ids = getattr(ctx.pool, "ids", None)
    by_id = getattr(ctx.pool, "by_id", None)
    if not isinstance(ids, list) or not callable(by_id):
        return []

    # Most pool titles do not occur in the question.  A literal prefilter keeps
    # the boundary/context regex below to only a handful of plausible surfaces
    # instead of dynamically compiling it ~55k times per question.
    question_folded = " ".join(str(ctx.question).casefold().split())
    matches_by_surface: dict[str, list[tuple[str, str, str]]] = {}
    for paper_id in ids:
        paper = by_id(paper_id)
        title = str(getattr(paper, "title", "") or "").strip()
        if not title:
            continue
        surfaces = [title]
        if ":" in title:
            surfaces.insert(0, title.split(":", 1)[0].strip())
        for surface in dict.fromkeys(surfaces):
            normalized = " ".join(surface.casefold().split())
            alphanumeric = re.sub(r"[^a-z0-9]", "", normalized)
            if (
                len(alphanumeric) < 4
                or normalized in _GENERIC_SOURCE_OWNER_SURFACES
                or normalized not in question_folded
                or not _question_marks_source_owner(ctx.question, surface)
            ):
                continue
            matches_by_surface.setdefault(squash_title(surface), []).append(
                (paper_id, surface, title)
            )

    # A one-source question naming two distinct title owners is ambiguous.  Do
    # not let pool order decide which one receives the only paper slot.
    matched_papers = {
        paper_id
        for matches in matches_by_surface.values()
        for paper_id, _surface, _title in matches
    }
    if len(matched_papers) != 1:
        return []

    target = targets[0]
    candidates: list[Candidate] = []
    seen_papers: set[str] = set()
    for matches in matches_by_surface.values():
        # The cue identifies a source only when its surface maps to one paper.
        unique_papers = {paper_id for paper_id, _surface, _title in matches}
        if len(unique_papers) != 1:
            continue
        paper_id, surface, title = matches[0]
        if paper_id in seen_papers:
            continue
        seen_papers.add(paper_id)
        candidates.append(Candidate(
            paper_id=paper_id,
            provenance=["name"],
            support=[{
                "source": "title_surface",
                "alias": surface,
                "paper_id": paper_id,
                "text": title,
                "page": None,
                "group_key": target.key,
                "role": "target",
                "title_match_kind": "exact",
                "title_match_count": 1,
                "title_surface_match_count": 1,
                "question_source_owner": True,
            }],
            route_signals=[RouteSignal(
                route="name", rank=0, score=None,
                group_key=target.key, role="target",
            )],
        ))
    return candidates


_GENERIC_TITLE_SUFFIXES = frozenset({
    "approach", "benchmark", "framework", "method", "model", "paper",
    "study", "system", "taxonomy", "work",
})


def _title_surface_variants(method: str) -> list[str]:
    """Conservative title aliases for planner-added generic head nouns.

    Planners commonly emit ``NAME benchmark`` or ``NAME taxonomy`` although
    the actual title begins ``NAME: ...``.  Strip exactly one generic trailing
    noun for title lookup only; body-alias resolution keeps the original text.
    """
    raw = str(method or "").strip()
    parts = raw.split()
    variants = [raw] if raw else []
    if (
        len(parts) > 1
        and parts[-1].strip(".,:;()[]{}").casefold()
        in _GENERIC_TITLE_SUFFIXES
    ):
        shorter = " ".join(parts[:-1]).strip()
        if len(re.sub(r"[^a-z0-9]", "", shorter.casefold())) >= 3:
            variants.append(shorter)
    return variants


def _title_surface_matches(
    method: str, pool: object | None
) -> list[_TitleSurfaceMatch]:
    """Return pool papers whose title contains ``method`` as a whole surface.

    Exact/leading title matches sort before interior matches, then paper id makes
    collisions deterministic.  Requiring at least three alphanumeric characters
    prevents planner fragments such as ``A`` from scanning into junk matches.
    The pool contains ~27k papers, so this bounded per-question scan is cheap and
    avoids changing or rebuilding the persisted indexes.
    """
    if pool is None:
        return []
    surface = str(method or "").strip().casefold()
    if len(re.sub(r"[^a-z0-9]", "", surface)) < 3:
        return []
    ids = getattr(pool, "ids", None)
    by_id = getattr(pool, "by_id", None)
    if not isinstance(ids, list) or not callable(by_id):
        return []

    # The metadata snapshot contains a small number of OCR-spaced titles such
    # as ``500x C ompressor: ...``.  Whole-token containment cannot recognize
    # ``500xCompressor`` in that title even though the pre-colon segment is an
    # exact paper-name identity.  Use the same whitespace/dash-insensitive
    # normalization as ``ExactAcronymIndex`` for exact full-title/pre-colon
    # identity, while retaining the stricter whole-surface containment as the
    # lower-priority fallback for names embedded elsewhere in a title.
    squashed_surface = squash_title(method)
    matches: list[tuple[int, str, str]] = []
    for paper_id in ids:
        paper = by_id(paper_id)
        raw_title = str(getattr(paper, "title", "") or "").strip()
        title = raw_title.casefold()
        pre_colon = raw_title.split(":", 1)[0].strip()
        exact_identity = bool(squashed_surface) and squashed_surface in {
            squash_title(raw_title),
            squash_title(pre_colon),
        }
        contains_surface = bool(title) and _contains_surface(title, surface)
        if not exact_identity and not contains_surface:
            continue
        if exact_identity or title == surface:
            priority = 0
            kind = "exact"
        elif title.startswith(f"{surface}:") or title.startswith(f"{surface} "):
            priority = 1
            kind = "leading"
        else:
            priority = 2
            kind = "interior"
        matches.append((priority, paper_id, kind))
    matches.sort()
    exact_match_count = sum(kind == "exact" for _, _, kind in matches)
    surface_match_count = len(matches)
    return [
        _TitleSurfaceMatch(
            paper_id=paper_id,
            surface=str(method or "").strip(),
            kind=kind,
            exact_match_count=exact_match_count if kind == "exact" else 0,
            surface_match_count=surface_match_count,
        )
        for _, paper_id, kind in matches
    ]


def _best_alias_record(records: list) -> object | None:
    """The alias record with the strongest evidence sentence (non-empty
    `support_text`); `records_for` already sorts best-first, so take the first
    usable one. None if none carry a sentence."""
    for r in records or []:
        if isinstance(getattr(r, "support_text", ""), str) and r.support_text.strip():
            return r
    return None


def _name_support(method: str, pid: str, ctx: RetrievalContext) -> tuple[str, int | None]:
    """Best `(text, page)` evidence for a name-resolved (`method` -> `pid`) link:
      1. PREFER the alias record's OWN coinage/definition sentence (`support_text`
         + `page`) -- it literally shows the paper coining/defining the alias.
      2. else a BM25 passage for that paper matching the alias (chunk `.text`/`.page`).
      3. else the pool paper's title+abstract.
    Returns `("", None)` when nothing is available (documented: such a candidate
    cannot ground, so the precision gate will drop it)."""
    records_for = getattr(ctx.aliases, "records_for", None)
    if callable(records_for):
        best = _best_alias_record(records_for(method, pid))
        if best is not None:
            return best.support_text, best.page

    if ctx.passages is not None:
        for chunk_id, _score in ctx.passages.search(method, k=_NAME_SUPPORT_SEARCH_K):
            rec = ctx.passages.get(chunk_id)
            if rec is not None and rec.paper_id == pid and str(getattr(rec, "text", "") or "").strip():
                return rec.text, rec.page

    if ctx.pool is not None:
        by_id = getattr(ctx.pool, "by_id", None)
        paper = by_id(pid) if callable(by_id) else None
        if paper is not None:
            text = f"{getattr(paper, 'title', '') or ''} {getattr(paper, 'abstract', '') or ''}".strip()
            if text:
                return text, None

    return "", None


# --------------------------------------------------------------------------- #
# Strategy: "baseline" -- follow a named paper's comparison edges (its baselines,
# and the papers that use it AS a baseline) via the relations index.
# --------------------------------------------------------------------------- #
@register_strategy("baseline")
class BaselineStrategy:
    """Resolve each named method to its primary paper (top-1 alias anchor), then
    emit that anchor's comparison neighbours from the relations index:
      * `baselines_of(anchor)`         -- papers the anchor compares AGAINST;
      * `compared_against_by(anchor)`  -- papers that use the anchor as a baseline.
    Each Candidate carries a RESOLVED-edge `relation_support` (shape accepted by
    the cascade's attestation gate). No relations index -> []."""

    name = "baseline"

    def search(self, plan: Plan, ctx: RetrievalContext) -> list[Candidate]:
        if ctx.relations is None or ctx.aliases is None:
            return []
        out: list[Candidate] = []
        for method in plan.named_methods:
            if not (isinstance(method, str) and method.strip()):
                continue
            group_key, role = _target_group(plan, method)
            anchors = ctx.aliases.resolve(method)
            if not anchors:
                continue
            anchor = anchors[0]  # top-1 alias hit is the anchor paper

            local_rank = 0
            for target in ctx.relations.baselines_of(anchor):
                out.append(Candidate(
                    paper_id=target,
                    provenance=["baseline"],
                    relation_support={
                        "relation_type": "compares_against",
                        "source_paper_id": anchor,
                        "target_paper_id": target,
                        "direction": "forward",
                        "anchor_method": method,
                    },
                    route_signals=[RouteSignal(
                        route="baseline", rank=local_rank, score=None,
                        group_key=group_key, role=role,
                    )],
                ))
                local_rank += 1

            for seed in ctx.relations.compared_against_by(anchor):
                out.append(Candidate(
                    paper_id=seed,
                    provenance=["baseline"],
                    relation_support={
                        "relation_type": "compares_against",
                        "source_paper_id": seed,
                        "target_paper_id": anchor,
                        "direction": "reverse",
                        "anchor_method": method,
                    },
                    route_signals=[RouteSignal(
                        route="baseline", rank=local_rank, score=None,
                        group_key=group_key, role=role,
                    )],
                ))
                local_rank += 1
        return out


# --------------------------------------------------------------------------- #
# Strategy: "citation" -- find papers whose text mentions a cited/baseline anchor.
# --------------------------------------------------------------------------- #
@register_strategy("citation")
class CitationStrategy:
    """Retrieve papers that mention a structured evidence anchor.

    This route is intentionally inert for legacy plans (which have no typed
    targets), so registering it cannot change existing production ordering.
    When enabled, it searches passages for each evidence anchor, excludes the
    anchor paper itself when aliases can resolve it, and preserves group-local
    ranks so downstream selection knows which anchor produced each candidate.
    """

    name = "citation"

    def search(self, plan: Plan, ctx: RetrievalContext) -> list[Candidate]:
        if ctx.passages is None:
            return []
        anchors = [
            target for target in getattr(plan, "targets", ())
            if getattr(target, "role", None) == "evidence_anchor"
        ]
        if not anchors:
            return []

        out: list[Candidate] = []
        for anchor in anchors:
            anchor_ids = _resolve_anchor_ids(anchor.text, ctx)
            hits = ctx.passages.search(anchor.text, k=_CITATION_SEARCH_K)
            by_paper: dict[str, list[dict]] = {}
            order: list[str] = []
            for chunk_id, score in hits:
                rec = ctx.passages.get(chunk_id)
                if rec is None or rec.paper_id in anchor_ids:
                    continue
                bucket = by_paper.get(rec.paper_id)
                if bucket is None:
                    bucket = []
                    by_paper[rec.paper_id] = bucket
                    order.append(rec.paper_id)
                if len(bucket) >= _CITATION_MAX_CHUNKS_PER_PAPER:
                    continue
                bucket.append({
                    "source": "citation_passage",
                    "chunk_id": rec.chunk_id,
                    "page": rec.page,
                    "text": rec.text,
                    "score": float(score),
                    "anchor": anchor.text,
                })

            for local_rank, paper_id in enumerate(order):
                support = by_paper[paper_id]
                out.append(Candidate(
                    paper_id=paper_id,
                    provenance=["citation"],
                    support=support,
                    route_signals=[RouteSignal(
                        route="citation",
                        rank=local_rank,
                        score=_best_support_score(support),
                        group_key=anchor.key,
                        role="evidence_anchor",
                    )],
                ))
        return out


def _resolve_anchor_ids(text: str, ctx: RetrievalContext) -> set[str]:
    if ctx.aliases is None:
        return set()
    resolve = getattr(ctx.aliases, "resolve", None)
    if not callable(resolve):
        return set()
    try:
        # Alias resolution can contain generic collisions or later papers that
        # reuse the name. Only the highest-ranked origin is safe to exclude.
        return set(resolve(text)[:1])
    except Exception:  # noqa: BLE001 -- unresolved anchor still permits search
        return set()


# --------------------------------------------------------------------------- #
# Strategy: "object" -- BM25 over table/figure captions (answer-bearing papers).
# --------------------------------------------------------------------------- #
@register_strategy("object")
class ObjectStrategy:
    """Retrieve papers through their table/figure captions.

    This route addresses a different question from ``name`` and ``property``:
    the correct paper may merely *report the requested values* in a table or
    figure rather than originate the named method.  The global query preserves
    broad object recall; structured named targets additionally receive
    group-local ``target_object`` searches so traces can distinguish which
    requested row/method surfaced a paper.

    Every support item carries the scorer-canonical visible ID and page already
    stored by ``ObjectsIndex``.  That makes a retrieved object directly useful
    to later evidence localization and makes the route auditable without opening
    the PDF.  No objects index -> ``[]``.
    """

    name = "object"

    def search(self, plan: Plan, ctx: RetrievalContext) -> list[Candidate]:
        if ctx.objects is None:
            return []

        query = _property_query(plan, ctx)
        source_type = _required_object_type(plan)
        out = _object_candidates(query, ctx, source_type=source_type)

        named_targets = _named_target_groups(plan)
        for target in named_targets:
            excluded_surfaces = [
                surface
                for other in named_targets
                if other.key.casefold() != target.key.casefold()
                for surface in (other.key, other.text)
            ]
            target_query = _target_property_query(
                target.text,
                query,
                excluded_targets=excluded_surfaces,
            )
            out.extend(_object_candidates(
                target_query,
                ctx,
                route="target_object",
                group_key=target.key,
                role=target.role,
                source_type=source_type,
            ))
        return out


def _required_object_type(plan: Plan) -> str | None:
    value = (plan.criterion or {}).get("required_source_type")
    return value if value in {"table", "figure"} else None


def _object_candidates(
    query: str,
    ctx: RetrievalContext,
    *,
    route: str = "object",
    group_key: str | None = None,
    role: str = "target",
    source_type: str | None = None,
) -> list[Candidate]:
    """Run one caption query and retain its rank local to ``group_key``."""
    if not query.strip():
        return []

    hits = ctx.objects.search(query, k=_OBJECT_SEARCH_K)
    by_paper: dict[str, list[dict]] = {}
    order: list[str] = []
    for object_key, score in hits:
        record = ctx.objects.get_by_key(object_key)
        if record is None or (
            source_type is not None and record.source_type != source_type
        ):
            continue
        bucket = by_paper.get(record.paper_id)
        if bucket is None:
            bucket = []
            by_paper[record.paper_id] = bucket
            order.append(record.paper_id)
        if len(bucket) >= _OBJECT_MAX_CAPTIONS_PER_PAPER:
            continue
        bucket.append({
            "source": "object_caption",
            "source_type": record.source_type,
            "object_key": record.object_key,
            "object_id": record.scorer_visible_id,
            "parser_visible_id": record.parser_visible_id,
            "page": record.page,
            "text": record.caption,
            "score": float(score),
        })

    return [
        Candidate(
            paper_id=paper_id,
            provenance=[route],
            support=by_paper[paper_id],
            route_signals=[RouteSignal(
                route=route,
                rank=local_rank,
                score=_best_support_score(by_paper[paper_id]),
                group_key=group_key,
                role=role,
            )],
        )
        for local_rank, paper_id in enumerate(order)
    ]


# --------------------------------------------------------------------------- #
# Strategy: "property" -- BM25 over the full-text passage KB ("papers that DO X").
# --------------------------------------------------------------------------- #
@register_strategy("property")
class PropertyStrategy:
    """BM25 the full-text passage KB for the criterion and explicit targets.

    The global criterion query preserves the proven property-first floor.  When
    the structured plan names concrete answer targets, one additional FREE BM25
    query is run per target (target text + global criterion).  Those hits carry a
    group-local ``target_property`` signal, which lets selection distinguish
    "this paper mentions ATT" from "this paper matches ATT *and* Tiny ImageNet,
    IPC=10".  Alias hits alone are intentionally not treated as that semantic
    corroboration.

    Hits are grouped by paper and carry their best passage chunks as support. No
    passages index -> []. Legacy plans without structured targets execute exactly
    the original single global query.
    """

    name = "property"

    def search(self, plan: Plan, ctx: RetrievalContext) -> list[Candidate]:
        if ctx.passages is None:
            return []
        query = _property_query(plan, ctx)
        out = _property_candidates(query, ctx)

        # Explicit named targets and sufficiently descriptive source targets
        # receive group queries. Generic planner slots such as "each proposed
        # method" remain inert and cannot manufacture arbitrary assignments.
        named_targets = _named_target_groups(plan)
        for target in named_targets:
            excluded_surfaces = [
                surface
                for other in named_targets
                if other.key.casefold() != target.key.casefold()
                for surface in (other.key, other.text)
            ]
            target_query = _target_property_query(
                target.text,
                query,
                excluded_targets=excluded_surfaces,
            )
            out.extend(_property_candidates(
                target_query,
                ctx,
                route="target_property",
                group_key=target.key,
                role=target.role,
            ))
        return out


def _property_candidates(
    query: str,
    ctx: RetrievalContext,
    *,
    route: str = "property",
    group_key: str | None = None,
    role: str = "target",
) -> list[Candidate]:
    """Run one property query and retain its rank local to ``group_key``."""
    if not query.strip():
        return []
    search_k = ctx.property_search_k
    if (
        isinstance(search_k, bool)
        or not isinstance(search_k, int)
        or search_k < 1
    ):
        search_k = DEFAULT_PROPERTY_SEARCH_K
    hits = ctx.passages.search(query, k=search_k)

    # Group by paper, preserving BM25 rank order (best-scoring paper first).
    by_paper: dict[str, list[dict]] = {}
    order: list[str] = []
    for chunk_id, score in hits:
        rec = ctx.passages.get(chunk_id)
        if rec is None:
            continue
        pid = rec.paper_id
        bucket = by_paper.get(pid)
        if bucket is None:
            bucket = []
            by_paper[pid] = bucket
            order.append(pid)
        if len(bucket) >= _PROPERTY_MAX_CHUNKS_PER_PAPER:
            continue
        bucket.append({
            "source": "passage",
            "route": route,
            "group_key": group_key,
            "role": role,
            "chunk_id": rec.chunk_id,
            "page": rec.page,
            "section_kind": rec.section_kind,
            "text": rec.text,
            "score": float(score),
        })

    return [
        Candidate(
            paper_id=pid,
            provenance=[route],
            support=by_paper[pid],
            route_signals=[RouteSignal(
                route=route,
                rank=local_rank,
                score=_best_support_score(by_paper[pid]),
                group_key=group_key,
                role=role,
            )],
        )
        for local_rank, pid in enumerate(order)
    ]


def _property_query(plan: Plan, ctx: RetrievalContext) -> str:
    """The description the planner distilled is the precise property statement;
    fall back to the raw question when it is missing."""
    desc = (plan.criterion or {}).get("description")
    if isinstance(desc, str) and desc.strip():
        return desc
    return ctx.question or ""


_TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_GENERIC_TARGET_TEXTS = frozenset({
    "paper", "first paper", "second paper", "method", "each method",
    "each proposed method", "objective", "result", "study",
})
_GENERIC_TARGET_TOKENS = frozenset({
    "a", "an", "the", "one", "first", "second", "other", "each",
    "paper", "papers", "proposed", "method", "methods", "result",
    "results", "study", "studies", "source", "target", "containing",
    "contains", "report", "reports", "reporting", "comparison",
})


def _named_target_groups(plan: Plan) -> list:
    """Concrete reporting-source targets worth an independent lexical query.

    An exact named-method mapping remains sufficient.  In addition, a detailed
    source description (for example "the paper using a recursive dueling-bandit
    equation") is itself a valuable retrieval query even when the question does
    not reveal a title or acronym.  Short generic slots stay inert so the
    planner cannot manufacture arbitrary paper assignments.
    """
    targets = [
        target for target in getattr(plan, "targets", ())
        if getattr(target, "role", None) == "target"
    ]
    if not targets:
        return []

    matched_keys: set[str] = set()
    out: list = []
    for method in getattr(plan, "named_methods", ()):
        if not (isinstance(method, str) and method.strip()):
            continue
        group_key, role = _target_group(plan, method)
        if role != "target":
            continue
        target = next(
            (
                item for item in targets
                if str(getattr(item, "key", "")).strip().casefold()
                == group_key.strip().casefold()
            ),
            None,
        )
        if target is None:
            continue
        normalized_key = group_key.strip().casefold()
        if normalized_key in matched_keys:
            continue
        matched_keys.add(normalized_key)
        out.append(target)

    for target in targets:
        normalized_key = str(getattr(target, "key", "")).strip().casefold()
        text = str(getattr(target, "text", "") or "").strip()
        normalized_text = " ".join(_TARGET_TOKEN_RE.findall(text.casefold()))
        if normalized_key in matched_keys:
            continue
        if normalized_text in _GENERIC_TARGET_TEXTS:
            continue
        tokens = _TARGET_TOKEN_RE.findall(text.casefold())
        informative_tokens = {
            token for token in tokens if token not in _GENERIC_TARGET_TOKENS
        }
        if len(tokens) < 4 or len(informative_tokens) < 2:
            continue
        matched_keys.add(normalized_key)
        out.append(target)
    return out


def _target_property_query(
    target_text: str,
    global_query: str,
    *,
    excluded_targets: list[str] | tuple[str, ...] = (),
) -> str:
    """Target identity plus global constraints, excluding competing targets.

    BM25 intentionally treats query terms as a set, so merely repeating the
    current target cannot focus a criterion that already names every requested
    method. Remove the *other* structured target surface forms first; each query
    then contains the current target plus the shared task constraints but not
    its competitors. Whole-surface boundaries prevent a short target from being
    deleted inside a longer token.
    """
    target_text = str(target_text or "").strip()
    focused_query = str(global_query or "").strip()
    if not target_text:
        return focused_query
    for surface in excluded_targets:
        surface = str(surface or "").strip()
        if not surface:
            continue
        focused_query = re.sub(
            rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])",
            " ",
            focused_query,
            flags=re.IGNORECASE,
        )
    focused_query = re.sub(r"\s+", " ", focused_query).strip()
    return f"{target_text}. {focused_query}".strip()


# --------------------------------------------------------------------------- #
# Strategy: "knn" -- DENSE semantic search over the pool's title+abstract
# embeddings (the recall lever the lexical-only new path lost).
# --------------------------------------------------------------------------- #
@register_strategy("knn")
class DenseKnnStrategy:
    """Semantic ("search by meaning") route: embed the QUERY and kNN the pool via
    the injected dense retriever (`ctx.dense`), REUSING the old path's cached
    title+abstract embeddings (`data/cache/pool_emb.npy`). The lexical-only new
    path (BM25 + name + relations) caps recall; for a property/criterion question
    the query is the semantic INTENT rather than a name, so dense recall surfaces
    papers the other routes miss.

    QUERY SOURCE: the planner's `criterion.description` (the distilled semantic
    intent) when present, else the raw `ctx.question` -- symmetric with
    `PropertyStrategy`. Emits ONE Candidate per retrieved paper carrying that
    paper's title+abstract as `support[].text` so it can attest/ground at the
    cascade gate; a text-LESS candidate silently dies there (the NameStrategy
    lesson). `ctx.dense is None` -> [] (clean degrade when dense is off/unavailable).

    Uses `ctx.dense.retrieve(query, k)` -> `list[(paper_id, score)]` (the same
    method + shape the OLD path's dense stack exposes; symmetric query encoding --
    no BGE query prefix, matching how the pool was embedded)."""

    name = "knn"

    def search(self, plan: Plan, ctx: RetrievalContext) -> list[Candidate]:
        if ctx.dense is None:
            return []
        query = _knn_query(plan, ctx)
        if not query.strip():
            return []
        hits = ctx.dense.retrieve(query, _KNN_K)
        out: list[Candidate] = []
        for pid, score in hits:
            out.append(Candidate(
                paper_id=pid,
                provenance=["knn"],
                support=[{
                    "source": "dense",
                    "paper_id": pid,
                    "text": _knn_text(pid, ctx),
                    "score": float(score),
                }],
            ))
        return out


def _knn_query(plan: Plan, ctx: RetrievalContext) -> str:
    """The semantic intent to embed: the criterion's distilled `description` (the
    precise property/criterion statement) if present, else the raw question."""
    desc = (plan.criterion or {}).get("description")
    if isinstance(desc, str) and desc.strip():
        return desc
    return ctx.question or ""


def _knn_text(pid: str, ctx: RetrievalContext) -> str:
    """The retrieved paper's title+abstract (from the pool) as the candidate's
    support text -- NON-EMPTY so the candidate can ground at the cascade's
    attestation / adversarial-validator gates. No pool / unknown paper -> ""
    (documented: such a candidate cannot ground, so the gate will drop it)."""
    if ctx.pool is not None:
        by_id = getattr(ctx.pool, "by_id", None)
        paper = by_id(pid) if callable(by_id) else None
        if paper is not None:
            text = f"{getattr(paper, 'title', '') or ''} {getattr(paper, 'abstract', '') or ''}".strip()
            if text:
                return text
    return ""


# --------------------------------------------------------------------------- #
# Router: run the plan's strategies and MERGE their output by paper_id.
# --------------------------------------------------------------------------- #
@dataclass
class _Merged:
    provenance: list[str] = field(default_factory=list)
    support: list[dict] = field(default_factory=list)
    relation_support: object | None = None
    # One RouteSignal per (route, semantic group). A global rank and a rank-1
    # result for the fourth named method are not interchangeable signals.
    route_signals: dict[tuple[str, str | None], "RouteSignal"] = field(
        default_factory=dict
    )


def _best_support_score(support: list[dict]) -> float | None:
    """The best (max) numeric `score` across a candidate's support items -- the
    property route's best passage BM25, or the knn cosine. None when no support
    item carries a score (name/baseline are rank-only routes)."""
    scores = [s["score"] for s in support
              if isinstance(s, dict) and isinstance(s.get("score"), (int, float))
              and not isinstance(s.get("score"), bool)]
    return max(scores) if scores else None


def _max_opt(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _merge_route_signal(slot: _Merged, signal: RouteSignal) -> None:
    key = (signal.route, signal.group_key)
    prev = slot.route_signals.get(key)
    if prev is None:
        slot.route_signals[key] = signal
        return
    slot.route_signals[key] = RouteSignal(
        route=signal.route,
        rank=min(prev.rank, signal.rank),
        score=_max_opt(prev.score, signal.score),
        group_key=signal.group_key,
        role=prev.role,
    )


def run_strategies(plan: Plan, ctx: RetrievalContext, *, question: str) -> list[Candidate]:
    """Run each REGISTERED strategy named in `plan.strategies` and merge their
    Candidates into one deduped list -- exactly the input to `PrecisionCascade.select`.

    MERGE-BY-paper_id: the same paper surfaced by several strategies becomes ONE
    Candidate -- provenances unioned (dedup, order-preserving), support lists
    concatenated, and the FIRST non-None `relation_support` kept. Output order is
    first-seen (deterministic: strategy order in the plan, then within-strategy
    order). An unregistered strategy is skipped with a WARNING; a strategy that
    raises is skipped with a WARNING -- the router never raises.

    FIX B: `plan.strategies` is DEDUPED order-preserving first, so a strategy named
    twice (e.g. `["name","name"]`) runs ONCE -- otherwise its support was merged in
    twice, inflating/duplicating a paper's support list."""
    ctx = replace(ctx, question=question)

    seen_names: set[str] = set()
    strategy_names = [
        n for n in plan.strategies if not (n in seen_names or seen_names.add(n))
    ]

    merged: dict[str, _Merged] = {}
    order: list[str] = []

    for name in strategy_names:
        strategy = RETRIEVAL_STRATEGIES.get(name)
        if strategy is None:
            _log.warning("run_strategies: skipping unregistered strategy %r", name)
            continue
        try:
            candidates = strategy.search(plan, ctx)
        except Exception as exc:  # noqa: BLE001 -- a broken strategy degrades to no-op
            _log.warning("run_strategies: strategy %r raised, skipping: %r", name, exc)
            continue

        # UNIQUE-PAPER rank per route: a route that lists the same paper twice
        # must not push the NEXT paper's rank up. `[p1, p1, p2]` -> p1 rank 0,
        # p2 rank 1 (not 2). Ranks are dense over distinct paper_ids in route
        # order, so RRF sees each paper's true within-route position.
        route_rank: dict[str, int] = {}
        for cand in candidates:
            pid = cand.paper_id
            if pid not in route_rank:
                route_rank[pid] = len(route_rank)
            rank = route_rank[pid]
            slot = merged.get(pid)
            if slot is None:
                slot = _Merged()
                merged[pid] = slot
                order.append(pid)
            for p in cand.provenance:
                if p not in slot.provenance:
                    slot.provenance.append(p)
            slot.support.extend(cand.support)
            if slot.relation_support is None and cand.relation_support is not None:
                slot.relation_support = cand.relation_support
            # Strategies that know their semantic sub-query (for example one
            # named method) supply group-local signals. Legacy/global routes get
            # the unique-paper rank computed above. Preserve every group: a flat
            # concatenation rank destroys the signal needed for target coverage.
            if cand.route_signals:
                for signal in cand.route_signals:
                    _merge_route_signal(slot, signal)
            else:
                _merge_route_signal(slot, RouteSignal(
                    route=name,
                    rank=rank,
                    score=_best_support_score(cand.support),
                ))

    # Descriptive reporting-source ownership is a post-retrieval annotation.
    # It is intentionally unable to add a paper: only IDs already in `merged`
    # are eligible, preserving both candidate cardinality and first-seen order.
    try:
        owner_candidates = _question_clause_owner_candidates(
            plan, ctx, set(merged)
        )
    except Exception as exc:  # noqa: BLE001 -- optional seam fails closed
        _log.warning(
            "run_strategies: descriptive source-owner scan raised, skipping: %r",
            exc,
        )
        owner_candidates = []
    for candidate in owner_candidates:
        slot = merged[candidate.paper_id]
        for provenance in candidate.provenance:
            if provenance not in slot.provenance:
                slot.provenance.append(provenance)
        slot.support.extend(candidate.support)
        for signal in candidate.route_signals:
            _merge_route_signal(slot, signal)

    # A named target may be self-defined only in its owner's abstract rather
    # than repeated literally in the title.  Recover that identity only for a
    # globally unique metadata definition with an existing top-three
    # target-property hit.  Like clause-owner, this annotation cannot expand or
    # reorder the candidate list by itself.
    try:
        definition_owner_candidates = _question_self_definition_owner_candidates(
            plan,
            ctx,
            {
                paper_id: tuple(slot.route_signals.values())
                for paper_id, slot in merged.items()
            },
        )
    except Exception as exc:  # noqa: BLE001 -- optional seam fails closed
        _log.warning(
            "run_strategies: named self-definition scan raised, skipping: %r",
            exc,
        )
        definition_owner_candidates = []
    for candidate in definition_owner_candidates:
        slot = merged[candidate.paper_id]
        for provenance in candidate.provenance:
            if provenance not in slot.provenance:
                slot.provenance.append(provenance)
        slot.support.extend(candidate.support)
        for signal in candidate.route_signals:
            _merge_route_signal(slot, signal)

    return [
        Candidate(
            paper_id=pid,
            provenance=merged[pid].provenance,
            support=merged[pid].support,
            relation_support=merged[pid].relation_support,
            route_signals=list(merged[pid].route_signals.values()),
        )
        for pid in order
    ]
