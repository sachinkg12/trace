"""Typed relations index: hidden-baseline extraction + open-set resolver (NIL).

WHY THIS EXISTS. The "comparison" questions name ONE paper (the SEED) but the
gold answer needs that paper's BASELINES -- the methods the seed compares itself
against. Those baseline names live only INSIDE the seed body (results tables,
"we compare against ...", related work) and are NEVER in the question. No amount
of question-side retrieval finds them; you have to read the seed and follow its
comparison edges. This index does exactly that: it mines baseline MENTIONS from a
seed's parsed artifact and links each to a pool `paper_id` OR records it as NIL.

Two-stage design (mirrors the aliases index's mine-then-resolve shape):

  1. MENTION EXTRACTION (`_extract_mentions`) from the parsed artifact:
     (a) TABLE captions (`source_type == "table"`) -- a "comparison with X, Y"
         caption names baselines directly, and we can attach the table object_id.
     (b) BODY sentences carrying a comparison CUE (`baseline`, `compare(d) with/
         to/against`, `we compare`, `versus|vs.`, `outperform*`, `prior work/
         methods`, `competing`, `state-of-the-art|SOTA`).
     (c) `related_work` SECTION text -- baselines are frequently only named here.
     From each source we pull METHOD-NAME-SHAPED tokens: >=2 uppercase letters so
     acronyms (`DEIM`, `DETR`), mixed-case (`FooNet`, `RetinaNet`) and hyphenated
     names (`RT-DETR`, `D-FINE`) are caught while ordinary Capitalized prose words
     (`Transformer`, `We`) are not. A small `_STOP_MENTIONS` list drops obvious
     datasets/metrics/venues (`COCO`, `FID`, `ICLR`, ...) that are not baselines.

  2. OPEN-SET RESOLUTION (`_MentionResolver`) -- each mention -> `paper_id` or
     NIL. Candidate generation reuses the existing indexes; acceptance is a
     PRECISION-ORDERED cascade, taking the FIRST generator that yields a
     CONFIDENT top-1:
       * exact/normalized/squash TITLE match (`ExactAcronymIndex.lookup_title`,
         incl. the pre-colon method segment of "Name: subtitle" titles);
       * ALIAS resolution (`AliasesIndex.resolve`) -- corpus-mined coined names,
         trusted only when UNAMBIGUOUS (<= `_AMBIG_MAX` candidates);
       * BM25 title fuzzy, but ONLY for a DISTINCTIVE mention (long, or bearing a
         digit/hyphen) AND ONLY if the mention is a whole TITLE WORD of the
         resolved paper -- a word-level (not substring) containment gate.
     We deliberately do NOT fuzzy-link a short bare acronym (`CNN`, `DETR`,
     `DINO`, `FPS`, `ECT`): it matches SOME coincidental title word almost
     everywhere, so snapping it to a paper is a force-match. Such baselines are
     left NIL -- the correct open-set outcome, since most cited methods are
     OUTSIDE the 27k pool. A resolution back to the SEED ITSELF (the paper's own
     coined name) is also rejected to NIL: a paper is never its own baseline.
     We NEVER force-match and NEVER invent an id -- a NIL mention is RECORDED,
     not dropped, so downstream can still surface "the seed compares against an
     un-poolable method" and never silently snaps a baseline to a wrong paper.

RELATION RECORDS are typed `compares_against` edges carrying their support
(page, source_type, optional table object_id, evidence text), a confidence, and
a `resolution_status` of `"resolved"` / `"nil"`. `RelationsIndex` exposes the
forward edge `baselines_of(seed) -> [target_id...]` (resolved only) and the
reverse edge `compared_against_by(target) -> [seed_id...]` ("who uses X as a
baseline"). Deterministic throughout; NEVER raises on a malformed artifact.

KNOWN LIMITATION (honest): baselines that appear ONLY as TABLE ROW LABELS (the
left-column method names of a results table, with no naming caption and no body
sentence) are NOT recovered here -- that needs Level-2 table-cell serialization
we do not have yet. Caption + body + related-work extraction covers the common
"comparison with X, Y" / "we compare against X" cases but will miss a bare
results table whose baselines are only in its rows.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from littraceqa.corpus.indexes.aliases import normalize_alias
from littraceqa.retrieval.bm25 import BM25Retriever
from littraceqa.retrieval.exact import ExactAcronymIndex
from littraceqa.retrieval.pool import PoolIndex


def _title_word_norms(title: str) -> set[str]:
    """Normalized (alphanumeric, upper) whole-word tokens of a title, e.g.
    "D-FINE: Redefine Regression" -> {"DFINE", "REDEFINE", "REGRESSION"}. Used as
    a high-precision containment gate: a fuzzy hit is only accepted if the mention
    IS one of these words (not a mere substring)."""
    return {n for n in (normalize_alias(w) for w in (title or "").split()) if n}

RELATION_TYPE = "compares_against"

# A method-name-shaped token: a letter-led run, optionally hyphen-joined
# (`RT-DETR`, `D-FINE`). Kept only if `_is_method_shaped` (>=2 uppercase).
_MENTION_TOK_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")

# Comparison CUE words that flag a BODY sentence as naming baselines.
_CUE_RE = re.compile(
    r"\b(?:baselines?"
    r"|compared?\s+(?:with|to|against)"
    r"|we\s+compare"
    r"|versus|vs\.?"
    r"|outperform\w*"
    r"|prior\s+(?:work|methods?)"
    r"|competing"
    r"|state[-\s]of[-\s]the[-\s]art|sota)\b",
    re.IGNORECASE,
)

# Sentence-ish split on terminal punctuation + space (PDF text already collapsed).
_SENT_SPLIT_RE = re.compile(r"(?<=[.?!;:])\s+")
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"(\d+)")

# Method-shaped tokens that are NOT baselines: datasets, metrics, venues, common
# ALLCAPS caption/section words, and generic terms. Normalized (alphanumeric,
# upper) for matching. NOTE: we deliberately do NOT stoplist ambiguous model
# FAMILIES (YOLO, DINO, ResNet, ...): those ARE baselines, so we keep them as
# recorded mentions and let the resolver return NIL (they are rarely poolable).
_STOP_MENTIONS = {
    normalize_alias(w) for w in (
        # datasets / benchmarks
        "COCO", "ImageNet", "CIFAR", "CIFAR-10", "MNIST", "LVIS", "VOC",
        "ADE20K", "GLUE", "SQuAD", "MMLU", "LAION", "WikiText",
        # metrics
        "FID", "IS", "AP", "mAP", "AUC", "BLEU", "ROUGE", "PSNR", "SSIM",
        "NFE", "NFES", "KID", "CLIP", "APval",
        # hardware / generic terms
        "GPU", "GPUs", "CPU", "TPU", "FLOPs", "RAM", "API", "NVIDIA",
        # fields / venues
        "NLP", "CV", "ML", "AI", "LLM", "LLMs", "VLM", "VLMs",
        "ICLR", "NeurIPS", "CVPR", "ICML", "ECCV", "ICCV", "ACL", "EMNLP",
        "AAAI", "arXiv",
        # common ALLCAPS caption / section / prose words that survive the
        # >=2-uppercase shape test (esp. from all-caps table captions).
        "COMPARISON", "COMPARISONS", "WITH", "AND", "THE", "FOR", "VS",
        "VERSUS", "RESULTS", "RESULT", "ABLATION", "ABLATIONS", "METHOD",
        "METHODS", "BASELINE", "BASELINES", "MODEL", "MODELS", "DATASET",
        "DATASETS", "TABLE", "FIGURE", "PERFORMANCE", "ACCURACY", "SPEED",
        "LATENCY", "OURS", "ALL", "SOTA", "STOA", "OUR", "PROPOSED",
    )
}

# Minimum normalized (alphanumeric) length for a mention: 1-2 char tokens
# ("XL", "RT", "DE", "GT", "CM") are too generic/ambiguous to be a baseline name.
_MIN_MENTION_LEN = 3

# Ambiguity ceiling for alias candidate lists: more than this many candidate
# papers is not a "clear top-1", so we do NOT trust it (prefer NIL).
_AMBIG_MAX = 3

# The BM25 fuzzy path is only allowed for a DISTINCTIVE mention -- one whose
# normalized form is long, OR carries a digit or hyphen (`RT-DETR`, `YOLOv10`,
# `500xCompressor`). A short bare acronym (`CNN`, `DETR`, `DINO`, `FPS`, `ECT`)
# matches SOME coincidental title word almost everywhere, so fuzzy-linking it is
# a force-match: those are left NIL (correct open-set behavior -- most such
# baselines are simply outside the pool).
_FUZZY_MIN_DISTINCTIVE_LEN = 5

# Support-source priority when the SAME mention is found in several places
# (table caption is the strongest, carrying an object_id; body is weakest).
_SOURCE_PRIORITY = {"table": 0, "related_work": 1, "body": 2}

# Confidences per accepting generator (documentary; resolve uses cascade order).
_CONF_TITLE = 0.95
_CONF_ALIAS = 0.85
_CONF_BM25 = 0.6
_BM25_K = 5


def _norm_text(text: str) -> str:
    return _WS_RE.sub(" ", text or "")


def _is_distinctive(mention: str) -> bool:
    """A mention distinctive enough to risk the fuzzy path: normalized length
    >= `_FUZZY_MIN_DISTINCTIVE_LEN`, OR it carries a digit or hyphen (`RT-DETR`,
    `YOLOv10`). Short bare acronyms (`CNN`, `DETR`) are NOT distinctive."""
    return (len(normalize_alias(mention)) >= _FUZZY_MIN_DISTINCTIVE_LEN
            or any(c.isdigit() for c in mention) or "-" in mention)


# Lowercase adjectival hyphen-suffixes that turn a method name into a phrase
# ("DETR-based", "DEIM-trained", "CNN-style"): stripped so the token collapses to
# the bare method name (which is then judged on its own merits, not the phrase).
_ADJ_SUFFIXES = {
    "based", "trained", "style", "like", "wise", "aware", "free", "guided",
    "driven", "level", "oriented", "specific", "agnostic", "conditioned",
}


def _strip_adjectival_suffix(tok: str) -> str:
    """Drop trailing `-<lowercase adjectival word>` segments: `DETR-based`
    -> `DETR`, `CNN-style` -> `CNN`. Size/variant tags (`ResNet-50`, `D-FINE-L`)
    are untouched (not lowercase words)."""
    parts = tok.split("-")
    while len(parts) > 1 and parts[-1].islower() and parts[-1] in _ADJ_SUFFIXES:
        parts.pop()
    return "-".join(parts)


def _is_method_shaped(tok: str) -> bool:
    """A method-name candidate: >=2 uppercase letters AND >=`_MIN_MENTION_LEN`
    alphanumeric chars (so `RT-DETR`, `DEIM`, `FooNet` qualify; `Transformer`,
    `We`, `and`, and 2-char `XL`/`RT` do not)."""
    return (sum(1 for c in tok if c.isupper()) >= 2
            and len(normalize_alias(tok)) >= _MIN_MENTION_LEN)


def _method_tokens(text: str) -> list[str]:
    """Ordered, de-duplicated method-name-shaped tokens in `text` (stop-mentions
    like COCO/FID/COMPARISON removed). Order is first-appearance for determinism."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _MENTION_TOK_RE.finditer(text):
        tok = _strip_adjectival_suffix(m.group(0))
        norm = normalize_alias(tok)
        if not _is_method_shaped(tok) or norm in _STOP_MENTIONS or norm in seen:
            continue
        seen.add(norm)
        out.append(tok)
    return out


@dataclass(frozen=True)
class Support:
    """Where a mention was found: page, which source, optional table object_id,
    and the caption/sentence text it was pulled from."""
    page: int | None
    source_type: str  # "table" | "body" | "related_work"
    text: str
    object_id: str | None = None


@dataclass(frozen=True)
class RelationRecord:
    relation_id: str
    source_paper_id: str
    target_paper_id: str | None  # None == NIL (mention outside the pool)
    mention_text: str
    relation_type: str
    support: Support
    confidence: float
    resolution_status: str  # "resolved" | "nil"


# --------------------------------------------------------------------------- #
# Mention extraction from a parsed seed artifact.
# --------------------------------------------------------------------------- #
@dataclass
class _RawMention:
    text: str
    page: int | None
    source_type: str
    object_id: str | None
    support_text: str


def _table_object_id(paper_id: str, scorer_visible_id: str) -> str:
    m = _NUM_RE.search(scorer_visible_id or "")
    return f"{paper_id}:table:{m.group(1) if m else ''}"


def _extract_mentions(artifact: dict) -> list[_RawMention]:
    """Collect candidate baseline mentions from an artifact's table captions,
    cue-bearing body sentences, and related_work-section pages. Never raises."""
    paper_id = artifact.get("paper_id", "") or ""
    out: list[_RawMention] = []

    # (a) TABLE captions -- names in a "comparison with X, Y" caption.
    for cap in artifact.get("captions") or []:
        if cap.get("source_type") != "table":
            continue
        caption = cap.get("caption", "") or ""
        object_id = _table_object_id(paper_id, cap.get("scorer_visible_id", ""))
        for tok in _method_tokens(caption):
            out.append(_RawMention(tok, cap.get("page"), "table", object_id, caption))

    # Pages that carry a related_work section heading -> mine their text too.
    rw_pages = {s.get("page") for s in (artifact.get("sections") or [])
                if s.get("kind") == "related_work"}

    # (b)/(c) body cue sentences + related_work-page sentences.
    for page in artifact.get("pages") or []:
        page_no = page.get("page")
        text = _norm_text(page.get("text") or "")
        if not text:
            continue
        for sent in _SENT_SPLIT_RE.split(text):
            if page_no in rw_pages:
                source_type = "related_work"
            elif _CUE_RE.search(sent):
                source_type = "body"
            else:
                continue
            for tok in _method_tokens(sent):
                out.append(_RawMention(tok, page_no, source_type, None, sent.strip()))

    return out


def _pick_best_support(mentions: list[_RawMention]) -> _RawMention:
    """Among several supports for the SAME mention, prefer table > related_work
    > body, then the earliest page (table captions carry the object_id)."""
    return min(
        mentions,
        key=lambda m: (_SOURCE_PRIORITY.get(m.source_type, 9),
                       m.page if m.page is not None else 10**9),
    )


# --------------------------------------------------------------------------- #
# Open-set resolution: mention -> paper_id or NIL.
# --------------------------------------------------------------------------- #
class _MentionResolver:
    """Precision-ordered cascade resolving a mention to a confident pool paper_id
    or None (NIL). Candidate generators are reused from the existing indexes."""

    def __init__(self, aliases_index, pool: PoolIndex, *, use_fuzzy: bool = True):
        self._aliases = aliases_index
        self._pool = pool
        self._exact = ExactAcronymIndex(pool)
        self._bm25 = BM25Retriever(pool) if use_fuzzy else None

    def resolve(self, mention: str) -> tuple[str | None, float]:
        """Return (paper_id, confidence) or (None, 0.0) for NIL. Precision-first:
        take the first generator that yields a CONFIDENT top-1, else NIL."""
        mention = (mention or "").strip()
        if not mention:
            return None, 0.0

        # 1. exact / normalized / squash title (incl. pre-colon method segment)
        #    -- highest precision.
        hits = self._exact.lookup_title(mention)
        if hits:
            return hits[0], _CONF_TITLE

        # 2. corpus-mined ALIAS resolution -- trusted only if UNAMBIGUOUS. The
        #    aliases index orders paper_specific/distinctive first, so a clear
        #    top-1 (few candidates) is a confident coined-name link.
        alias_hits = [pid for pid in self._aliases.resolve(mention)
                      if self._pool.by_id(pid) is not None]
        if alias_hits and len(alias_hits) <= _AMBIG_MAX:
            return alias_hits[0], _CONF_ALIAS

        # 3. BM25 fuzzy, accepted ONLY for a DISTINCTIVE mention AND ONLY if the
        #    mention is a whole title WORD of the resolved paper (word-level,
        #    normalized) -- not a mere substring. This catches a distinctive
        #    single-word method named mid/post-title while rejecting both
        #    incidental substring hits ("cnn" inside "msf-CNN") and generic
        #    bare-acronym snaps ("DETR"/"DINO"/"FPS" onto a coincidental title
        #    word). A short bare acronym is deliberately left NIL.
        if self._bm25 is not None and _is_distinctive(mention):
            mnorm = normalize_alias(mention)
            try:
                fuzzy = self._bm25.retrieve(mention, _BM25_K)
            except Exception:  # noqa: BLE001 -- degrade to NIL, never crash
                fuzzy = []
            for pid, _score in fuzzy:
                paper = self._pool.by_id(pid)
                if paper is not None and mnorm in _title_word_norms(paper.title):
                    return pid, _CONF_BM25

        # Nothing confident -> NIL (mention is outside the pool).
        return None, 0.0


# --------------------------------------------------------------------------- #
# RelationsIndex.
# --------------------------------------------------------------------------- #
def _as_pool(pool) -> PoolIndex:
    """Accept a `PoolIndex` or a {paper_id: Paper} mapping."""
    if pool is None:
        raise ValueError("RelationsIndex requires a pool (PoolIndex or mapping)")
    if hasattr(pool, "by_id") and hasattr(pool, "ids"):
        return pool
    return PoolIndex(list(pool.values()))


class RelationsIndex:
    """Typed `compares_against` edges from seed papers to their baselines,
    resolved to pool paper_ids or NIL, with forward/reverse lookups."""

    def __init__(self, records: list[RelationRecord]):
        self.records = list(records)
        self._forward: dict[str, list[str]] = {}
        self._reverse: dict[str, list[str]] = {}
        for r in self.records:
            if r.resolution_status == "resolved" and r.target_paper_id is not None:
                self._forward.setdefault(r.source_paper_id, []).append(r.target_paper_id)
                self._reverse.setdefault(r.target_paper_id, []).append(r.source_paper_id)

    @classmethod
    def build(cls, seed_artifacts, aliases_index, pool, *,
              use_fuzzy: bool = True) -> "RelationsIndex":
        """Extract baseline mentions from each seed artifact and resolve them to
        pool paper_ids (or NIL). `aliases_index` is an `AliasesIndex` (alias
        candidate-gen); `pool` is a `PoolIndex` or {paper_id: Paper} mapping.
        Deterministic; a malformed artifact degrades to no edges, never raises."""
        pool = _as_pool(pool)
        resolver = _MentionResolver(aliases_index, pool, use_fuzzy=use_fuzzy)

        records: list[RelationRecord] = []
        for art in seed_artifacts or []:
            try:
                raw = _extract_mentions(art)
            except Exception:  # noqa: BLE001 -- one bad artifact must not crash the build
                raw = []
            paper_id = (art.get("paper_id", "") if isinstance(art, dict) else "") or ""

            # Dedup mentions per seed by normalized surface form; keep best support.
            grouped: dict[str, list[_RawMention]] = {}
            for m in raw:
                grouped.setdefault(normalize_alias(m.text), []).append(m)

            for i, norm in enumerate(sorted(grouped)):  # sorted -> deterministic ids
                best = _pick_best_support(grouped[norm])
                target, conf = resolver.resolve(best.text)
                # A paper is never its OWN baseline: a mention that resolves back
                # to the seed (its own coined name/alias) is not a comparison
                # edge -> record it as NIL rather than a self-loop.
                if target == paper_id:
                    target, conf = None, 0.0
                status = "resolved" if target is not None else "nil"
                records.append(RelationRecord(
                    relation_id=f"{paper_id}:{RELATION_TYPE}:{i}",
                    source_paper_id=paper_id,
                    target_paper_id=target,
                    mention_text=best.text,
                    relation_type=RELATION_TYPE,
                    support=Support(page=best.page, source_type=best.source_type,
                                    text=best.support_text, object_id=best.object_id),
                    confidence=conf if target is not None else 0.0,
                    resolution_status=status,
                ))
        return cls(records)

    @classmethod
    def from_records(cls, records: list[dict]) -> "RelationsIndex":
        """Reconstruct from persisted `all_records()` dicts. Each record's
        `support` is a nested dict (`asdict` output) rebuilt into a `Support`;
        the forward/reverse edges are recomputed from the stored resolutions --
        no artifact re-read, no pool, no re-resolution."""
        out: list[RelationRecord] = []
        for r in records:
            r = dict(r)
            support = Support(**(r.pop("support") or {}))
            out.append(RelationRecord(support=support, **r))
        return cls(out)

    def baselines_of(self, paper_id: str) -> list[str]:
        """Resolved (non-NIL) baseline target paper_ids the seed compares against.
        Distinct, first-appearance order for determinism."""
        return _dedup(self._forward.get(paper_id, []))

    def compared_against_by(self, target_paper_id: str) -> list[str]:
        """Reverse edge: seed papers that use `target_paper_id` as a baseline
        ("who compares against X"). Distinct, first-appearance order."""
        return _dedup(self._reverse.get(target_paper_id, []))

    def all_records(self) -> list[dict]:
        return [asdict(r) for r in self.records]


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
