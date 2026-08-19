"""Aliases index: acronym / short-name records RE-DERIVED from the parsed
artifact's raw PAGE TEXT (plus the pool's title/abstract), with a filter cascade
and a corpus-relative distinctiveness signal.

WHY re-derive here (downstream) instead of trusting the artifact's `acronyms`
field: the parser's acronym detector only captures a run of Title-case words
before `(ABBR)`, so a body-coined method whose expansion has a LOWERCASE leading
word ("simplified Consistency Models (sCM)") loses that word, and a method used
BARE ("1-step sCT", never written "Full Name (sCT)") is never captured at all.
Re-deriving from `pages` text lets us fix both without a 26k re-parse; the
artifact `acronyms` field is now OPTIONAL / DEPRECATED and is not consumed.

Sources per paper (each mined from raw text):
  * PARENTHETICAL definitions `<Expansion> (<ALIAS>)` -- source "body_definition"
    -- keeping LOWERCASE leading expansion words (the sCM fix).
  * COINAGE fallbacks ("referred to as sCT", "we call this X", "denoted as ECM")
    -- source "body_coinage" -- for aliases NOT written in `Name (ALIAS)` form.
  * BARE mixed-case candidates (a `sCT`-shaped token that recurs and co-occurs
    with method/title words) -- source "bare_mixedcase", low `confidence` -- the
    only way a bare-use coinage gets captured at all.
  * TITLE / ABSTRACT parenthetical acronyms mined from the pool metadata --
    source "title" / "abstract".

FILTER CASCADE (in order):
  1. STRUCTURAL INITIALS CHECK -- for records that carry an expansion, keep the
     pair only if the alias letters plausibly align with the expansion's word
     initials (`initials_match`). Coinage / bare records carry no expansion and
     are admitted on their pattern / recurrence evidence instead.
  2. CORPUS DOC FREQUENCY -- distinct papers using each normalized alias.
  3. ALIAS SCOPE -- CORPUS-RELATIVE: `generic_domain_term` when the alias's
     doc-frequency exceeds `max(3, 0.02 * n_papers)` (so LLM/NLP/FID that recur
     do not masquerade as coined terms on a small corpus, and the rule still
     scales to 26k); otherwise `paper_specific`. BOTH are kept -- generic terms
     still help retrieval -- but the scope lets `resolve` prefer the coining
     paper, then the lower-df (more distinctive) one.

`resolve(alias)` returns candidate paper_ids, paper_specific first, then lower
corpus doc-frequency, then paper_id for determinism.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

# Corpus-relative scope floor: an alias must exceed max(GENERIC_DF_FLOOR,
# GENERIC_DF_FRACTION * n_papers) distinct papers to count as a generic domain
# term. The floor keeps tiny corpora sane; the fraction scales to 26k.
GENERIC_DF_FLOOR = 3
GENERIC_DF_FRACTION = 0.02

# Retained for backward-compatible callers / tests that referenced it; the scope
# rule is now corpus-relative (see `_scope_for`) rather than a fixed cutoff.
PAPER_SPECIFIC_DF_MAX = GENERIC_DF_FLOOR

# Expansion stop-words that may be skipped (or matched by a lowercase alias
# letter) during the structural initials check.
_STOPWORDS = {
    "of", "for", "and", "the", "a", "an", "to", "in", "on", "or", "with",
    "per", "at", "by", "from", "via", "over",
}

# Method / paper cue words used to gate BARE mixed-case candidates: a bare token
# is only admitted if it co-occurs near one of these (or a title word), which
# keeps stray CamelCase noise out while catching real coined method names.
_METHOD_CUES = {
    "our", "ours", "we", "method", "model", "models", "training", "train",
    "trained", "distillation", "consistency", "propose", "proposed", "approach",
    "framework", "step", "sample", "samples", "objective", "architecture",
    "scheme", "variant", "baseline", "achieves", "outperforms",
}

# A bare token must recur at least this many times in a paper to be a candidate.
_BARE_MIN_COUNT = 3

# Confidence per source (kept on the record; resolve uses scope/df, not this).
_SOURCE_CONFIDENCE = {
    "body_definition": 1.0, "title": 1.0, "abstract": 1.0,
    "body_coinage": 0.7, "bare_mixedcase": 0.3,
}


def normalize_alias(alias: str) -> str:
    """Canonical alias key: alphanumerics only, upper-cased. `sCT` -> `SCT`,
    `AMoPO` -> `AMOPO`. Used for doc-frequency counting and `resolve` lookup."""
    return "".join(ch for ch in (alias or "") if ch.isalnum()).upper()


def _uppercase_count(s: str) -> int:
    return sum(1 for c in s if c.isupper())


def _is_alias_token(tok: str) -> bool:
    """A plausible acronym/short-name token: >=2 chars and >=1 uppercase letter
    (so `sCT`, `sCM`, `ECM`, `LCM` qualify; `as`, `this`, `ii` do not)."""
    return len(tok) >= 2 and _uppercase_count(tok) >= 1 and any(c.isalpha() for c in tok)


def _is_mixed_case(tok: str) -> bool:
    """A `sCT`-shaped token: >=2 chars with BOTH a lowercase and an uppercase
    letter (rules out all-caps `ECT`/`FID` and all-lower words)."""
    return (len(tok) >= 2 and any(c.islower() for c in tok)
            and any(c.isupper() for c in tok))


def _expansion_words(expansion: str) -> list[str]:
    """Split an expansion into comparison words, breaking hyphens too so
    `Fine-Tuning` contributes both `F` and `T`."""
    words: list[str] = []
    for raw in (expansion or "").replace("-", " ").split():
        w = "".join(ch for ch in raw if ch.isalpha())
        if w:
            words.append(w)
    return words


def initials_match(alias: str, expansion: str) -> bool:
    """True if `alias`'s letters align, in order, with the leading letters of
    `expansion`'s words.

    Case-consistent for mixed-case aliases: matching is done case-insensitively
    letter-by-letter, expansion stop-words (of/for/and/the/to/a/with/via/on/...)
    may be SKIPPED, and hyphenated expansion words are split so each part
    contributes an initial. A non-stop-word expansion word that fails to line up
    rejects the pair. So `sCM` <-> "simplified Consistency Models" and
    `sCT` <-> "simplified Consistency Training" pass, while
    `QA` <-> "demonstrating generalization" is rejected (Q vs d)."""
    alias_chars = [ch.lower() for ch in (alias or "") if ch.isalpha()]
    if not alias_chars:
        return False
    words = _expansion_words(expansion)
    if not words:
        return False
    inits = [(w[0].lower(), w.lower() in _STOPWORDS) for w in words]

    i = 0  # index into alias_chars
    for init, is_stop in inits:
        if i >= len(alias_chars):
            break  # alias fully consumed; trailing expansion words are fine
        if init == alias_chars[i]:
            i += 1
        elif is_stop:
            continue  # a stop-word the alias skips over
        else:
            return False  # a content word that doesn't line up -> reject
    return i == len(alias_chars)


# --------------------------------------------------------------------------- #
# Text miners (operate on raw page / title / abstract text).
# --------------------------------------------------------------------------- #
_PAREN_RE = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9'’\-]*)\s*\)")
# Sentence-ish boundary: trailing punctuation followed by space. Used to stop the
# backward expansion scan from crossing into a previous sentence.
_SENT_SPLIT_RE = re.compile(r"[.?!:;]\s+")


def _clean_word(w: str) -> str:
    """Strip surrounding punctuation but keep internal hyphens/apostrophes."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w)


def _best_preceding_expansion(alias: str, before: str) -> str | None:
    """Given the text `before` an `(ALIAS)`, return the LONGEST trailing run of
    words that `initials_match`es the alias -- INCLUDING leading lowercase words
    (`simplified`, `scaled`). Bounded to ~2*len(alias)+2 words and cut at the
    nearest preceding sentence boundary. Returns None if nothing aligns."""
    n_alpha = sum(1 for c in alias if c.isalpha())
    chunk = _SENT_SPLIT_RE.split(before)[-1]          # stay within one sentence
    words = [_clean_word(w) for w in chunk.split()]
    words = [w for w in words if w]
    window = 2 * n_alpha + 2
    words = words[-window:]
    # Longest valid suffix == earliest start that still aligns (keeps leading
    # lowercase words like "simplified").
    for start in range(len(words)):
        cand = " ".join(words[start:])
        if initials_match(alias, cand):
            return cand
    return None


def _parenthetical_records(text: str, page: int | None, source: str) -> list[dict]:
    """Mine `<Expansion> (<ALIAS>)` definitions, keeping leading lowercase
    expansion words."""
    out: list[dict] = []
    for m in _PAREN_RE.finditer(text):
        alias = m.group(1)
        if not _is_alias_token(alias) or _uppercase_count(alias) < 2:
            continue
        expansion = _best_preceding_expansion(alias, text[:m.start()])
        if not expansion or expansion.upper() == normalize_alias(alias):
            continue
        out.append({
            "alias": alias, "expansion": expansion, "page": page,
            "source": source, "support_text": f"{expansion} ({alias})",
            "confidence": _SOURCE_CONFIDENCE.get(source, 1.0),
        })
    return out


# Coinage cues for aliases NOT written in `Name (ALIAS)` form. Case-insensitive
# on the cue words; the captured token's own casing is validated afterwards.
_TOKEN = r"([A-Za-z][A-Za-z0-9\-]*)"
_COINAGE_RES = (
    # "... referred to as sCT", "we name such framework ... as TrigFlow",
    # "we call/term/dub/denote it/this/our <NAME> as X".
    re.compile(
        r"\b(?:we\s+)?(?:call|name|term|dub|denote|abbreviate|refer)\w*"
        r"\s+(?:to\s+)?(?:it|this|them|our|the|such)?\s*"
        r"(?:\w+\s+){0,4}?as\s+" + _TOKEN + r"\b", re.IGNORECASE),
    # Direct object, no "as": "we call this sCT", "we name it sCM".
    re.compile(
        r"\bwe\s+(?:call|name|term|dub|denote)\s+(?:it|this|them|our|the)?\s*"
        + _TOKEN + r"\b", re.IGNORECASE),
    # Past-participle: "denoted (as) ECM", "termed sCT", "abbreviated LCM".
    re.compile(
        r"\b(?:denoted|termed|dubbed|abbreviated|named|called|coined)"
        r"\s+(?:as\s+)?" + _TOKEN + r"\b", re.IGNORECASE),
)


def _coinage_records(text: str, page: int | None) -> list[dict]:
    """Mine body-coined aliases via coinage cue patterns (no expansion)."""
    out: list[dict] = []
    seen: set[str] = set()
    for rx in _COINAGE_RES:
        for m in rx.finditer(text):
            tok = m.group(1)
            if not _is_alias_token(tok):
                continue
            key = normalize_alias(tok)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "alias": tok, "expansion": "", "page": page,
                "source": "body_coinage", "support_text": m.group(0).strip(),
                "confidence": _SOURCE_CONFIDENCE["body_coinage"],
            })
    return out


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _bare_records(full_text: str, title: str) -> list[dict]:
    """Best-effort BARE mixed-case candidates: `sCT`-shaped tokens that recur
    (>= `_BARE_MIN_COUNT`) AND co-occur near a method/title cue word. Low
    confidence -- this is how a bare-use coinage (never in `Name (ALIAS)` form)
    gets captured at all."""
    counts = Counter(t for t in _WORD_RE.findall(full_text) if _is_mixed_case(t))
    title_words = {w.lower() for w in re.findall(r"[A-Za-z]+", title or "")
                   if len(w) > 3}
    cues = _METHOD_CUES | title_words
    lowered = full_text.lower()
    out: list[dict] = []
    for tok, c in counts.items():
        if c < _BARE_MIN_COUNT:
            continue
        if tok.lower() in title_words or _has_cue_context(full_text, lowered, tok, cues):
            out.append({
                "alias": tok, "expansion": "", "page": None,
                "source": "bare_mixedcase",
                "support_text": f"bare mixed-case token x{c}",
                "confidence": _SOURCE_CONFIDENCE["bare_mixedcase"],
            })
    return out


def _has_cue_context(text: str, lowered: str, tok: str, cues: set[str]) -> bool:
    pat = re.compile(r"\b" + re.escape(tok) + r"\b")
    for m in pat.finditer(text):
        window = lowered[max(0, m.start() - 50):m.end() + 50]
        for cue in cues:
            if re.search(r"\b" + re.escape(cue) + r"\b", window):
                return True
    return False


@dataclass
class AliasRecord:
    alias: str
    alias_normalized: str
    expansion: str
    paper_id: str
    page: int | None
    source: str  # body_definition | body_coinage | bare_mixedcase | title | abstract
    support_text: str
    alias_scope: str  # "paper_specific" | "generic_domain_term"
    corpus_doc_frequency: int
    confidence: float = 1.0


def _norm_text(text: str) -> str:
    """PDF text wraps definitions across lines; collapse whitespace first."""
    return re.sub(r"\s+", " ", text or "")


def _raw_records_for(artifact: dict, paper) -> list[dict]:
    """Collect candidate alias hits for one paper by RE-DERIVING from the
    artifact's raw page text (parenthetical + coinage + bare) and the pool's
    title/abstract, BEFORE the initials filter / distinctiveness pass. The
    artifact `acronyms` field is intentionally NOT consulted (deprecated)."""
    paper_id = artifact.get("paper_id", "")
    records: list[dict] = []
    page_texts: list[str] = []
    for p in artifact.get("pages") or []:
        page = p.get("page")
        text = _norm_text(p.get("text") or "")
        page_texts.append(text)
        records.extend(_parenthetical_records(text, page, "body_definition"))
        records.extend(_coinage_records(text, page))

    title = (getattr(paper, "title", "") or "") if paper is not None else ""
    abstract = (getattr(paper, "abstract", "") or "") if paper is not None else ""
    full_text = " ".join(page_texts + [_norm_text(title), _norm_text(abstract)])
    records.extend(_bare_records(full_text, title))

    if paper is not None:
        records.extend(_parenthetical_records(_norm_text(title), None, "title"))
        records.extend(_parenthetical_records(_norm_text(abstract), None, "abstract"))

    return [{**r, "paper_id": paper_id} for r in records]


def _scope_for(doc_frequency: int, n_papers: int) -> str:
    """Corpus-relative scope: generic once the alias exceeds
    `max(GENERIC_DF_FLOOR, GENERIC_DF_FRACTION * n_papers)` distinct papers."""
    threshold = max(GENERIC_DF_FLOOR, GENERIC_DF_FRACTION * n_papers)
    return "generic_domain_term" if doc_frequency > threshold else "paper_specific"


class AliasesIndex:
    """Filtered, scope-tagged alias records with `resolve` + `all_records`."""

    def __init__(self, records: list[AliasRecord]):
        self.records = list(records)
        self._by_norm: dict[str, list[AliasRecord]] = {}
        for r in self.records:
            self._by_norm.setdefault(r.alias_normalized, []).append(r)

    @classmethod
    def build(cls, artifacts, pool=None, **_ignored) -> "AliasesIndex":
        """Build from parsed `artifacts` (dicts, re-derived from their `pages`
        text) + an optional `pool` giving title/abstract text. `pool` may be a
        `PoolIndex` (`.by_id`) or a {paper_id: Paper} mapping; `None` skips
        title/abstract mining. (`**_ignored` swallows the retired
        `paper_specific_df_max` kwarg for backward compatibility.)"""
        artifacts = list(artifacts)
        n_papers = len(artifacts)

        def _paper_for(pid):
            if pool is None:
                return None
            if hasattr(pool, "by_id"):
                return pool.by_id(pid)
            return pool.get(pid)

        # 1. collect + structural initials filter (only where an expansion
        # exists; coinage/bare records are admitted on their own evidence).
        kept: list[dict] = []
        seen: set[tuple[str, str, str, str]] = set()  # (pid, norm, source, expansion)
        for art in artifacts:
            for hit in _raw_records_for(art, _paper_for(art.get("paper_id", ""))):
                alias = hit.get("alias", "")
                expansion = hit.get("expansion") or ""
                if expansion and not initials_match(alias, expansion):
                    continue
                norm = normalize_alias(alias)
                key = (hit["paper_id"], norm, hit["source"], expansion.lower())
                if key in seen:
                    continue
                seen.add(key)
                kept.append({
                    "alias": alias,
                    "alias_normalized": norm,
                    "expansion": expansion,
                    "paper_id": hit["paper_id"],
                    "page": hit.get("page"),
                    "source": hit.get("source", "body_definition"),
                    "support_text": hit.get("support_text", ""),
                    "confidence": hit.get("confidence", 1.0),
                })

        # 2. corpus doc frequency: DISTINCT papers per normalized alias.
        papers_per_alias: dict[str, set[str]] = {}
        for k in kept:
            papers_per_alias.setdefault(k["alias_normalized"], set()).add(k["paper_id"])
        doc_freq = {a: len(pids) for a, pids in papers_per_alias.items()}

        # 3. corpus-relative scope tag.
        records: list[AliasRecord] = []
        for k in kept:
            df = doc_freq[k["alias_normalized"]]
            records.append(AliasRecord(
                alias=k["alias"],
                alias_normalized=k["alias_normalized"],
                expansion=k["expansion"],
                paper_id=k["paper_id"],
                page=k["page"],
                source=k["source"],
                support_text=k["support_text"],
                alias_scope=_scope_for(df, n_papers),
                corpus_doc_frequency=df,
                confidence=k["confidence"],
            ))
        return cls(records)

    @classmethod
    def from_records(cls, records: list[dict]) -> "AliasesIndex":
        """Reconstruct from persisted `all_records()` dicts (the scope / doc-
        frequency were computed at build time and are stored on each record, so
        no artifact or pool re-read is needed to answer `resolve`)."""
        return cls([AliasRecord(**r) for r in records])

    def resolve(self, alias: str) -> list[str]:
        """Candidate paper_ids for `alias`: paper_specific first, then lower
        corpus doc-frequency (more distinctive), then paper_id for determinism.
        Distinct paper_ids, order-preserving."""
        norm = normalize_alias(alias)
        matches = self._by_norm.get(norm, [])
        ordered = sorted(
            matches,
            key=lambda r: (0 if r.alias_scope == "paper_specific" else 1,
                           r.corpus_doc_frequency, r.paper_id),
        )
        seen: set[str] = set()
        out: list[str] = []
        for r in ordered:
            if r.paper_id not in seen:
                seen.add(r.paper_id)
                out.append(r.paper_id)
        return out

    def records_for(self, alias: str, paper_id: str) -> list[AliasRecord]:
        """The `AliasRecord`s that link `alias` (normalized) to `paper_id`, best
        evidence first (higher `confidence`, then a longer `support_text`).

        This is the accessor the `name` retrieval strategy uses to recover the
        alias's OWN evidence sentence (`support_text` + `page`) -- the ideal
        support for a name-resolved candidate, since it literally shows the paper
        coining/defining the alias. Empty list if the pair is not in the index."""
        norm = normalize_alias(alias)
        matches = [r for r in self._by_norm.get(norm, []) if r.paper_id == paper_id]
        return sorted(
            matches,
            key=lambda r: (-(r.confidence or 0.0), -len(r.support_text or "")),
        )

    def all_records(self) -> list[dict]:
        return [asdict(r) for r in self.records]
