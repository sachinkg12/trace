"""`BM25Retriever`: a small, dependency-free BM25 index over the pool.

Hardens `analysis/bm25.py`'s `BM25Index` (independently verified in the
ceiling work: full 27,487-paper run completes in low single-digit seconds)
into the `Retriever` Protocol, built from a `PoolIndex`. The scoring math
(BM25 with K1=1.5, B=0.75, +1-smoothed idf) and inverted-index construction
are ported UNCHANGED from the verified prototype; this module adds the
`retrieve(query, k) -> list[tuple[str, float]]` Retriever signature, the
`register_retriever("bm25")` registration, and a deterministic tie-break
(`(-score, paper_id)`, mirroring `DenseRetriever`) so callers get a stable
ranking even when raw BM25 scores tie exactly.

We deliberately avoid numpy/sklearn for the index itself: a pure-Python
inverted index is fast enough at this corpus scale and keeps this module
identical in spirit to the analysis prototype it hardens.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from littraceqa.retrieval.interfaces import register_retriever
from littraceqa.retrieval.pool import PoolIndex

# Same short, hand-picked stopword list as analysis/common.py's tokenize --
# enough to keep BM25 from being swamped by function words, without pulling
# in an NLP dependency.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "with", "by", "at", "as", "that", "this", "these", "those",
    "which", "what", "who", "how", "many", "does", "do", "did", "it", "its",
    "be", "been", "being", "from", "into", "than", "then", "we", "our",
    "their", "paper", "papers", "using", "used", "use", "based", "via",
    "not", "no", "can", "will", "have", "has", "had", "such", "each",
    "between", "among", "about", "over", "under", "across", "vs",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokenization with stopword removal (ported from
    `analysis/common.py`, kept deliberately simple/deterministic)."""
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """Ported verbatim from `analysis/bm25.py` (verified prototype)."""

    K1 = 1.5
    B = 0.75

    def __init__(self, doc_ids: list[str], doc_tokens: list[list[str]]):
        assert len(doc_ids) == len(doc_tokens)
        self.doc_ids = doc_ids
        self.n_docs = len(doc_ids)
        self.doc_len = [len(toks) for toks in doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.n_docs)

        # postings: term -> list[(doc_idx, term_freq)]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, toks in enumerate(doc_tokens):
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, f in tf.items():
                self.postings[t].append((idx, f))

        self.idf: dict[str, float] = {}
        for t, plist in self.postings.items():
            df = len(plist)
            # BM25 idf with +1 smoothing to keep it non-negative
            self.idf[t] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score_all(self, query_tokens: list[str]) -> dict[int, float]:
        """Return {doc_idx: bm25_score} for every doc that shares >=1 query
        term (docs with zero overlap are simply absent, i.e. score 0)."""
        scores: dict[int, float] = defaultdict(float)
        seen_terms = set(query_tokens)
        for t in seen_terms:
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = self.idf[t]
            for doc_idx, tf in plist:
                dl = self.doc_len[doc_idx]
                denom = tf + self.K1 * (1 - self.B + self.B * dl / self.avgdl)
                scores[doc_idx] += idf * (tf * (self.K1 + 1)) / denom
        return scores


@register_retriever("bm25")
class BM25Retriever:
    """BM25 retriever built from a `PoolIndex`.

    Tokenizes each paper's `doc_text` (title + abstract) into an inverted
    index on construction, so `retrieve` calls are cheap.
    """

    def __init__(self, pool: PoolIndex):
        self.pool = pool
        self.ids = pool.ids
        doc_tokens = [tokenize(pool.doc_text(pid)) for pid in self.ids]
        self.index = BM25Index(self.ids, doc_tokens)

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Tokenize `query`, BM25-score every doc sharing >=1 term, return
        top-k sorted by score descending, tie-broken by paper_id ascending
        (mirrors `DenseRetriever`'s tie-break -- deterministic even when
        raw BM25 scores tie exactly, e.g. identical doc_text)."""
        query_tokens = tokenize(query)
        scores = self.index.score_all(query_tokens)
        ranked = sorted(
            scores.items(), key=lambda kv: (-kv[1], self.ids[kv[0]])
        )[:k]
        return [(self.ids[idx], score) for idx, score in ranked]
