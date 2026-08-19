"""Composition root: the SOLE place concrete backends are named and wired
into a `Pipeline`.

Every other module in this codebase depends on Protocols (`Retriever`,
`LLMClient`, `PdfFetcher`, `PdfParser`, `EvidenceLocalizer`,
`SeedKnnExpander`, ...) -- this module is where the real local stack
(local pool JSONL, local sentence-transformer embeddings, BM25+dense RRF
hybrid retrieval, PyMuPDF parsing, Gemini) gets instantiated and injected.
The planned GCP swap (#2 in the tracker) edits ONLY `build_pipeline`;
`Pipeline` and everything it depends on stay untouched.
"""
from __future__ import annotations

import json
import os
import pathlib

from littraceqa.answer.pipeline import AnswerPipeline
from littraceqa.llm.interfaces import build_llm
from littraceqa.localize.fetch_local import LocalPdfFetcher
from littraceqa.localize.fetch_openreview import OpenReviewFetcher
from littraceqa.localize.localizer import LlmEvidenceLocalizer
from littraceqa.localize.parse_pymupdf import PyMuPdfParser
from littraceqa.localize.service import EvidenceLocalizationService
from littraceqa.paperset.dense_expander import DenseSeedKnnExpander
from littraceqa.paperset.selector import PaperSetSelector
from littraceqa.pipeline.orchestrator import Pipeline
from littraceqa.retrieval.bm25 import BM25Retriever
from littraceqa.retrieval.dense import DenseRetriever
from littraceqa.retrieval.embedder_local import LocalEmbedder
from littraceqa.retrieval.exact import ExactAcronymIndex
from littraceqa.retrieval.hybrid import RRFHybridRetriever
from littraceqa.retrieval.pool import DEFAULT_POOL_PATH, PoolIndex, load_pool
from littraceqa.seed.finder import SeedFinder


def _openreview_id_map(pool_path: str | pathlib.Path | None) -> dict[str, str]:
    """paper_id -> OpenReview forum id, read from the raw pool metadata
    (`Paper` doesn't carry `openreview_id`, so read the JSONL directly).
    Papers without an OpenReview id are simply omitted."""
    path = pathlib.Path(pool_path) if pool_path is not None else DEFAULT_POOL_PATH
    id_map: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            oid = rec.get("openreview_id")
            if oid:
                id_map[rec["paper_id"]] = oid
    return id_map


def build_pipeline(
    *,
    llm_name: str = "gemini",
    top_n_seeds: int = 3,
    use_expander: bool = False,
    expand_k: int = 10,
    evidence_confidence_floor: float = 0.5,
    pool_path: str | pathlib.Path | None = None,
    openreview_token: str | None = None,
    vision_model: str | None = "gemini-2.5-pro",
) -> Pipeline:
    """Construct the real local stack and inject it into a `Pipeline`.

    - `pool`: the full paper-metadata pool, loaded from `pool_path` (or the
      default `data/paper_metadata.jsonl` if `pool_path` is None).
    - `dense`: a `DenseRetriever` over a local sentence-transformer
      embedder; its constructor reuses `embed_pool`'s on-disk cache, so the
      ~27k-paper pool is not re-embedded on every run.
    - `hybrid`: BM25 + dense fused via Reciprocal Rank Fusion, passed to
      `SeedFinder` as its retriever (best recall of the two).
    - `dense` is ALSO passed directly to `DenseSeedKnnExpander` for the
      seed-kNN expansion lever (`retrieve_by_seed`), independent of the
      hybrid retriever `SeedFinder` uses.
    - `llm`: built once via the `build_llm` registry and shared by seed
      finding (anchor extraction, tier-parametric, confirmation) and
      evidence localization, so both consume the same backend/model.
    """
    papers = load_pool(pool_path) if pool_path is not None else load_pool()
    pool = PoolIndex(papers)

    embedder = LocalEmbedder()
    dense = DenseRetriever(pool, embedder)
    bm25 = BM25Retriever(pool)
    hybrid = RRFHybridRetriever([dense, bm25])

    exact = ExactAcronymIndex(pool)
    llm = build_llm(llm_name)

    seed_finder = SeedFinder(llm, pool, hybrid, exact)

    # OpenReview PDFs (~12k of the pool) 403 on direct HTTP; when a token is
    # available (env `OPENREVIEW_TOKEN`), wrap the local fetcher with the
    # OpenReview API fetcher so those papers are downloadable. Reproducible:
    # the token comes from env, not hardcoded.
    fetcher = LocalPdfFetcher()
    token = openreview_token if openreview_token is not None else os.getenv("OPENREVIEW_TOKEN")
    if token:
        id_map = _openreview_id_map(pool_path)
        fetcher = OpenReviewFetcher(fetcher, id_map, token=token)

    evidence_service = EvidenceLocalizationService(
        fetcher, PyMuPdfParser(), LlmEvidenceLocalizer(llm)
    )

    expander = DenseSeedKnnExpander(dense) if use_expander else None
    selector = PaperSetSelector(expander=expander, expand_k=expand_k)

    answer_pipeline = AnswerPipeline(evidence_confidence_floor=evidence_confidence_floor)

    # Figure/table VISION path uses a STRONGER multimodal model (gemini-2.5-pro
    # reads dense figures reliably where flash mis-reads them). Only the vision
    # sub-call pays the higher cost; all other steps stay on the cheaper `llm`.
    vision_llm = None
    if vision_model:
        from littraceqa.llm.gemini import GeminiClient
        vision_llm = GeminiClient(model=vision_model)

    return Pipeline(
        seed_finder=seed_finder,
        evidence_service=evidence_service,
        paperset_selector=selector,
        answer_pipeline=answer_pipeline,
        pool=pool,
        llm=llm,
        top_n_seeds=top_n_seeds,
        vision_llm=vision_llm,
    )
