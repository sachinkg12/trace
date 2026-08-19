"""Local sentence-transformer `Embedder` + cached pool embeddings.

Implements the `Embedder` Protocol (`interfaces.py`) with a local
sentence-transformer model, and `embed_pool`, which embeds every document
in a `PoolIndex` and caches the result to disk so the (currently
27,487-paper) pool doesn't get re-embedded on every run.

CRITICAL usage constraint (from the verified `analysis/` ceiling work,
`analysis/embeddings.py`): documents/passages (pool docs, seed papers) are
embedded SYMMETRICALLY, with NO query-instruction prefix --
`model.encode(texts, normalize_embeddings=True)`. This module is
passage-to-passage usage only. A future query-text retriever may need a
BGE-style query prefix ("Represent this sentence for searching relevant
passages: ..."); that is out of scope here and must live in the query
encoding path, not this one.

Model: BAAI/bge-small-en-v1.5 (384-dim, CPU-friendly for a ~27k-document
pool), falling back to sentence-transformers/all-MiniLM-L6-v2 if the
primary can't be loaded (e.g. an offline re-run without cached HF
weights). The model is loaded lazily and cached at module scope keyed by
resolved model name, so repeated `LocalEmbedder()` construction across a
process never reloads weights.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from littraceqa.retrieval.interfaces import Embedder
from littraceqa.retrieval.pool import PoolIndex

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_POOL_EMB_CACHE = REPO_ROOT / "data" / "cache" / "pool_emb.npy"

PRIMARY_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level model cache, keyed by the model name that actually loaded
# (not necessarily the name requested, if a fallback happened). Shared
# across every `LocalEmbedder` instance in the process so the expensive
# part (loading transformer weights) happens at most once per model.
_model_cache: dict[str, object] = {}


def _load_model(model_name: str):
    """Lazily load (and process-cache) a `SentenceTransformer` by name.

    Falls back to `FALLBACK_MODEL` if `model_name` is `PRIMARY_MODEL` and
    fails to load (e.g. no network / weights not cached locally). Returns
    `(model, resolved_name)`.
    """
    from sentence_transformers import SentenceTransformer

    candidates = [model_name]
    if model_name == PRIMARY_MODEL and FALLBACK_MODEL not in candidates:
        candidates.append(FALLBACK_MODEL)

    last_err: Exception | None = None
    for cand in candidates:
        if cand in _model_cache:
            return _model_cache[cand], cand
        try:
            model = SentenceTransformer(cand, device="cpu")
        except Exception as e:  # noqa: BLE001 -- deliberately broad, we just fall back
            last_err = e
            continue
        _model_cache[cand] = model
        return model, cand
    raise RuntimeError(
        f"Could not load embedding model {model_name!r} or fallback {FALLBACK_MODEL!r}: {last_err}"
    )


class LocalEmbedder:
    """`Embedder` Protocol implementation backed by a local
    sentence-transformer.

    Symmetric passage encoding only -- see module docstring for why no
    query-instruction prefix is applied here.
    """

    def __init__(self, model_name: str = PRIMARY_MODEL):
        self.requested_model_name = model_name
        self._model, self.model_name = _load_model(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalized embeddings, shape `(len(texts), dim)`, float32.

        Empty or near-empty abstracts are valid input: the title remains in
        ``doc_text``, and the tokenizer still yields a normalized embedding.
        """
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


def _pool_id_hash(ids: list[str]) -> str:
    """Stable, order-sensitive hash of a pool's id list.

    Used as part of `embed_pool`'s cache key so an edited pool (papers
    added/removed/reordered) invalidates a stale cache instead of
    silently returning misaligned embeddings.
    """
    h = hashlib.sha256()
    for pid in ids:
        h.update(pid.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _ids_cache_path(cache_path: pathlib.Path) -> pathlib.Path:
    """`data/cache/pool_emb.npy` -> `data/cache/pool_emb.ids.json`."""
    return cache_path.with_name(cache_path.stem + ".ids.json")


def embed_pool(
    pool_index: PoolIndex,
    embedder: Embedder,
    cache_path: str | pathlib.Path = DEFAULT_POOL_EMB_CACHE,
) -> tuple[np.ndarray, list[str]]:
    """Embed every document in `pool_index`, cached to `cache_path` (plus
    a sibling `*.ids.json` file).

    `embedder` is any `Embedder` Protocol implementation -- `LocalEmbedder`
    or otherwise (e.g. a future Vertex-backed embedder) -- so this function
    never needs to change to support a new backend.

    Cache key is `(embedder.model_name, hash of pool_index.ids)`. If
    either changes -- a different model, or the pool gained/lost/reordered
    papers -- the cache is treated as stale and rebuilt. Returns
    `(embeddings, ids)` where `embeddings[i]` corresponds to `ids[i]`.

    CAVEAT: the id-set hash is order-sensitive over `paper_id`s only, not
    over each paper's `title`/`abstract` text. Editing a paper's title or
    abstract WITHOUT changing the pool's paper_id set will NOT invalidate
    this cache -- stale embeddings will silently be reused. This is fine
    for the frozen shared-task pool (paper text doesn't change), but is a
    trap for any future pool that gets edited in place.
    """
    cache_path = pathlib.Path(cache_path)
    ids_path = _ids_cache_path(cache_path)

    ids = pool_index.ids
    id_hash = _pool_id_hash(ids)

    if cache_path.exists() and ids_path.exists():
        cached = json.loads(ids_path.read_text())
        if (
            cached.get("model_name") == embedder.model_name
            and cached.get("id_hash") == id_hash
            and cached.get("ids") == ids
        ):
            emb = np.load(cache_path)
            if emb.shape[0] == len(ids):
                return emb, cached["ids"]
        # else: model changed, or pool ids changed -- fall through and rebuild

    texts = [pool_index.doc_text(pid) for pid in ids]
    emb = embedder.embed(texts)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    ids_path.write_text(json.dumps({"model_name": embedder.model_name, "id_hash": id_hash, "ids": ids}))

    return emb, ids
