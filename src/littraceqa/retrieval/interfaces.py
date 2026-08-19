"""Retrieval core interfaces: `Paper`, `Embedder`/`Retriever` Protocols, and
the retriever registry.

Dispatch is OCP: each retriever implementation registers itself with
`register_retriever`; `build_retriever` looks it up by name and never needs
to change when a new retriever backend is added. `Embedder` and `Retriever`
are `Protocol`s (DIP) so concrete implementations (e.g. a Vertex-backed
embedder) can be injected without callers depending on a concrete class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Callable
import numpy as np


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    venue: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    pdf_url: str | None = None


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into L2-normalized embedding vectors.

    `model_name` MUST be exposed (not just `embed`): `embed_pool` uses it
    as part of its disk-cache key, so a Vertex-backed (or any other)
    implementation still cache-keys correctly with zero changes to
    `embed_pool`/`DenseRetriever`.
    """

    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]: ...


_RETRIEVERS: dict[str, Callable[..., Retriever]] = {}


def register_retriever(name: str):
    """Decorator: register `cls` as the retriever builder for `name`.

    This is the sole extension point — adding a new retriever backend never
    requires touching `build_retriever`.
    """
    def deco(cls):
        if name in _RETRIEVERS:
            raise ValueError(f"retriever {name!r} already registered")
        _RETRIEVERS[name] = cls
        return cls
    return deco


def build_retriever(name: str, **kwargs) -> Retriever:
    """Build the retriever registered as `name`, dispatched through the
    `register_retriever` registry. Closed for modification: a new retriever
    is added by registration, not by editing this function."""
    if name not in _RETRIEVERS:
        raise KeyError(f"no retriever registered as {name!r}; have {sorted(_RETRIEVERS)}")
    return _RETRIEVERS[name](**kwargs)
