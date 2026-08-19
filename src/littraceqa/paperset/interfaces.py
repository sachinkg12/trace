"""Paper-set selection contracts: the `SeedKnnExpander` Protocol + registry.

Dispatch is OCP (mirrors littraceqa.retrieval.interfaces): a new expander
backend registers itself; `build_expander` looks it up and never changes.
The Protocol is the DIP seam so `PaperSetSelector` depends on the interface,
not a concrete dense/Vertex implementation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Callable


@runtime_checkable
class SeedKnnExpander(Protocol):
    """Given seed paper_ids, return up to `k` nearest-neighbour papers
    (paper_id, score), seeds excluded — the recall-expansion lever."""

    def expand(self, seed_ids: list[str], k: int) -> list[tuple[str, float]]: ...


_EXPANDERS: dict[str, Callable[..., SeedKnnExpander]] = {}


def register_expander(name: str):
    def deco(cls):
        if name in _EXPANDERS:
            raise ValueError(f"expander {name!r} already registered")
        _EXPANDERS[name] = cls
        return cls
    return deco


def build_expander(name: str, **kwargs) -> SeedKnnExpander:
    if name not in _EXPANDERS:
        raise KeyError(f"no expander registered as {name!r}; have {sorted(_EXPANDERS)}")
    return _EXPANDERS[name](**kwargs)
