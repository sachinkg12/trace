"""LLM client interfaces: the `LLMClient` Protocol and the LLM registry.

Dispatch is OCP, mirroring `littraceqa.retrieval.interfaces`: each backend
(FakeLLM, GeminiClient, and later Claude/OpenAI/Vertex-Gemini) registers
itself with `register_llm`; `build_llm` looks it up by name and never needs
to change when a new backend is added. `LLMClient` is a `Protocol` (DIP) so
seed-finding (#4) and the answer pipeline (#7) can depend on the interface
rather than a concrete class, and swap backends (e.g. for a bake-off, or
Gemini -> Vertex-Gemini at the final) with zero changes to calling code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Callable


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can complete a prompt and report which model answered.

    `model_name` MUST be exposed (not just `complete`): the rules require
    reporting the model used for every answer, so callers need this without
    reaching into backend-specific internals.
    """

    model_name: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        images: list[bytes] | None = None,
    ) -> str: ...


_LLMS: dict[str, Callable[..., LLMClient]] = {}


def register_llm(name: str):
    """Decorator: register `cls` as the LLM builder for `name`.

    This is the sole extension point — adding a new LLM backend never
    requires touching `build_llm`.
    """
    def deco(cls):
        if name in _LLMS:
            raise ValueError(f"llm {name!r} already registered")
        _LLMS[name] = cls
        return cls
    return deco


def build_llm(name: str, **kwargs) -> LLMClient:
    """Build the LLM client registered as `name`, dispatched through the
    `register_llm` registry. Closed for modification: a new LLM backend is
    added by registration, not by editing this function."""
    if name not in _LLMS:
        raise KeyError(f"no llm registered as {name!r}; have {sorted(_LLMS)}")
    return _LLMS[name](**kwargs)
