"""Deterministic `LLMClient` test double.

`FakeLLM` is what seed-finding (#4) and the answer pipeline (#7) unit-test
against, so it needs to be genuinely usable, not a stub: it accepts a list
of scripted responses (returned in call order), a dict keyed by exact
prompt, or a callable `(prompt) -> str`, and it records every call so tests
can assert on what was asked (prompt, system, temperature, max_tokens).
"""

from __future__ import annotations

from typing import Callable

from littraceqa.llm.interfaces import register_llm


@register_llm("fake")
class FakeLLM:
    model_name = "fake"

    def __init__(
        self,
        responses: list[str] | dict[str, str] | Callable[[str], str] | None = None,
        default: str = "ok",
    ):
        self._fn: Callable[[str], str] | None = None
        self._by_prompt: dict[str, str] | None = None
        self._sequence: list[str] | None = None

        if callable(responses):
            self._fn = responses
        elif isinstance(responses, dict):
            self._by_prompt = responses
        elif responses is not None:
            self._sequence = list(responses)

        self._default = default
        self._next = 0
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        # `images` is RECORDED (so vision tests can assert the figure PNG was
        # passed) but never affects the scripted output -- FakeLLM stays a
        # deterministic text double regardless of modality.
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "images": images,
            }
        )

        if self._fn is not None:
            return self._fn(prompt)
        if self._by_prompt is not None:
            return self._by_prompt.get(prompt, self._default)
        if self._sequence is not None:
            if self._next < len(self._sequence):
                resp = self._sequence[self._next]
                self._next += 1
                return resp
            return self._default
        return self._default
