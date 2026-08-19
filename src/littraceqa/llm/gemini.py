"""Gemini-backed `LLMClient`, via Google AI Studio (the `google-genai` SDK).

Confirmed against google-genai==2.16.0:

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=...)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    resp.text

`DEFAULT_MODEL` is a module constant naming the cheap dev default. To use a
different model (e.g. "gemini-2.5-pro" for hard steps, or a Vertex-Gemini
model id at the final), pass `model=` to the constructor:
`GeminiClient(model="gemini-2.5-pro")`. Note the default binds at
class-definition time, so reassigning `gemini.DEFAULT_MODEL` after import
does NOT change the constructor default — always override via the `model=`
kwarg.
"""

from __future__ import annotations

import os

from littraceqa.llm.interfaces import register_llm

DEFAULT_MODEL = "gemini-2.5-flash"


@register_llm("gemini")
class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set: export GEMINI_API_KEY=<your Google AI "
                "Studio key> or pass api_key= explicitly to GeminiClient()."
            )
        self.model_name = model
        self._api_key = api_key
        self._client = None  # lazily built so import never touches the network

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # Multimodal path: when PNG image bytes are supplied, send them as
        # inline image Parts BEFORE the text prompt so Gemini SEES the figure
        # page (a figure's answer lives in pixels, not in the extracted text).
        # No images -> the exact original text-only contents=prompt path.
        if images:
            contents: object = [
                types.Part.from_bytes(data=img, mime_type="image/png")
                for img in images
            ] + [prompt]
        else:
            contents = prompt
        resp = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        text = resp.text
        if text is None:
            # `GenerateContentResponse.text` is Optional[str]: it is None when
            # the response has no usable text part — safety/recitation blocks,
            # max_tokens truncation before any text, or a function-call-only
            # response. Fail loudly with the cause instead of returning None
            # and breaking the declared `-> str` contract downstream (#4/#7).
            finish_reason = None
            try:
                finish_reason = resp.candidates[0].finish_reason
            except (AttributeError, IndexError, TypeError):
                pass
            raise RuntimeError(
                f"Gemini returned no text (finish_reason={finish_reason}); "
                "the response was blocked, truncated, or contained no text part."
            )
        return text
