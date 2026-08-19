"""Answer emitters.

Builds the `answer` payloads the LitTraceQA scorer expects. Dispatch is
OCP: each answer type contributes a serializer registered with `register`;
`serialize` looks the serializer up and never needs to change when a new
answer type is added.

Scorer-contract landmines encoded here (verified against `vendor/evaluate.py`):
- `freeform` text is emitted VERBATIM — never reformat numbers/units (e.g.
  a trailing zero in "14.70" must survive byte-for-byte).
- `multiple_choice` puts the predicted letter in a field literally named
  `gold`, uppercased — not the option text.
- `table` cell values pass through unchanged: numeric cells stay JSON
  numbers (never stringified) and `None` stays `None`.
"""

from typing import Callable

Serializer = Callable[[object], dict]
_SERIALIZERS: dict[str, Serializer] = {}


def register(answer_type: str) -> Callable[[Serializer], Serializer]:
    """Decorator: register `fn` as the serializer for `answer_type`.

    This is the sole extension point — adding a new answer type never
    requires touching `serialize`.
    """
    def deco(fn: Serializer) -> Serializer:
        _SERIALIZERS[answer_type] = fn
        return fn
    return deco


def serialize(answer_type: str, value) -> dict:
    """Build the `answer` payload for `answer_type`, dispatched through the
    `register` registry. Closed for modification: a new answer type is
    added by registration, not by editing this function."""
    if answer_type not in _SERIALIZERS:
        raise KeyError(f"no serializer registered for answer_type={answer_type!r}")
    return _SERIALIZERS[answer_type](value)


@register("freeform")
def freeform(text: str) -> dict:
    return {"freeform": {"text": text}}  # verbatim — never reformat


@register("multiple_choice")
def multiple_choice(letter: str) -> dict:
    return {"multiple_choice": {"gold": letter.upper()}}


@register("table")
def table(rows: list) -> dict:
    return {"table": {"rows": rows}}  # values pass through: numbers stay numbers, None stays None
