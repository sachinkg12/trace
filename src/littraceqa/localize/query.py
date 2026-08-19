"""Build gold-blind evidence queries from a system's own predicted answer."""
from __future__ import annotations

from typing import Any

from littraceqa.pipeline.input import InputRecord


MAX_ANSWER_CLUE_CHARS = 2000


def _scalar(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None


def _answer_clues(record: InputRecord, answer: Any) -> list[str]:
    if not isinstance(answer, dict):
        return []
    clues: list[str] = []

    multiple_choice = answer.get("multiple_choice")
    if isinstance(multiple_choice, dict):
        label = str(
            multiple_choice.get("gold")
            or multiple_choice.get("answer")
            or multiple_choice.get("predicted_answer_id")
            or ""
        ).strip().upper()
        option = (record.mc_options or {}).get(label)
        if option and option.strip():
            clues.append(option.strip())

    freeform = answer.get("freeform")
    if isinstance(freeform, dict):
        text = _scalar(freeform.get("text"))
        if text:
            clues.append(text)

    table = answer.get("table")
    rows = table.get("rows") if isinstance(table, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            cells = [text for value in row.values() if (text := _scalar(value))]
            if cells:
                clues.append(" | ".join(cells))

    # Preserve answer order, but do not let repeated table cells enlarge the
    # prompt or overweight a value merely because it appeared twice.
    return list(dict.fromkeys(clues))


def build_answer_aware_evidence_query(
    record: InputRecord,
    answer: Any,
    *,
    max_clue_chars: int = MAX_ANSWER_CLUE_CHARS,
) -> str:
    """Append bounded predicted-answer clues without treating them as truth."""
    if (
        isinstance(max_clue_chars, bool)
        or not isinstance(max_clue_chars, int)
        or max_clue_chars < 1
    ):
        raise ValueError("max_clue_chars must be a positive integer")
    clues = _answer_clues(record, answer)
    if not clues:
        return record.question
    rendered = "\n".join(f"- {clue}" for clue in clues)
    rendered = rendered[:max_clue_chars].rstrip()
    return (
        f"{record.question}\n\n"
        "The system predicted the answer below. Use it only as a search clue; "
        "emit evidence only if the paper itself verifies it:\n"
        f"{rendered}"
    )
