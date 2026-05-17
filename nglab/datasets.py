"""Dataset builders for neural geometry experiments.

The functions return pandas DataFrames with at least:
  prompt       - text sent to the model
  target       - text span whose hidden state should be extracted, when present
  target_start - character start offset of target inside prompt, or -1
  target_end   - character end offset of target inside prompt, or -1
  value        - numeric/cyclic value used for probes
  label        - display label

The offset fields let a fast tokenizer extract the hidden state of the concept token(s),
rather than blindly taking the final token.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
HOURS = [f"{h:02d}:00" for h in range(24)]

WEEKDAY_CONTEXTS = [
    "Today is {x}.",
    "The day of the week is {x}.",
    "I have a meeting on {x}.",
    "The calendar says it is {x}.",
    "This happens every {x}.",
    "The weekly schedule starts on {x}.",
    "Please mark {x} on the planner.",
    "The next class is on {x}.",
]

MONTH_CONTEXTS = [
    "The month is {x}.",
    "The calendar month is {x}.",
    "The event takes place in {x}.",
    "Please mark the report for {x}.",
    "The season changes around {x}.",
    "The schedule starts in {x}.",
    "The invoice is dated {x}.",
    "The deadline falls in {x}.",
]

NUMBER_CONTEXTS = [
    "The number is {x}.",
    "Answer: {x}",
    "Value = {x}.",
    "I counted {x} objects.",
    "The score was {x}.",
    "There are {x} items in the list.",
]

ADDITION_CONTEXTS = [
    "Q: What is {a} + {b}?\nA:",
    "Compute {a} + {b}. Answer:",
    "{a} + {b} =",
]

WEEKDAY_CYCLIC_CONTEXTS = [
    "Q: What day is {offset} days after {base}?\nA:",
    "If today is {base}, what day will it be in {offset} days? Answer:",
    "Starting from {base}, {offset} days later is",
]

MONTH_CYCLIC_CONTEXTS = [
    "Q: What month is {offset} months after {base}?\nA:",
    "If it is {base}, what month will it be in {offset} months? Answer:",
    "Starting from {base}, {offset} months later is",
]

HOUR_CYCLIC_CONTEXTS = [
    "Q: It is currently {base}. What time will it be in {offset} hours?\nA:",
    "Starting at {base}, {offset} hours later it is",
    "If the clock reads {base}, after {offset} hours it will read",
]


def _row_from_template(template: str, x: str, value: int, label: str, *, meta: dict | None = None) -> dict:
    before, after = template.split("{x}", 1)
    prompt = before + x + after
    row = {
        "prompt": prompt,
        "target": x,
        "target_start": len(before),
        "target_end": len(before) + len(x),
        "value": int(value),
        "label": label,
        "context": template,
    }
    if meta:
        row.update(meta)
    return row


def build_concept_dataset(labels: Sequence[str], contexts: Sequence[str], *, start_index: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, label in enumerate(labels):
        value = start_index + idx
        for template in contexts:
            rows.append(_row_from_template(template, label, value, label))
    return pd.DataFrame(rows)


def make_weekday_dataset(contexts: Sequence[str] | None = None, *, monday_is: int = 0) -> pd.DataFrame:
    """Return prompts containing weekday names.

    monday_is=0 is convenient for modulo-7 math. Set monday_is=1 if you prefer
    human 1-based indexing; the circular probes only care about values modulo 7.
    """
    return build_concept_dataset(WEEKDAYS, contexts or WEEKDAY_CONTEXTS, start_index=monday_is)


def make_month_dataset(contexts: Sequence[str] | None = None, *, january_is: int = 1) -> pd.DataFrame:
    """Return prompts containing month names.

    january_is=1 mirrors Goodfire's explanation where August is 8; set 0 for
    zero-based modular arithmetic.
    """
    return build_concept_dataset(MONTHS, contexts or MONTH_CONTEXTS, start_index=january_is)


def make_number_dataset(
    start: int = 0,
    end: int = 99,
    contexts: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for n in range(start, end + 1):
        label = str(n)
        for template in contexts or NUMBER_CONTEXTS:
            rows.append(_row_from_template(template, label, n, label))
    return pd.DataFrame(rows)


def make_addition_dataset(
    max_a: int = 20,
    max_b: int = 20,
    contexts: Sequence[str] | None = None,
    *,
    min_a: int = 0,
    min_b: int = 0,
) -> pd.DataFrame:
    """Return addition prompts ending just before the model should answer.

    Extraction target is absent; activation extraction should use the final token.
    """
    rows: list[dict] = []
    for a in range(min_a, max_a + 1):
        for b in range(min_b, max_b + 1):
            for template in contexts or ADDITION_CONTEXTS:
                prompt = template.format(a=a, b=b)
                rows.append(
                    {
                        "prompt": prompt,
                        "target": "",
                        "target_start": -1,
                        "target_end": -1,
                        "value": a + b,
                        "sum": a + b,
                        "a": a,
                        "b": b,
                        "label": str(a + b),
                        "context": template,
                    }
                )
    return pd.DataFrame(rows)


def make_cyclic_addition_dataset(
    kind: str = "weekday",
    max_offset: int | None = None,
    contexts: Sequence[str] | None = None,
    *,
    base_start_index: int | None = None,
) -> pd.DataFrame:
    """Return prompts like "what day is 3 days after Friday?".

    Columns:
      base_value       - numeric value of the starting concept
      offset           - addition offset
      premod_sum       - base_value + offset
      output_value     - answer value modulo period, in same indexing scheme
      output_label     - expected answer label
    """
    kind = kind.lower()
    if kind in {"weekday", "weekdays", "day", "days"}:
        labels, period = WEEKDAYS, 7
        templates = contexts or WEEKDAY_CYCLIC_CONTEXTS
        start = 0 if base_start_index is None else base_start_index
    elif kind in {"month", "months"}:
        labels, period = MONTHS, 12
        templates = contexts or MONTH_CYCLIC_CONTEXTS
        start = 1 if base_start_index is None else base_start_index
    elif kind in {"hour", "hours", "time"}:
        labels, period = HOURS, 24
        templates = contexts or HOUR_CYCLIC_CONTEXTS
        start = 0 if base_start_index is None else base_start_index
    else:
        raise ValueError(f"Unknown cyclic kind: {kind!r}")

    if max_offset is None:
        max_offset = period - 1

    rows: list[dict] = []
    for i, base in enumerate(labels):
        base_value = start + i
        for offset in range(0, max_offset + 1):
            premod_sum = base_value + offset
            # Convert back into label index while respecting one-based months.
            output_index = (premod_sum - start) % period
            output_label = labels[output_index]
            output_value = start + output_index
            for template in templates:
                prompt = template.format(base=base, offset=offset)
                rows.append(
                    {
                        "prompt": prompt,
                        "target": "",
                        "target_start": -1,
                        "target_end": -1,
                        "value": premod_sum,
                        "premod_sum": premod_sum,
                        "output_value": output_value,
                        "output_label": output_label,
                        "base": base,
                        "base_value": base_value,
                        "offset": offset,
                        "period": period,
                        "label": f"{base}+{offset}->{output_label}",
                        "context": template,
                        "kind": kind,
                    }
                )
    return pd.DataFrame(rows)
