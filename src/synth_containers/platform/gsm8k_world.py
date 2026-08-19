"""GSM8K one-turn world. Gold stays private; public observation is the question.

Fixture rows are the PR CI source. HuggingFace ``openai/gsm8k`` is opt-in via
``SYNTH_GSM8K_SOURCE=hf`` so tests never download — the same switch Banking77
uses (``SYNTH_BANKING77_SOURCE=hf``).

Two things this module refuses to blur:

- the reference answer never reaches :func:`public_observation`;
- a completion that cannot be parsed is *unparsed*, not wrong. :func:`parse_answer`
  returns the reason and the raw text alongside the value, and the runtime keeps
  the reward ``None`` for that case rather than scoring a zero nobody measured.

See: workshop/docs/aug_12_update.md §2.2 (Environment / Policy / TaskWorld).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import Any


SOLVE_SYSTEM = (
    "Solve the grade-school math word problem. Reason step by step, then state the "
    "final numeric answer on its own last line in the form '#### <answer>'. The final "
    "answer must be a bare number with no units, no currency symbol and no words."
)

WORLD_REF_PREFIX = "world:gsm8k"
TRAIN_SPLIT = "train"
HELDOUT_SPLIT = "heldout"

#: Deterministic split indices into :data:`_FIXTURE_POOL`, persisted here rather
#: than in a data file: ``pyproject.toml`` declares no ``package-data``, so a
#: JSON sidecar would be present in the checkout and missing from the wheel —
#: the split would then differ between a source run and an installed run, which
#: is precisely the failure the persisted split exists to prevent.
TRAIN_INDICES: tuple[int, ...] = tuple(range(0, 16))
HELDOUT_INDICES: tuple[int, ...] = tuple(range(16, 32))


@dataclass(frozen=True)
class Gsm8kRow:
    question: str
    answer_text: str

    @property
    def answer(self) -> str:
        """Normalized reference answer. Env-private."""
        parsed = parse_answer(self.answer_text)
        return parsed.value or ""


@dataclass(frozen=True)
class ParsedAnswer:
    """Result of reading a completion. ``value is None`` means *unparsed*.

    ``raw`` is kept next to ``value`` so a run can tell "the policy said nothing
    a parser could read" apart from "the policy answered and was wrong". Those
    two produce different rewards (``None`` versus ``0.0``) and collapsing them
    is how a broken prompt reads as a bad model.
    """

    value: str | None
    source: str
    raw: str

    @property
    def parsed(self) -> bool:
        return self.value is not None


_FIXTURE_POOL: tuple[Gsm8kRow, ...] = (
    Gsm8kRow(
        "Natalia sold clips to 48 of her friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether in April and May?",
        "In May she sold 48 / 2 = 24 clips.\nAltogether she sold 48 + 24 = 72 clips.\n#### 72",
    ),
    Gsm8kRow(
        "Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes of "
        "babysitting. How much did she earn in dollars?",
        "Weng earns 12 / 60 = 0.2 dollars per minute.\nFor 50 minutes she earned "
        "0.2 * 50 = 10 dollars.\n#### 10",
    ),
    Gsm8kRow(
        "Betty is saving for a wallet that costs $100. She has half of the money she "
        "needs. Her parents give her $15 and her grandparents give her twice as much "
        "as her parents. How much more money does Betty need?",
        "Betty already has 100 / 2 = 50 dollars.\nHer grandparents give 15 * 2 = 30 "
        "dollars.\nShe now has 50 + 15 + 30 = 95 dollars.\nShe still needs 100 - 95 = 5 "
        "dollars.\n#### 5",
    ),
    Gsm8kRow(
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
        "bolts does it take in total?",
        "The white fiber is 2 / 2 = 1 bolt.\nIn total that is 2 + 1 = 3 bolts.\n#### 3",
    ),
    Gsm8kRow(
        "James writes a 3-page letter to 2 different friends twice a week. How many "
        "pages does he write in a year of 52 weeks?",
        "Each week he writes 3 * 2 * 2 = 12 pages.\nIn a year that is 12 * 52 = 624 "
        "pages.\n#### 624",
    ),
    Gsm8kRow(
        "Mark has 4 boxes with 9 pencils in each box. He gives away 11 pencils. How "
        "many pencils does he have left?",
        "Mark starts with 4 * 9 = 36 pencils.\nAfter giving some away he has 36 - 11 = "
        "25 pencils.\n#### 25",
    ),
    Gsm8kRow(
        "A bakery sells 20 loaves of bread in the morning and three times as many in "
        "the afternoon. How many loaves did it sell that day?",
        "In the afternoon it sold 20 * 3 = 60 loaves.\nThat day it sold 20 + 60 = 80 "
        "loaves.\n#### 80",
    ),
    Gsm8kRow(
        "Sara runs 5 kilometers on each of the five weekdays and 8 kilometers on "
        "Saturday. She rests on Sunday. How many kilometers does she run in a week?",
        "On weekdays she runs 5 * 5 = 25 kilometers.\nAdding Saturday gives 25 + 8 = 33 "
        "kilometers.\n#### 33",
    ),
    Gsm8kRow(
        "A pack of 6 water bottles costs $9. How much does one bottle cost in dollars?",
        "One bottle costs 9 / 6 = 1.5 dollars.\n#### 1.5",
    ),
    Gsm8kRow(
        "Tom buys 3 shirts at $14 each and pays with a $50 bill. How much change does "
        "he get in dollars?",
        "The shirts cost 3 * 14 = 42 dollars.\nHis change is 50 - 42 = 8 dollars.\n#### 8",
    ),
    Gsm8kRow(
        "A classroom has 7 rows of 6 desks. If 4 of the desks are broken, how many "
        "usable desks are there?",
        "There are 7 * 6 = 42 desks.\nUsable desks are 42 - 4 = 38.\n#### 38",
    ),
    Gsm8kRow(
        "Lisa reads 26 pages on Monday and 15 fewer pages on Tuesday. How many pages "
        "did she read over the two days?",
        "On Tuesday she read 26 - 15 = 11 pages.\nOver two days she read 26 + 11 = 37 "
        "pages.\n#### 37",
    ),
    Gsm8kRow(
        "A farmer collects 120 eggs and packs them into cartons that hold 8 eggs each. "
        "How many cartons does he fill?",
        "He fills 120 / 8 = 15 cartons.\n#### 15",
    ),
    Gsm8kRow(
        "A movie ticket costs $9 and a box of popcorn costs $6. How much do 4 tickets "
        "and 2 boxes of popcorn cost in dollars?",
        "The tickets cost 4 * 9 = 36 dollars.\nThe popcorn costs 2 * 6 = 12 dollars.\n"
        "Together that is 36 + 12 = 48 dollars.\n#### 48",
    ),
    Gsm8kRow(
        "Anna had $75. She spent 40% of it on books. How many dollars does she have "
        "left?",
        "She spent 75 * 0.4 = 30 dollars.\nShe has 75 - 30 = 45 dollars left.\n#### 45",
    ),
    Gsm8kRow(
        "A tank holds 250 liters and is 60% full. How many more liters are needed to "
        "fill it?",
        "The tank holds 250 * 0.6 = 150 liters now.\nIt needs 250 - 150 = 100 more "
        "liters.\n#### 100",
    ),
    Gsm8kRow(
        "A gardener plants 8 rows of 7 tulips. If 9 of the tulips do not bloom, how "
        "many tulips bloom?",
        "The gardener planted 8 * 7 = 56 tulips.\nOf those 56 - 9 = 47 bloom.\n#### 47",
    ),
    Gsm8kRow(
        "Ben bikes 12 miles a day for 5 days and then takes a day off. How many miles "
        "does he bike over those 6 days?",
        "He bikes 12 * 5 = 60 miles.\nThe rest day adds 0 miles, so the total is 60.\n"
        "#### 60",
    ),
    Gsm8kRow(
        "A shirt costs $28 and is on sale for 25% off. What is the sale price in "
        "dollars?",
        "The discount is 28 * 0.25 = 7 dollars.\nThe sale price is 28 - 7 = 21 "
        "dollars.\n#### 21",
    ),
    Gsm8kRow(
        "Maria buys 3 notebooks at $4 each and 2 pens at $2 each. How many dollars does "
        "she spend?",
        "The notebooks cost 3 * 4 = 12 dollars.\nThe pens cost 2 * 2 = 4 dollars.\n"
        "She spends 12 + 4 = 16 dollars.\n#### 16",
    ),
    Gsm8kRow(
        "A recipe needs 3/4 of a cup of sugar for one batch. How many cups of sugar are "
        "needed for 6 batches?",
        "Six batches need 6 * 3 / 4 = 4.5 cups.\n#### 4.5",
    ),
    Gsm8kRow(
        "A train travels 180 kilometers in 3 hours. At the same speed, how far does it "
        "travel in 5 hours?",
        "The train travels 180 / 3 = 60 kilometers per hour.\nIn 5 hours it travels "
        "60 * 5 = 300 kilometers.\n#### 300",
    ),
    Gsm8kRow(
        "Jenna has 5 more marbles than Kyle. Kyle has 22 marbles. How many marbles do "
        "they have together?",
        "Jenna has 22 + 5 = 27 marbles.\nTogether they have 27 + 22 = 49 marbles.\n"
        "#### 49",
    ),
    Gsm8kRow(
        "A theater has 14 rows with 20 seats in each row. If 55 seats are empty, how "
        "many seats are taken?",
        "The theater has 14 * 20 = 280 seats.\nTaken seats are 280 - 55 = 225.\n#### 225",
    ),
    Gsm8kRow(
        "Paul saves $35 a month for 8 months and then spends $90. How many dollars has "
        "he saved?",
        "He saves 35 * 8 = 280 dollars.\nAfter spending he has 280 - 90 = 190 "
        "dollars.\n#### 190",
    ),
    Gsm8kRow(
        "A pizza is cut into 8 slices. Three friends each eat 2 slices. How many slices "
        "are left?",
        "The friends eat 3 * 2 = 6 slices.\nThat leaves 8 - 6 = 2 slices.\n#### 2",
    ),
    Gsm8kRow(
        "A book has 315 pages. Dana reads 45 pages a day. How many days does it take "
        "her to finish the book?",
        "She needs 315 / 45 = 7 days.\n#### 7",
    ),
    Gsm8kRow(
        "Carl earns $18 for each lawn he mows and mows 7 lawns. He spends $31 on gas. "
        "How many dollars of profit does he make?",
        "He earns 18 * 7 = 126 dollars.\nHis profit is 126 - 31 = 95 dollars.\n#### 95",
    ),
    Gsm8kRow(
        "A rectangle is 9 meters long and 4 meters wide. What is its perimeter in "
        "meters?",
        "The perimeter is 2 * (9 + 4) = 26 meters.\n#### 26",
    ),
    Gsm8kRow(
        "A store sold 240 apples on Friday and one third as many on Saturday. How many "
        "apples did it sell in total?",
        "On Saturday it sold 240 / 3 = 80 apples.\nIn total it sold 240 + 80 = 320 "
        "apples.\n#### 320",
    ),
    Gsm8kRow(
        "Nina has 2.5 liters of juice and pours 400 milliliters into each glass. How "
        "many full glasses can she pour?",
        "She has 2.5 * 1000 = 2500 milliliters.\nThat fills 2500 / 400 = 6.25 glasses, "
        "so 6 full glasses.\n#### 6",
    ),
    Gsm8kRow(
        "A team scored 12 points in the first half and twice as many in the second "
        "half. How many points did the team score in total?",
        "In the second half the team scored 12 * 2 = 24 points.\nThe total is 12 + 24 = "
        "36 points.\n#### 36",
    ),
)

_TRAIN_FIXTURE: tuple[Gsm8kRow, ...] = tuple(_FIXTURE_POOL[index] for index in TRAIN_INDICES)
_HELDOUT_FIXTURE: tuple[Gsm8kRow, ...] = tuple(_FIXTURE_POOL[index] for index in HELDOUT_INDICES)


def user_prompt(question: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        "Show your reasoning, then end with a line of the form '#### <answer>'."
    )


# `\boxed{...}` first, then GSM8K's own `#### N`, then the last bare number.
# Order matters: a completion that shows its arithmetic contains many numbers and
# only the marked one is a claim about the answer.
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_HASH = re.compile(r"####[ \t]*([^\n]*)")
_NUMBER = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?(?:[ \t]*/[ \t]*[-+]?\d[\d,]*(?:\.\d+)?)?"
)

PARSE_SOURCE_BOXED = "boxed"
PARSE_SOURCE_HASH = "hash_marker"
PARSE_SOURCE_TRAILING = "trailing_number"
PARSE_SOURCE_NONE = "unparsed"


def normalize_number(token: str) -> str | None:
    """Canonical decimal string for one numeric token, or ``None``.

    Handles a leading sign, thousands separators, decimals and ``a/b`` fractions.
    The canonical form is what exact match compares, so ``1,000``, ``+1000`` and
    ``1000.0`` are one answer and ``3/4`` and ``0.75`` are one answer.
    """
    text = (token or "").strip().strip("$").strip()
    text = text.rstrip(".,;:!?")
    text = text.replace(",", "").replace(" ", "").replace("\t", "")
    if not text:
        return None
    try:
        if "/" in text:
            numerator, _, denominator = text.partition("/")
            if not numerator or not denominator:
                return None
            value = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        else:
            value = Fraction(Decimal(text))
    except (ArithmeticError, ValueError):
        return None
    return _canonical(value)


def _canonical(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    residue = value.denominator
    for prime in (2, 5):
        while residue % prime == 0:
            residue //= prime
    if residue != 1:
        # Non-terminating: keep the reduced fraction rather than inventing a
        # rounding rule that would make 1/3 and 0.333 compare equal by accident.
        return f"{value.numerator}/{value.denominator}"
    with localcontext() as context:
        context.prec = 60
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
    text = format(decimal.normalize(), "f")
    return "0" if text in {"-0", "0"} else text


def _first_number(text: str) -> str | None:
    match = _NUMBER.search(text or "")
    return normalize_number(match.group(0)) if match else None


def _last_number(text: str) -> str | None:
    matches = _NUMBER.findall(text or "")
    for candidate in reversed(matches):
        normalized = normalize_number(candidate)
        if normalized is not None:
            return normalized
    return None


def parse_answer(completion: str | None) -> ParsedAnswer:
    """Deterministic read of a completion. Never raises; never guesses a zero."""
    raw = completion if isinstance(completion, str) else ""
    boxed = _BOXED.findall(raw)
    if boxed:
        value = _first_number(boxed[-1])
        if value is not None:
            return ParsedAnswer(value, PARSE_SOURCE_BOXED, raw)
    hashed = _HASH.findall(raw)
    if hashed:
        value = _first_number(hashed[-1])
        if value is not None:
            return ParsedAnswer(value, PARSE_SOURCE_HASH, raw)
    value = _last_number(raw)
    if value is not None:
        return ParsedAnswer(value, PARSE_SOURCE_TRAILING, raw)
    return ParsedAnswer(None, PARSE_SOURCE_NONE, raw)


def split_from_world_ref(world_ref: str | None) -> str:
    raw = (world_ref or "").strip()
    if "@" in raw:
        suffix = raw.rsplit("@", 1)[-1].strip().lower()
        if suffix in {TRAIN_SPLIT, "train_split"}:
            return TRAIN_SPLIT
        if suffix in {HELDOUT_SPLIT, "test", "eval"}:
            return HELDOUT_SPLIT
    return HELDOUT_SPLIT


def public_observation(row: Gsm8kRow, *, seed: int, split: str) -> dict[str, Any]:
    """Question only. The reference answer is env-private and must not appear here."""
    return {
        "question": row.question,
        "seed": seed,
        "split": split,
        "system": SOLVE_SYSTEM,
        "prompt": user_prompt(row.question),
    }


def source_name() -> str:
    return os.environ.get("SYNTH_GSM8K_SOURCE", "fixture").strip().lower() or "fixture"


def load_row(split: str, seed: int) -> Gsm8kRow | None:
    rows = _rows_for(split)
    if seed < 0 or seed >= len(rows):
        return None
    return rows[seed]


def split_size(split: str) -> int:
    return len(_rows_for(split))


def _rows_for(split: str) -> tuple[Gsm8kRow, ...]:
    name = source_name()
    if name in {"hf", "huggingface", "openai"}:
        return _hf_rows(split)
    if split == TRAIN_SPLIT:
        return _TRAIN_FIXTURE
    return _HELDOUT_FIXTURE


@lru_cache(maxsize=4)
def _hf_rows(split: str) -> tuple[Gsm8kRow, ...]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("SYNTH_GSM8K_SOURCE=hf requires the datasets package") from exc
    hf_split = "train" if split == TRAIN_SPLIT else "test"
    dataset = load_dataset("openai/gsm8k", "main", split=hf_split)
    return tuple(
        Gsm8kRow(question=str(item["question"]), answer_text=str(item["answer"]))
        for item in dataset
    )
