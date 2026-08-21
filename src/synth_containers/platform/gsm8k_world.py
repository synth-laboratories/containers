"""GSM8K one-turn world. Gold stays private; public observation is the question.

Fixture rows are the PR CI source. The real dataset is ``openai/gsm8k`` pinned
to one HuggingFace revision (:data:`HF_REVISION`) with a recorded digest per
split (:data:`SPLIT_PINS`); a load whose rows do not reproduce those digests is
refused rather than scored. Which rows back the world is a *declared profile*
(:func:`declare_profile`): ``fixture`` (32 in-module rows), ``hf`` (the pinned
revision through ``datasets``) or ``snapshot`` (the pinned rows baked into a
directory, as the eval target image does). ``SYNTH_GSM8K_SOURCE=hf`` is kept
only as a test-time opt-in; the pin itself never comes from the environment,
and :func:`dataset_manifest` reports which profile produced a number.

Three things this module refuses to blur:

- a pinned profile and an unpinned one: the manifest says ``pinned`` only for
  rows whose digest matched the recorded one;

- the reference answer never reaches :func:`public_observation`;
- a completion that cannot be parsed is *unparsed*, not wrong. :func:`parse_answer`
  returns the reason and the raw text alongside the value, and the runtime keeps
  the reward ``None`` for that case rather than scoring a zero nobody measured.

See: workshop/docs/aug_12_update.md §2.2 (Environment / Policy / TaskWorld).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any


SOLVE_SYSTEM = (
    "Solve the grade-school math word problem. Reason step by step, then state the "
    "final numeric answer on its own last line in the form '#### <answer>'. The final "
    "answer must be a bare number with no units, no currency symbol and no words."
)

WORLD_REF_PREFIX = "world:gsm8k"
TRAIN_SPLIT = "train"
HELDOUT_SPLIT = "heldout"

# --- the dataset pin -----------------------------------------------------------
#
# One HuggingFace revision, one digest per split. These live in code, not in an
# environment variable, so the pin is part of the container's capability digest
# and a score can always say which bytes it was measured on. Change them only
# with a new receipt: the digests are over the canonical JSON lines of
# ``{"answer", "question"}`` in the dataset's own row order (see
# :func:`rows_digest`), so they are independent of the shuffle below.
HF_DATASET = "openai/gsm8k"
HF_CONFIG = "main"
HF_REVISION = "740312add88f781978c0658806c59bc2815b9866"
#: Seeds index a deterministic permutation of each pinned split rather than its
#: head, so ``seed:0..9`` samples the dataset instead of its first ten rows.
#: Recorded in the manifest; the same seed always names the same row.
SHUFFLE_SEED = 20260820
MANIFEST_SCHEMA = "gsm8k.dataset-manifest.v1"


@dataclass(frozen=True)
class SplitPin:
    split: str
    hf_split: str
    rows: int
    digest: str

    def to_json(self) -> dict[str, Any]:
        return {"hf_split": self.hf_split, "rows": self.rows, "digest": self.digest}


SPLIT_PINS: dict[str, SplitPin] = {
    TRAIN_SPLIT: SplitPin(
        TRAIN_SPLIT,
        "train",
        7473,
        "sha256:dca449882e67a1e7716b5da453650a7b16588a1f2d1a0d82394a43d75ea7e9ca",
    ),
    HELDOUT_SPLIT: SplitPin(
        HELDOUT_SPLIT,
        "test",
        1319,
        "sha256:32c548f08195e19e33408b844dd7be6aa4bcae457d957bc05909ff4bd4a00595",
    ),
}

PROFILE_FIXTURE = "fixture"
PROFILE_HF = "hf"
PROFILE_SNAPSHOT = "snapshot"
PROFILES: tuple[str, ...] = (PROFILE_FIXTURE, PROFILE_HF, PROFILE_SNAPSHOT)
#: Test-time opt-in only. It can choose ``hf`` over the fixture; it cannot name
#: a revision, a split, or a digest, and a declared profile always wins.
LEGACY_SOURCE_ENV = "SYNTH_GSM8K_SOURCE"
_LEGACY_HF_VALUES = frozenset({"hf", "huggingface", "openai"})

SPLIT_DIGEST_MISMATCH = "gsm8k_split_digest_mismatch"
SNAPSHOT_INVALID = "gsm8k_snapshot_invalid"

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

    @property
    def parse_mode(self) -> str:
        """``exact`` / ``trailing_number`` / ``unparsed`` — reported per trial.

        ``exact`` means the completion used a marked answer (``#### N`` or
        ``\\boxed{}``), which is what the prompt asks for. ``trailing_number`` is
        the fallback: the last bare number in the text. The fallback keeps a
        rambling-but-correct answer scorable, but it also means a 0% parse
        failure rate says nothing about format compliance — only the share of
        ``exact`` trials does, and that is why the mode travels with every
        action rather than being folded into ``parsed``.
        """
        if self.source in _EXACT_SOURCES:
            return PARSE_MODE_EXACT
        if self.source == PARSE_SOURCE_TRAILING:
            return PARSE_MODE_TRAILING
        return PARSE_MODE_UNPARSED

    @property
    def format_compliant(self) -> bool:
        return self.parse_mode == PARSE_MODE_EXACT


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

PARSE_MODE_EXACT = "exact"
PARSE_MODE_TRAILING = "trailing_number"
PARSE_MODE_UNPARSED = "unparsed"
PARSE_MODES: tuple[str, ...] = (PARSE_MODE_EXACT, PARSE_MODE_TRAILING, PARSE_MODE_UNPARSED)
_EXACT_SOURCES = frozenset({PARSE_SOURCE_BOXED, PARSE_SOURCE_HASH})


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


# --- profiles ------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetProfile:
    """Which rows back the world, and who said so.

    ``source`` is ``declared`` (set in code via :func:`declare_profile`),
    ``env`` (the legacy test-time opt-in) or ``default``. Anything reportable
    reads the profile from here, never from the environment directly.
    """

    name: str
    source: str
    snapshot_dir: Path | None = None

    @property
    def pinned(self) -> bool:
        return self.name in {PROFILE_HF, PROFILE_SNAPSHOT}


_declared_profile: DatasetProfile | None = None


def declare_profile(name: str, *, snapshot_dir: str | Path | None = None) -> DatasetProfile:
    """Declare, in code, which profile backs the world. Wins over the env var."""
    global _declared_profile
    if name not in PROFILES:
        raise ValueError(f"unknown gsm8k profile {name!r}; expected one of {PROFILES}")
    directory: Path | None = None
    if name == PROFILE_SNAPSHOT:
        if snapshot_dir is None:
            raise ValueError("the snapshot profile needs snapshot_dir")
        directory = Path(snapshot_dir).expanduser().resolve()
        for pin in SPLIT_PINS.values():
            if not (directory / f"{pin.hf_split}.jsonl").is_file():
                raise RuntimeError(f"{SNAPSHOT_INVALID}:{pin.hf_split}.jsonl missing in {directory}")
    elif snapshot_dir is not None:
        raise ValueError("snapshot_dir is only meaningful for the snapshot profile")
    _declared_profile = DatasetProfile(name, "declared", directory)
    _clear_row_caches()
    return _declared_profile


def clear_profile() -> None:
    """Forget a declaration (tests). The env opt-in / fixture default apply again."""
    global _declared_profile
    _declared_profile = None
    _clear_row_caches()


def active_profile() -> DatasetProfile:
    if _declared_profile is not None:
        return _declared_profile
    raw = os.environ.get(LEGACY_SOURCE_ENV, "").strip().lower()
    if raw in _LEGACY_HF_VALUES:
        return DatasetProfile(PROFILE_HF, "env")
    if raw == PROFILE_FIXTURE:
        return DatasetProfile(PROFILE_FIXTURE, "env")
    return DatasetProfile(PROFILE_FIXTURE, "default")


def source_name() -> str:
    """Name of the active profile (compatibility alias for :func:`active_profile`)."""
    return active_profile().name


def _clear_row_caches() -> None:
    _hf_rows.cache_clear()
    _snapshot_rows.cache_clear()


# --- digests and order ---------------------------------------------------------


def _canonical_line(row: Gsm8kRow) -> bytes:
    return json.dumps(
        {"question": row.question, "answer": row.answer_text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def rows_digest(rows: Iterable[Gsm8kRow]) -> tuple[int, str]:
    """``(row_count, sha256:…)`` over canonical JSON lines in the given order."""
    hasher = hashlib.sha256()
    count = 0
    for row in rows:
        hasher.update(_canonical_line(row))
        hasher.update(b"\n")
        count += 1
    return count, "sha256:" + hasher.hexdigest()


def verify_split(split: str, rows: tuple[Gsm8kRow, ...]) -> SplitPin:
    """Refuse rows that are not the pinned split. Returns the matching pin."""
    pin = SPLIT_PINS[split]
    count, digest = rows_digest(rows)
    if count != pin.rows or digest != pin.digest:
        raise RuntimeError(
            f"{SPLIT_DIGEST_MISMATCH}:{split}:expected {pin.rows} rows {pin.digest}, "
            f"got {count} rows {digest}"
        )
    return pin


def shuffled_order(count: int, *, seed: int = SHUFFLE_SEED) -> tuple[int, ...]:
    """The deterministic permutation seeds index into. Same seed, same order."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return tuple(order)


def _pinned_rows(split: str, source_rows: tuple[Gsm8kRow, ...]) -> tuple[Gsm8kRow, ...]:
    verify_split(split, source_rows)
    return tuple(source_rows[index] for index in shuffled_order(len(source_rows)))


def dataset_manifest() -> dict[str, Any]:
    """What backs the world right now: pin, profile, order, parse modes.

    Exposed on the target's metadata so a rollout record can be read back
    against the exact rows and the exact parser behaviour it was scored with.
    """
    profile = active_profile()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "revision": HF_REVISION,
        "profile": profile.name,
        "profile_source": profile.source,
        "pinned": profile.pinned,
        "splits": {split: pin.to_json() for split, pin in SPLIT_PINS.items()},
        "shuffle_seed": SHUFFLE_SEED if profile.pinned else None,
        "order": (
            "random.Random(shuffle_seed).shuffle(list(range(rows)))"
            if profile.pinned
            else "persisted_indices"
        ),
        "snapshot_dir": str(profile.snapshot_dir) if profile.snapshot_dir else None,
        "parse_modes": list(PARSE_MODES),
    }
    if profile.name == PROFILE_FIXTURE:
        manifest["fixture"] = {
            TRAIN_SPLIT: _fixture_split_json(_TRAIN_FIXTURE, TRAIN_INDICES),
            HELDOUT_SPLIT: _fixture_split_json(_HELDOUT_FIXTURE, HELDOUT_INDICES),
        }
    return manifest


def _fixture_split_json(rows: tuple[Gsm8kRow, ...], indices: tuple[int, ...]) -> dict[str, Any]:
    count, digest = rows_digest(rows)
    return {"rows": count, "digest": digest, "indices": list(indices)}


# --- rows ----------------------------------------------------------------------


def load_row(split: str, seed: int) -> Gsm8kRow | None:
    rows = _rows_for(split)
    if seed < 0 or seed >= len(rows):
        return None
    return rows[seed]


def split_size(split: str) -> int:
    return len(_rows_for(split))


def _rows_for(split: str) -> tuple[Gsm8kRow, ...]:
    profile = active_profile()
    if profile.name == PROFILE_HF:
        return _hf_rows(split)
    if profile.name == PROFILE_SNAPSHOT:
        return _snapshot_rows(split, str(profile.snapshot_dir))
    if split == TRAIN_SPLIT:
        return _TRAIN_FIXTURE
    return _HELDOUT_FIXTURE


@lru_cache(maxsize=4)
def _hf_rows(split: str) -> tuple[Gsm8kRow, ...]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("the hf profile requires the datasets package") from exc
    pin = SPLIT_PINS[split]
    dataset = load_dataset(HF_DATASET, HF_CONFIG, split=pin.hf_split, revision=HF_REVISION)
    source_rows = tuple(
        Gsm8kRow(question=str(item["question"]), answer_text=str(item["answer"]))
        for item in dataset
    )
    return _pinned_rows(split, source_rows)


@lru_cache(maxsize=4)
def _snapshot_rows(split: str, snapshot_dir: str) -> tuple[Gsm8kRow, ...]:
    pin = SPLIT_PINS[split]
    path = Path(snapshot_dir) / f"{pin.hf_split}.jsonl"
    source_rows: list[Gsm8kRow] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                source_rows.append(
                    Gsm8kRow(question=str(item["question"]), answer_text=str(item["answer"]))
                )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"{SNAPSHOT_INVALID}:{path}") from exc
    return _pinned_rows(split, tuple(source_rows))


def write_snapshot(directory: str | Path, rows_by_split: dict[str, Iterable[Gsm8kRow]]) -> Path:
    """Bake pinned rows into ``<dir>/<hf_split>.jsonl`` + ``manifest.json``.

    The rows are verified against the pin *before* anything is written, so a
    snapshot directory either reproduces the recorded digests or does not exist.
    """
    target = Path(directory).expanduser().resolve()
    verified: dict[str, tuple[Gsm8kRow, ...]] = {}
    for split, rows in rows_by_split.items():
        materialized = tuple(rows)
        verify_split(split, materialized)
        verified[split] = materialized
    target.mkdir(parents=True, exist_ok=True)
    for split, materialized in verified.items():
        with (target / f"{SPLIT_PINS[split].hf_split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in materialized:
                handle.write(_canonical_line(row).decode("utf-8") + "\n")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "revision": HF_REVISION,
        "splits": {split: SPLIT_PINS[split].to_json() for split in verified},
        "shuffle_seed": SHUFFLE_SEED,
        "order": "random.Random(shuffle_seed).shuffle(list(range(rows)))",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target
