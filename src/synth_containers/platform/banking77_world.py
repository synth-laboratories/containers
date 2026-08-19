"""Banking77 dataset world. Gold stays private; public observation is the query.

Fixture rows are the PR CI source. HuggingFace PolyAI/banking77 is opt-in via
``SYNTH_BANKING77_SOURCE=hf`` so tests never download.

See: workshop/docs/aug_12_update.md §2.2 (Environment / Policy / TaskWorld).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


CLASSIFY_SYSTEM = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return exactly one label from the allowed label list, preserving the label's spelling, "
    "underscores, capitalization, and punctuation. Use the full query, not one keyword. "
    "Prefer the label for the user's concrete banking action, status, or problem: separate "
    "physical-card ordering from delivery timing, virtual-card creation from virtual-card "
    "problems, card payments from cash withdrawals, top-ups from incoming transfers, "
    "pending from failed/declined/reverted, passcodes from card PINs, and phone loss from "
    "card compromise. Return only the label."
)

WORLD_REF_PREFIX = "world:banking77"
TRAIN_SPLIT = "train"
HELDOUT_SPLIT = "heldout"


@dataclass(frozen=True)
class Banking77Row:
    text: str
    label: str


_TRAIN_FIXTURE: tuple[Banking77Row, ...] = (
    Banking77Row("I ordered a new card last week. Has it been posted yet?", "card_arrival"),
    Banking77Row("How do I link this physical card to my account?", "card_linking"),
    Banking77Row("What is the USD to EUR rate right now?", "exchange_rate"),
    Banking77Row("Why was I charged a different FX rate on my card payment?", "card_payment_wrong_exchange_rate"),
    Banking77Row("There is an extra fee on my statement I did not authorize.", "extra_charge_on_statement"),
    Banking77Row("My ATM withdrawal is still pending after two days.", "pending_cash_withdrawal"),
    Banking77Row("Do you support topping up in Japanese yen?", "fiat_currency_support"),
    Banking77Row("When should I expect the replacement card to be delivered?", "card_delivery_estimate"),
    Banking77Row("Please turn on automatic top-up when my balance is low.", "automatic_top_up"),
    Banking77Row("My card is declined at every terminal today.", "card_not_working"),
    Banking77Row("Can I exchange currency inside the app?", "exchange_via_app"),
    Banking77Row("I lost my phone and need to freeze the app login.", "lost_or_stolen_phone"),
    Banking77Row("How do I add this card to Apple Pay?", "apple_pay_or_google_pay"),
    Banking77Row("Which merchants accept this card?", "card_acceptance"),
    Banking77Row("Is there a fee to top up by bank transfer?", "top_up_by_bank_transfer_charge"),
    Banking77Row("My top-up is still pending. When will it land?", "pending_top_up"),
)

_HELDOUT_FIXTURE: tuple[Banking77Row, ...] = (
    Banking77Row("I want to cancel a transfer I just sent.", "cancel_transfer"),
    Banking77Row("Why can't I add this beneficiary?", "beneficiary_not_allowed"),
    Banking77Row("The card payment was declined at the grocery store.", "declined_card_payment"),
    Banking77Row("How long do international transfers take?", "transfer_timing"),
    Banking77Row("I need a new PIN for my card.", "change_pin"),
    Banking77Row("The cash I withdrew never came out of the ATM.", "cash_withdrawal_not_recognised"),
    Banking77Row("Please activate the virtual card I just created.", "get_physical_card"),
    Banking77Row("Why did my transfer revert back to my account?", "reverted_card_payment"),
    Banking77Row("I was charged twice for the same top-up.", "top_up_reverted"),
    Banking77Row("Can I get a refund for a card payment?", "request_refund"),
    Banking77Row("My passcode is not working after the update.", "passcode_forgotten"),
    Banking77Row("The incoming transfer has not arrived yet.", "transfer_not_received_by_recipient"),
    Banking77Row("I need to verify my identity to keep using the app.", "verify_my_identity"),
    Banking77Row("How do I order a physical card?", "order_physical_card"),
    Banking77Row("Why is my balance lower after a declined payment?", "declined_cash_withdrawal"),
    Banking77Row("The virtual card details are not showing in the app.", "virtual_card_not_working"),
)


def user_prompt(text: str, *, labels: tuple[str, ...] | None = None) -> str:
    vocabulary = ""
    if labels:
        vocabulary = "Allowed labels:\n" + "\n".join(labels) + "\n\n"
    return (
        f"{vocabulary}Customer query:\n{text}\n\n"
        "Return EXACTLY one Banking77 intent label as written, no other text."
    )


def normalize_label(text: str) -> str:
    first = (text or "").strip().splitlines()[0].strip().strip("`\"'")
    return first.lower().replace("-", "_").replace(" ", "_")


def split_from_world_ref(world_ref: str | None) -> str:
    raw = (world_ref or "").strip()
    if "@" in raw:
        suffix = raw.rsplit("@", 1)[-1].strip().lower()
        if suffix in {TRAIN_SPLIT, "train_split"}:
            return TRAIN_SPLIT
        if suffix in {HELDOUT_SPLIT, "test", "eval"}:
            return HELDOUT_SPLIT
    return HELDOUT_SPLIT


def label_vocabulary() -> tuple[str, ...]:
    """The full action space, sorted. Not gold: it is identical for every item.

    Withholding it does not make the task harder, it makes it unanswerable --
    the policy is asked to emit one exact string out of 77 it has never seen.
    Measured on Qwen3.5-0.8B without it: 0/40, with every prediction a plausible
    intent name (`locate_card`, `delivery_timing`) that is simply not in the
    vocabulary. That is a measurement of the prompt, not of the model.
    """

    return tuple(sorted({row.label for row in _rows_for(TRAIN_SPLIT)} |
                        {row.label for row in _rows_for(HELDOUT_SPLIT)}))


def public_observation(row: Banking77Row, *, seed: int, split: str) -> dict[str, Any]:
    """Query plus the action space. Gold is env-private and must not appear here."""
    labels = label_vocabulary()
    return {
        "text": row.text,
        "seed": seed,
        "split": split,
        "system": CLASSIFY_SYSTEM,
        "labels": list(labels),
        "prompt": user_prompt(row.text, labels=labels),
    }


def source_name() -> str:
    return os.environ.get("SYNTH_BANKING77_SOURCE", "fixture").strip().lower() or "fixture"


def load_row(split: str, seed: int) -> Banking77Row | None:
    rows = _rows_for(split)
    if seed < 0 or seed >= len(rows):
        return None
    return rows[seed]


def split_size(split: str) -> int:
    return len(_rows_for(split))


def _rows_for(split: str) -> tuple[Banking77Row, ...]:
    name = source_name()
    if name in {"hf", "huggingface", "polyai"}:
        return _hf_rows(split)
    if split == TRAIN_SPLIT:
        return _TRAIN_FIXTURE
    return _HELDOUT_FIXTURE


@lru_cache(maxsize=4)
def _hf_rows(split: str) -> tuple[Banking77Row, ...]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "SYNTH_BANKING77_SOURCE=hf requires the datasets package"
        ) from exc
    hf_split = "train" if split == TRAIN_SPLIT else "test"
    dataset = load_dataset("PolyAI/banking77", split=hf_split, trust_remote_code=True)
    names = list(dataset.features["label"].names)
    rows = []
    for item in dataset:
        rows.append(
            Banking77Row(
                text=str(item["text"]),
                label=str(names[int(item["label"])]),
            )
        )
    return _deterministic_order(rows, split)


#: Fixed forever. Changing it renumbers every seed and silently invalidates any
#: comparison against a previously reported number.
_SHUFFLE_SEED = 20260818


def _deterministic_order(rows: list[Banking77Row], split: str) -> tuple[Banking77Row, ...]:
    """Shuffle once, deterministically, before seeds index into the split.

    PolyAI/banking77 ships label-sorted. Taking seeds 0..N in dataset order
    therefore draws N consecutive items of the SAME class -- measured: seeds 0-3
    of the test split are all `card_arrival`. An accuracy over such a slice is
    not a benchmark number, it is one class sampled N times, and it moves wildly
    with N for reasons that have nothing to do with the policy.
    """

    ordered = list(rows)
    random.Random(f"{_SHUFFLE_SEED}:{split}").shuffle(ordered)
    return tuple(ordered)
