from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


CONDITIONS = ("alpha0", "instruction", "cocoemo")


@dataclass(frozen=True)
class Assignment:
    rater_id: str
    group: int
    item_id: str
    condition: str


def condition_for(*, group: int, item_index: int) -> str:
    if group not in {0, 1, 2}:
        raise ValueError("group must be 0, 1, or 2")
    if not 0 <= item_index < 24:
        raise ValueError("item_index must be between 0 and 23")
    return CONDITIONS[(group + (item_index % 3)) % 3]


def build_retained_assignments() -> list[Assignment]:
    assignments = []
    for group in range(3):
        for rater in range(10):
            rater_id = f"g{group}-r{rater:02d}"
            for item_index in range(24):
                assignments.append(
                    Assignment(
                        rater_id=rater_id,
                        group=group,
                        item_id=f"item-{item_index:02d}",
                        condition=condition_for(group=group, item_index=item_index),
                    )
                )
    return assignments


def validate_balance(assignments: list[Assignment]) -> None:
    by_rater = Counter((row.rater_id, row.condition) for row in assignments)
    by_stimulus = Counter((row.item_id, row.condition) for row in assignments)
    content_seen = Counter((row.rater_id, row.item_id) for row in assignments)

    if len({row.rater_id for row in assignments}) != 30:
        raise ValueError("design must contain exactly 30 retained raters")
    if any(count != 8 for count in by_rater.values()) or len(by_rater) != 90:
        raise ValueError("every rater must receive exactly 8 trials per condition")
    if any(count != 10 for count in by_stimulus.values()) or len(by_stimulus) != 72:
        raise ValueError("every stimulus must receive exactly 10 ratings")
    if any(count != 1 for count in content_seen.values()):
        raise ValueError("a rater must not hear multiple conditions for one content family")
