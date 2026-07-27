from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from repro.contracts import ContractError


PROVIDED_LABELS = {
    "ANG": "A",
    "DIS": "D",
    "FEA": "F",
    "HAP": "H",
    "NEU": "N",
    "SAD": "S",
}
SUPPORTED_COUNTS = ("A", "H", "N", "S")
NON_NEUTRAL_COUNTS = ("A", "H", "S")


@dataclass(frozen=True)
class VoteRow:
    row_id: int
    file_name: str
    provided_label: str
    counts: dict[str, int]
    num_responses: int
    majority_labels: tuple[str, ...]


def load_audio_vote_rows(path: str | Path) -> list[VoteRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        id_field = reader.fieldnames[0] if reader.fieldnames else None
        if id_field is None:
            raise ContractError("CREMA-D vote table has no header")
        for raw in reader:
            row_id = int(raw[id_field])
            if not 100_000 <= row_id < 200_000:
                continue
            parts = raw["fileName"].split("_")
            if len(parts) != 4 or parts[2] not in PROVIDED_LABELS:
                raise ContractError(f"invalid CREMA-D filename: {raw['fileName']}")
            counts = {label: int(raw[label]) for label in ("A", "D", "F", "H", "N", "S")}
            num_responses = int(raw["numResponses"])
            if sum(counts.values()) != num_responses:
                raise ContractError(f"vote-count mismatch for {raw['fileName']}")
            rows.append(
                VoteRow(
                    row_id=row_id,
                    file_name=raw["fileName"],
                    provided_label=PROVIDED_LABELS[parts[2]],
                    counts=counts,
                    num_responses=num_responses,
                    majority_labels=tuple(str(raw["emoVote"]).split(":")),
                )
            )
    if not rows:
        raise ContractError("no audio-only CREMA-D vote rows found")
    return rows


def is_supported_disagreement(row: VoteRow) -> bool:
    return (
        row.counts["D"] == 0
        and row.counts["F"] == 0
        and row.provided_label not in row.majority_labels
    )


def selection_variants(rows: list[VoteRow]) -> dict[str, list[VoteRow]]:
    closest = [row for row in rows if is_supported_disagreement(row)]
    literal_unique = [
        row
        for row in closest
        if sum(row.counts[label] > 0 for label in NON_NEUTRAL_COUNTS) > 2
    ]
    three_non_neutral_votes = [
        row
        for row in closest
        if sum(row.counts[label] for label in NON_NEUTRAL_COUNTS) >= 3
    ]
    return {
        "audio_only": rows,
        "closest_supported_disagreement": closest,
        "literal_gt2_unique_non_neutral": literal_unique,
        "at_least_3_non_neutral_votes": three_non_neutral_votes,
    }


def target_distribution(row: VoteRow) -> dict[str, float]:
    if not is_supported_disagreement(row):
        raise ContractError("target distribution requires a supported disagreement row")
    total = row.num_responses
    distribution = {
        "p_angry": row.counts["A"] / total,
        "p_happy": row.counts["H"] / total,
        "p_sad": row.counts["S"] / total,
        "p_surprise": 0.0,
        "p_neutral": row.counts["N"] / total,
    }
    if abs(sum(distribution.values()) - 1.0) > 1e-12:
        raise ContractError(f"target distribution does not sum to one: {row.file_name}")
    return distribution
