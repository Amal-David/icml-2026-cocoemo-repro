"""Immutable public RAVDESS manifest and pairing contracts for Claims 2 and 3.

This is intentionally a public-data proxy.  It implements the paper's paired
mean-difference equation and layer-site analysis without claiming access to the
authors' larger private/restricted extraction corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from repro.contracts import ContractError
from repro.ravdess_manifest import EMOTIONS, RavdessClip, parse_ravdess_clip


CLAIM23_EMOTIONS = ("neutral", "angry", "happy", "sad", "surprise")
TARGET_EMOTIONS = CLAIM23_EMOTIONS[1:]
SPLITS = {
    "train": tuple(f"{actor:02d}" for actor in range(1, 13)),
    "validation": tuple(f"{actor:02d}" for actor in range(13, 18)),
    "test": tuple(f"{actor:02d}" for actor in range(18, 25)),
}
EXPECTED_CLIPS_BY_SPLIT = {"train": 240, "validation": 100, "test": 140}
EXPECTED_PAIRS_BY_SPLIT = {"train": 192, "validation": 80, "test": 112}


@dataclass(frozen=True)
class Claim23Clip:
    clip_id: str
    group_id: str
    path: str
    actor: str
    statement: str
    repetition: str
    emotion: str
    transcript: str
    split: str


@dataclass(frozen=True)
class Claim23Pair:
    pair_id: str
    split: str
    actor: str
    statement: str
    repetition: str
    transcript: str
    emotion: str
    emotional_clip_id: str
    neutral_clip_id: str


def _split_for_actor(actor: str) -> str:
    for split, actors in SPLITS.items():
        if actor in actors:
            return split
    raise ContractError(f"Claim 2/3 RAVDESS actor is outside the fixed split: {actor}")


def _group_id(clip: RavdessClip) -> str:
    return f"actor_{clip.actor}_statement_{clip.statement}_repetition_{clip.repetition}"


def build_claim23_manifest(paths: Iterable[str | Path]) -> list[Claim23Clip]:
    """Build all 480 intensity-01 speech clips and verify the public scope."""

    by_key: dict[tuple[str, str, str, str], Claim23Clip] = {}
    for path in paths:
        parsed = parse_ravdess_clip(path)
        if parsed is None:
            continue
        if parsed.emotion not in CLAIM23_EMOTIONS:
            continue
        group_id = _group_id(parsed)
        record = Claim23Clip(
            clip_id=f"{group_id}_emotion_{parsed.emotion}",
            group_id=group_id,
            path=parsed.path,
            actor=parsed.actor,
            statement=parsed.statement,
            repetition=parsed.repetition,
            emotion=parsed.emotion,
            transcript=parsed.transcript,
            split=_split_for_actor(parsed.actor),
        )
        key = (record.actor, record.statement, record.repetition, record.emotion)
        if key in by_key:
            raise ContractError(f"duplicate Claim 2/3 RAVDESS clip: {key}")
        by_key[key] = record

    expected_groups = {
        (f"{actor:02d}", statement, repetition)
        for actor in range(1, 25)
        for statement in ("01", "02")
        for repetition in ("01", "02")
    }
    observed_groups = {(actor, statement, repetition) for actor, statement, repetition, _ in by_key}
    if observed_groups != expected_groups:
        raise ContractError(
            "Claim 2/3 RAVDESS scope must contain exactly 96 actor-statement-repetition groups; "
            f"missing={len(expected_groups - observed_groups)}, unexpected={len(observed_groups - expected_groups)}"
        )
    for group in sorted(expected_groups):
        actual = {emotion for actor, statement, repetition, emotion in by_key if (actor, statement, repetition) == group}
        if actual != set(CLAIM23_EMOTIONS):
            raise ContractError(f"incomplete Claim 2/3 RAVDESS group {group}: {sorted(actual)}")
    manifest = sorted(by_key.values(), key=lambda row: (row.actor, row.statement, row.repetition, row.emotion))
    if len(manifest) != 480:
        raise ContractError(f"Claim 2/3 full public scope requires 480 clips, found {len(manifest)}")
    counts = {split: sum(row.split == split for row in manifest) for split in SPLITS}
    if counts != EXPECTED_CLIPS_BY_SPLIT:
        raise ContractError(f"Claim 2/3 split clip accounting changed: {counts}")
    return manifest


def build_claim23_pair_ledger(manifest: Iterable[Claim23Clip]) -> list[Claim23Pair]:
    rows = list(manifest)
    by_key = {(row.actor, row.statement, row.repetition, row.emotion): row for row in rows}
    if len(by_key) != len(rows):
        raise ContractError("Claim 2/3 manifest has duplicate actor-statement-repetition-emotion rows")
    pairs: list[Claim23Pair] = []
    for row in sorted(rows, key=lambda item: item.clip_id):
        if row.emotion == "neutral":
            continue
        neutral = by_key.get((row.actor, row.statement, row.repetition, "neutral"))
        if neutral is None:
            raise ContractError(f"Claim 2/3 pair has no neutral counterpart: {row.clip_id}")
        if row.split != neutral.split or row.transcript != neutral.transcript:
            raise ContractError(f"Claim 2/3 pair contract failed for {row.clip_id}")
        pairs.append(
            Claim23Pair(
                pair_id=f"{row.group_id}_emotion_{row.emotion}",
                split=row.split,
                actor=row.actor,
                statement=row.statement,
                repetition=row.repetition,
                transcript=row.transcript,
                emotion=row.emotion,
                emotional_clip_id=row.clip_id,
                neutral_clip_id=neutral.clip_id,
            )
        )
    if len(pairs) != 384:
        raise ContractError(f"Claim 2/3 full pair ledger requires 384 pairs, found {len(pairs)}")
    counts = {split: sum(pair.split == split for pair in pairs) for split in SPLITS}
    if counts != EXPECTED_PAIRS_BY_SPLIT:
        raise ContractError(f"Claim 2/3 split pair accounting changed: {counts}")
    return sorted(pairs, key=lambda pair: pair.pair_id)


def select_claim23_canary(manifest: Iterable[Claim23Clip]) -> list[Claim23Clip]:
    """Return the fixed 12-clip actor-01 structural canary.

    The canary deliberately tests extraction mechanics rather than paired
    estimates; the complete pair ledger is mandatory only for the 480-clip run.
    """

    rows = [row for row in manifest if row.actor == "01"]
    if len(rows) != 20:
        raise ContractError(f"Claim 2/3 actor-01 canary requires 20 available clips, found {len(rows)}")
    selected = sorted(rows, key=lambda row: row.clip_id)[:12]
    if len(selected) != 12:
        raise ContractError("Claim 2/3 canary must contain exactly 12 clips")
    return selected


def manifest_records(manifest: Iterable[Claim23Clip]) -> list[dict[str, str]]:
    return [asdict(row) for row in manifest]


def pair_records(pairs: Iterable[Claim23Pair]) -> list[dict[str, str]]:
    return [asdict(pair) for pair in pairs]
