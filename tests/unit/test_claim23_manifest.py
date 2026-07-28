from pathlib import Path

import pytest

from repro.claim23_manifest import (
    EXPECTED_CLIPS_BY_SPLIT,
    EXPECTED_PAIRS_BY_SPLIT,
    build_claim23_manifest,
    build_claim23_pair_ledger,
    select_claim23_canary,
)
from repro.contracts import ContractError


def _paths() -> list[str]:
    files: list[str] = []
    for actor in range(1, 25):
        for statement in (1, 2):
            for repetition in (1, 2):
                for emotion in (1, 3, 4, 5, 8):
                    files.append(f"Actor_{actor:02d}/03-01-{emotion:02d}-01-{statement:02d}-{repetition:02d}-{actor:02d}.wav")
    return files


def test_full_public_manifest_has_fixed_speaker_split_and_exact_pairs() -> None:
    manifest = build_claim23_manifest(_paths())
    pairs = build_claim23_pair_ledger(manifest)

    assert len(manifest) == 480
    assert {split: sum(row.split == split for row in manifest) for split in EXPECTED_CLIPS_BY_SPLIT} == EXPECTED_CLIPS_BY_SPLIT
    assert len(pairs) == 384
    assert {split: sum(pair.split == split for pair in pairs) for split in EXPECTED_PAIRS_BY_SPLIT} == EXPECTED_PAIRS_BY_SPLIT
    assert all(pair.transcript and pair.emotional_clip_id != pair.neutral_clip_id for pair in pairs)


def test_manifest_rejects_missing_emotion_even_when_all_other_rows_exist() -> None:
    paths = _paths()
    paths.remove("Actor_01/03-01-08-01-01-01-01.wav")

    with pytest.raises(ContractError, match="96 actor-statement-repetition groups|incomplete"):
        build_claim23_manifest(paths)


def test_canary_is_a_stable_twelve_clip_actor_one_subset() -> None:
    selected = select_claim23_canary(build_claim23_manifest(_paths()))

    assert len(selected) == 12
    assert {row.actor for row in selected} == {"01"}
    assert [row.clip_id for row in selected] == sorted(row.clip_id for row in selected)
