import csv
from pathlib import Path

import pytest

from repro.contracts import ContractError
from repro.cremad_selection import load_audio_vote_rows, selection_variants, target_distribution


FIELDNAMES = ["", "A", "D", "F", "H", "N", "S", "fileName", "numResponses", "emoVote"]


def write_votes(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_variants_do_not_conflate_unique_labels_with_vote_count(tmp_path: Path) -> None:
    votes = tmp_path / "votes.csv"
    write_votes(
        votes,
        [
            {"": 100001, "A": 2, "D": 0, "F": 0, "H": 2, "N": 1, "S": 2, "fileName": "1001_IEO_NEU_XX", "numResponses": 7, "emoVote": "A:H:S"},
            {"": 100002, "A": 3, "D": 0, "F": 0, "H": 0, "N": 2, "S": 0, "fileName": "1001_IEO_NEU_XX", "numResponses": 5, "emoVote": "A"},
            {"": 200001, "A": 3, "D": 0, "F": 0, "H": 0, "N": 2, "S": 0, "fileName": "1001_IEO_NEU_XX", "numResponses": 5, "emoVote": "A"},
        ],
    )

    variants = selection_variants(load_audio_vote_rows(votes))
    assert len(variants["audio_only"]) == 2
    assert len(variants["closest_supported_disagreement"]) == 2
    assert len(variants["literal_gt2_unique_non_neutral"]) == 1
    assert len(variants["at_least_3_non_neutral_votes"]) == 2


def test_target_distribution_includes_neutral_mass(tmp_path: Path) -> None:
    votes = tmp_path / "votes.csv"
    write_votes(
        votes,
        [{"": 100001, "A": 2, "D": 0, "F": 0, "H": 1, "N": 2, "S": 0, "fileName": "1001_IEO_HAP_XX", "numResponses": 5, "emoVote": "A:N"}],
    )
    row = load_audio_vote_rows(votes)[0]

    assert target_distribution(row) == {
        "p_angry": 0.4,
        "p_happy": 0.2,
        "p_sad": 0.0,
        "p_surprise": 0.0,
        "p_neutral": 0.4,
    }


def test_vote_count_mismatch_fails_loudly(tmp_path: Path) -> None:
    votes = tmp_path / "votes.csv"
    write_votes(
        votes,
        [{"": 100001, "A": 2, "D": 0, "F": 0, "H": 0, "N": 2, "S": 0, "fileName": "1001_IEO_NEU_XX", "numResponses": 5, "emoVote": "A:N"}],
    )

    with pytest.raises(ContractError, match="vote-count mismatch"):
        load_audio_vote_rows(votes)
