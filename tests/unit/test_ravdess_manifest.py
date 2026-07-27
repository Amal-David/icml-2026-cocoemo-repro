import pytest

from repro.contracts import ContractError
from repro.ravdess_manifest import build_claim1_groups, parse_ravdess_clip, require_full_public_scope


def group_files(actor: int, statement: int = 1, repetition: int = 1) -> list[str]:
    return [
        f"03-01-{emotion}-01-{statement:02d}-{repetition:02d}-{actor:02d}.wav"
        for emotion in ("01", "03", "04", "05", "08")
    ]


def test_parser_uses_only_normal_intensity_speech_audio() -> None:
    clip = parse_ravdess_clip("03-01-05-01-02-01-07.wav")
    assert clip is not None
    assert clip.emotion == "angry"
    assert clip.transcript == "Dogs are sitting by the door."
    assert parse_ravdess_clip("03-01-05-02-02-01-07.wav") is None
    assert parse_ravdess_clip("03-02-05-01-02-01-07.wav") is None


def test_group_builder_requires_all_five_emotions() -> None:
    with pytest.raises(ContractError, match="missing surprise"):
        build_claim1_groups(group_files(1)[:-1])


def test_full_public_scope_is_96_independent_groups() -> None:
    paths = [
        path
        for actor in range(1, 25)
        for statement in (1, 2)
        for repetition in (1, 2)
        for path in group_files(actor, statement, repetition)
    ]
    groups = build_claim1_groups(paths)

    require_full_public_scope(groups)
    assert len(groups) == 96
    assert len({group["group_id"] for group in groups}) == 96


def test_scope_check_rejects_upsampled_or_partial_counts() -> None:
    with pytest.raises(ContractError, match="96 independent"):
        require_full_public_scope([{"group_id": "one"}] * 300)
