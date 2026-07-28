from pathlib import Path

import pytest
import torch

from cocoemo.steering import create_sample_specific_mixed_vectors
from repro.claim4_mixing_audit import (
    PROBE_DISTRIBUTION,
    _released_synthesis_contract,
    audit_released_cosyvoice_neutral_mixing,
    compare_neutral_semantics,
)
from repro.contracts import ContractError


REPO = Path(__file__).resolve().parents[2]


def _distribution(*, neutral: float) -> dict[str, float]:
    return {
        "p_angry": 0.20,
        "p_happy": 0.40,
        "p_sad": 0.00,
        "p_surprise": 0.40 - neutral,
        "p_neutral": neutral,
    }


def test_released_semantics_equal_explicit_zero_neutral_and_attenuate_renormalized() -> None:
    vectors = {
        "angry": torch.tensor([2.0, 4.0]),
        "happy": torch.tensor([4.0, 8.0]),
        "sad": torch.tensor([1.0, 2.0]),
        "surprise": torch.tensor([3.0, 6.0]),
    }
    report = compare_neutral_semantics(vectors, _distribution(neutral=0.40))

    assert report["missing_positive_weight_emotions"] == ["neutral"]
    assert report["released_semantics"] == "unrenormalized_non_neutral"
    assert report["released_vs_explicit_zero_max_abs_diff"] == 0.0
    assert report["retained_non_neutral_mass"] == pytest.approx(0.60)
    assert report["released_to_renormalized_norm_ratio"] == pytest.approx(0.60)
    assert report["released_vs_renormalized_l2"] > 0


def test_all_neutral_distribution_has_no_renormalized_non_neutral_interpretation() -> None:
    vectors = {"angry": torch.tensor([2.0]), "happy": torch.tensor([4.0])}
    report = compare_neutral_semantics(
        vectors,
        {"p_angry": 0.0, "p_happy": 0.0, "p_sad": 0.0, "p_surprise": 0.0, "p_neutral": 1.0},
    )

    assert report["retained_non_neutral_mass"] == 0.0
    assert report["renormalized_non_neutral_semantics"] == "undefined_all_neutral"
    assert report["released_vs_renormalized_l2"] is None


def test_distribution_contract_rejects_non_unit_mass() -> None:
    distribution = {
        "p_angry": 0.5,
        "p_happy": 0.0,
        "p_sad": 0.0,
        "p_surprise": 0.0,
        "p_neutral": 0.0,
    }
    with pytest.raises(ContractError, match="sum to 1.0"):
        compare_neutral_semantics({"angry": torch.tensor([1.0])}, distribution)


def test_source_contract_rejects_direct_neutral_weight_reference(tmp_path: Path) -> None:
    script = tmp_path / "synthesize.py"
    script.write_text(
        'EMOTIONS = ["angry", "happy", "sad", "surprise"]\n'
        'weight = row["p_neutral"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="directly references p_neutral: string:2"):
        _released_synthesis_contract(script)


def test_released_helper_ignores_a_present_neutral_weight_when_no_neutral_file_exists(tmp_path: Path) -> None:
    paths = {}
    for emotion, value in {"angry": 2.0, "happy": 4.0}.items():
        path = tmp_path / f"{emotion}.pt"
        torch.save({"steering_vectors": {"attn_output": {0: torch.tensor([value])}}}, path)
        paths[emotion] = str(path)

    output = create_sample_specific_mixed_vectors(
        paths,
        {"p_angry": 0.3, "p_happy": 0.3, "p_neutral": 0.4},
    )

    # p_neutral has no matching file, so the released helper returns 0.3*2 + 0.3*4.
    assert output["attn_output"][0].tolist() == pytest.approx([1.8])


def test_release_audit_records_source_and_exact_neutral_mass_defect() -> None:
    report = audit_released_cosyvoice_neutral_mixing(REPO)

    assert report["source"]["script_emotions"] == ["angry", "happy", "sad", "surprise"]
    assert not report["source"]["p_neutral_read_by_synthesis_script"]
    assert report["source"]["p_neutral_direct_references"] == []
    assert report["probe"]["distribution"] == PROBE_DISTRIBUTION
    assert report["probe"]["retained_non_neutral_mass"] == pytest.approx(0.60)
    assert report["probe"]["released_vs_explicit_zero_max_abs_diff"] == 0.0
    assert report["probe"]["released_to_renormalized_norm_ratio"] == pytest.approx(0.60)
