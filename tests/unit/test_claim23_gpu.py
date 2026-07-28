from pathlib import Path

import pytest

from repro.claim23_gpu import parse_claim23_settings
from repro.claim23_gpu import _confusion_matrix
from repro.config import ReproConfig, load_config
from repro.contracts import ContractError
from repro.run_all import run_stage


REPO = Path(__file__).resolve().parents[2]


def test_hash_locked_full_and_canary_configs_parse_with_exact_denominators() -> None:
    full = parse_claim23_settings(load_config(REPO / "configs" / "claim23-ravdess-paired-sites.yaml"))
    canary = parse_claim23_settings(load_config(REPO / "configs" / "claim23-ravdess-canary.yaml"))

    assert (full.mode, full.expected_clips, full.expected_activation_records, full.evidence) == ("full", 480, 115200, True)
    assert (canary.mode, canary.expected_clips, canary.expected_activation_records, canary.evidence) == ("canary", 12, 2880, False)


def test_canary_rejects_a_tampered_hash_lock() -> None:
    config = load_config(REPO / "configs" / "claim23-ravdess-canary.yaml")
    raw = {**config.raw, "claim23_canary": {**config.raw["claim23_canary"], "base_config_sha256": "0" * 64}}
    with pytest.raises(ContractError, match="hash mismatch"):
        parse_claim23_settings(ReproConfig(source=config.source, raw=raw, digest=config.digest))


def test_confusion_matrix_preserves_all_test_predictions() -> None:
    matrix = _confusion_matrix([
        {"truth": "angry", "prediction": "angry"}, {"truth": "angry", "prediction": "sad"},
        {"truth": "neutral", "prediction": "neutral"},
    ])
    assert matrix["angry"] == {"angry": 1, "happy": 0, "neutral": 0, "sad": 1, "surprise": 0}
    assert sum(sum(row.values()) for row in matrix.values()) == 3


def test_fixed_runner_routes_claim23_without_touching_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = load_config(REPO / "configs" / "claim23-ravdess-canary.yaml")
    expected = {"stage": "claim2", "claim_evidence": False, "verdict": "routed"}

    monkeypatch.setattr(
        "repro.claim23_gpu.run_claim23_gpu",
        lambda received, *, repo, run_root: expected
        if received == config and repo == REPO and run_root == tmp_path
        else (_ for _ in ()).throw(AssertionError("Claim 2/3 route changed its arguments")),
    )

    assert run_stage(config, run_root=tmp_path) == expected
