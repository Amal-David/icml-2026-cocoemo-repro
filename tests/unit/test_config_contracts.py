from pathlib import Path

import pytest

from repro.config import ConfigError, load_config
from repro.contracts import ContractError, require_complete_counts, require_metrics


def test_config_digest_is_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("run: {name: smoke, seed: 7, stage: preflight}\nclaims: [1]\n")
    second.write_text("claims: [1]\nrun:\n  stage: preflight\n  seed: 7\n  name: smoke\n")

    assert load_config(first).digest == load_config(second).digest


def test_config_rejects_unknown_claim(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("run: {name: smoke, seed: 7, stage: preflight}\nclaims: [6]\n")

    with pytest.raises(ConfigError, match="1 through 5"):
        load_config(config)


def test_incomplete_counts_fail_loudly() -> None:
    with pytest.raises(ContractError, match="requested=3, generated=2, evaluated=2"):
        require_complete_counts(requested=3, generated=2, evaluated=2)


def test_missing_metrics_fail_loudly() -> None:
    records = [{"id": "a", "tep": 0.2}, {"id": "b", "tep": None}]

    with pytest.raises(ContractError, match="record 1.*tep"):
        require_metrics(records, ["tep"])
