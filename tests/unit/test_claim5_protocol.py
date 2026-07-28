from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from repro.claim5_protocol import (
    CLAIM5_ARMS,
    TARGETS,
    WRONG_TARGET,
    build_claim5_arms,
    build_claim5_cells,
    load_claim5_catalog,
    parse_claim5_settings,
)
from repro.config import ReproConfig, load_config
from repro.contracts import ContractError


REPO = Path(__file__).resolve().parents[2]


def _groups() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for actor in range(1, 25):
        for statement in ("01", "02"):
            for repetition in ("01", "02"):
                refs = {
                    emotion: {"path": f"/refs/{actor:02d}-{statement}-{repetition}-{emotion}.wav"}
                    for emotion in ("neutral", *TARGETS)
                }
                result.append({"group_id": f"actor_{actor:02d}_statement_{statement}_repetition_{repetition}", "references": refs})
    return result


def test_claim5_full_and_canary_are_hash_locked_and_accounted() -> None:
    full = parse_claim5_settings(load_config(REPO / "configs/claim5-ravdess-semantic-conflict.yaml"))
    canary = parse_claim5_settings(load_config(REPO / "configs/claim5-ravdess-semantic-conflict-canary.yaml"))
    assert (full.mode, full.expected_cells, full.expected_rendered_wavs) == ("full", 96, 768)
    assert (canary.mode, canary.expected_cells, canary.expected_rendered_wavs) == ("canary", 4, 32)
    catalog = load_claim5_catalog(repo=REPO, settings=full)
    assert {target: len(items) for target, items in catalog.items()} == {target: 4 for target in TARGETS}


def test_catalog_and_protocol_never_claim_a_va_or_model_eligibility_gate() -> None:
    source = (REPO / "repro/claim5_protocol.py").read_text(encoding="utf-8").lower()
    config = (REPO / "configs/claim5-ravdess-semantic-conflict.yaml").read_text(encoding="utf-8").lower()
    assert "vad" not in source + config
    assert "high-mismatch" not in source + config
    assert "robrokools" not in source + config


def test_full_cells_are_exactly_balanced_across_content_families() -> None:
    settings = parse_claim5_settings(load_config(REPO / "configs/claim5-ravdess-semantic-conflict.yaml"))
    cells = build_claim5_cells(_groups(), catalog=load_claim5_catalog(repo=REPO, settings=settings), actor_ids=settings.actor_ids)
    assert len(cells) == 96
    assert {cell.target for cell in cells} == set(TARGETS)
    by_content: dict[str, set[str]] = {}
    for cell in cells:
        by_content.setdefault(cell.content_id, set()).add(cell.actor_id)
        assert cell.neutral_reference != cell.emotional_reference
    assert len(by_content) == 16
    assert {len(actors) for actors in by_content.values()} == {6}


def test_vector_controls_have_both_layers_and_fixed_wrong_derangement() -> None:
    vectors = {target: {17: np.full((1, 896), index + 1, dtype=np.float32), 14: np.full((1, 896), index + 2, dtype=np.float32)} for index, target in enumerate(TARGETS)}
    arms = build_claim5_arms(vectors, target="angry", base_seed=9, cell_id="actor_01__angry__angry_01")
    assert tuple(arms) == CLAIM5_ARMS
    assert arms["instruction_target"]["instruction"] == "Say it in an angry tone."
    assert arms["wrong_target_alpha6"]["vector_target"] == WRONG_TARGET["angry"] == "happy"
    assert tuple(arms["cocoemo_alpha6"]["vectors"]) == (17, 14)
    assert np.allclose(np.linalg.norm(arms["random_norm_matched_alpha6"]["vectors"][17]), np.linalg.norm(arms["cocoemo_alpha6"]["vectors"][17]))
    again = build_claim5_arms(vectors, target="angry", base_seed=9, cell_id="actor_01__angry__angry_01")
    assert np.array_equal(arms["random_norm_matched_alpha6"]["vectors"][14], again["random_norm_matched_alpha6"]["vectors"][14])


def test_config_drift_is_rejected() -> None:
    config = load_config(REPO / "configs/claim5-ravdess-semantic-conflict.yaml")
    raw = {**config.raw, "claim5": {**config.raw["claim5"], "expected_rendered_wavs": 767}}
    with pytest.raises(ContractError, match="768 WAVs"):
        parse_claim5_settings(replace(config, raw=raw))
