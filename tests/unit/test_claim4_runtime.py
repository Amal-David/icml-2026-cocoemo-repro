from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import random

import numpy as np
import torch

from repro.claim4_evaluation import build_claim4_arms
from repro.claim4_protocol import Claim4ManifestItem, build_claim4_manifest, cremad_actor_id, is_claim4_eligible
from repro.claim4_runtime import (
    _transcript_for_filename,
    audit_claim4_wav_hashes,
    claim4_target_and_reference_text,
    load_claim4_vectors,
    parse_claim4_settings,
    run_seeded_claim4_arms,
)
from repro.config import load_config
from repro.contracts import ContractError
from repro.cremad_selection import load_audio_vote_rows


REPO = Path(__file__).resolve().parents[2]


def test_claim4_runtime_config_is_fully_pinned() -> None:
    config = load_config(REPO / "configs/claim4-cremad-mixed-directional.yaml")
    settings = parse_claim4_settings(config)
    assert settings.actor_count == 60
    assert settings.expected_rendered_wavs == 420
    assert settings.evaluators["emotion2vec"] == {
        "id": "emotion2vec/emotion2vec_plus_large",
        "revision": "6c303ba987b86b93193de93e34bb2b077a6bedc4",
        "canonical_labels": ["angry", "happy", "neutral", "sad", "surprise"],
        "raw_label_map": settings.evaluators["emotion2vec"]["raw_label_map"],
    }
    assert settings.evaluators["wavlm"]["revision"] == "0a23162ffc49adcf42bdf836a00cb2eb45af3601"
    assert settings.evaluators["whisper"]["revision"] == "06f233fe06e710322aca913c1bc4249a0d71fce1"
    vectors = load_claim4_vectors(settings, repo=REPO)
    assert {name: value.shape for name, value in vectors.items()} == {
        "angry": (1, 896), "happy": (1, 896), "sad": (1, 896), "surprise": (1, 896)
    }
    assert _transcript_for_filename("1001_IEO_HAP_XX") == "It's eleven o'clock."


def test_claim4_canary_is_separate_from_full_evidence_accounting() -> None:
    full = parse_claim4_settings(load_config(REPO / "configs/claim4-cremad-mixed-directional.yaml"))
    canary = parse_claim4_settings(load_config(REPO / "configs/claim4-cremad-mixed-canary.yaml"))
    assert full.mode == "full"
    assert (full.expected_samples, full.expected_rendered_wavs) == (60, 420)
    assert canary.mode == "canary"
    assert (canary.expected_samples, canary.expected_rendered_wavs) == (1, 7)


def test_canary_uses_reference_sentence_as_prompt_when_codes_differ() -> None:
    item = Claim4ManifestItem(
        actor_id="1001",
        row_id=1,
        file_name="1001_IEO_HAP_XX",
        provided_label="H",
        distribution={"p_angry": 0.4, "p_happy": 0.2, "p_sad": 0.1, "p_surprise": 0.0, "p_neutral": 0.3},
        reference_row_id=2,
        reference_file_name="1001_DFA_NEU_XX",
        reference_provided_label="N",
        reference_majority_labels=("N",),
    )
    target_text, reference_text = claim4_target_and_reference_text(item)
    assert target_text == "It's eleven o'clock."
    assert reference_text == "Don't forget a jacket."
    assert target_text != reference_text


def test_pinned_vote_inventory_matches_claim4_manifest_contract() -> None:
    votes_path = REPO / ".cache" / "cremad" / "1658cd342dff90010aa843eaeebd53610a08b1dc" / "tabulatedVotes.csv"
    rows = load_audio_vote_rows(votes_path)
    eligible = [row for row in rows if is_claim4_eligible(row)]
    assert len(rows) == 7442
    assert len(eligible) == 118
    assert len({cremad_actor_id(row.file_name) for row in eligible}) == 60
    manifest = build_claim4_manifest(rows, actor_count=60, seed=20260728)
    assert all(item.reference_provided_label == "N" for item in manifest)
    assert all(item.reference_majority_labels == ("N",) for item in manifest)
    assert all(item.file_name != item.reference_file_name for item in manifest)


def test_claim4_runtime_config_rejects_arm_or_label_drift(tmp_path: Path) -> None:
    config = load_config(REPO / "configs/claim4-cremad-mixed-directional.yaml")
    raw = deepcopy(config.raw)
    raw["claim4"]["arms"] = raw["claim4"]["arms"][:-1]
    altered = config.__class__(source=tmp_path / "altered.yaml", raw=raw, digest=config.digest)
    with pytest.raises(ContractError, match="seven frozen arms"):
        parse_claim4_settings(altered)
    raw = deepcopy(config.raw)
    raw["claim4"]["evaluators"]["emotion2vec"]["canonical_labels"] = ["angry"]
    altered = config.__class__(source=tmp_path / "altered.yaml", raw=raw, digest=config.digest)
    with pytest.raises(ContractError, match="Emotion2Vec pin/label contract"):
        parse_claim4_settings(altered)


def test_generation_seed_is_identical_for_every_arm_and_order_independent() -> None:
    vectors = {emotion: np.array([index + 1.0]) for index, emotion in enumerate(("angry", "happy", "sad", "surprise"))}
    arms = build_claim4_arms(
        vectors,
        {"p_angry": 0.4, "p_happy": 0.2, "p_sad": 0.1, "p_surprise": 0.0, "p_neutral": 0.3},
        seed=20260728,
        sample_id="1001_IEO_HAP_XX",
    )

    def fake_generate(_: str, __: object) -> tuple[float, float, float]:
        return (random.random(), float(np.random.random()), float(torch.rand(()).item()))

    forward = run_seeded_claim4_arms(arms, generate=fake_generate)
    reverse = run_seeded_claim4_arms(dict(reversed(list(arms.items()))), generate=fake_generate)
    assert forward == reverse


def test_wav_hash_audit_allows_only_the_explicit_zero_neutral_pair() -> None:
    records = []
    for arm in (
        "released_four_way", "dominant_non_neutral", "shuffled_distribution", "random_norm_matched",
        "renormalized_non_neutral_diagnostic", "neutral_zero_vector_diagnostic", "identity_neutral_control",
    ):
        records.append({
            "sample_id": "sample",
            "arm": arm,
            "wav_sha256": "same" if arm in {"released_four_way", "neutral_zero_vector_diagnostic"} else arm,
        })
    assert audit_claim4_wav_hashes(records)["collisions"][0]["arms"] == ["neutral_zero_vector_diagnostic", "released_four_way"]
    records[-1]["wav_sha256"] = "same"
    with pytest.raises(ContractError, match="unexpected Claim 4 WAV hash collisions"):
        audit_claim4_wav_hashes(records)
