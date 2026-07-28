from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import pytest

import random

import numpy as np
import torch

from repro.claim4_evaluation import build_claim4_arms
from repro.claim4_protocol import (
    Claim4ManifestItem,
    FrozenClaim4Asset,
    build_claim4_manifest,
    cremad_actor_id,
    is_claim4_eligible,
    load_frozen_claim4_asset_manifest,
    require_claim4_asset_coverage,
)
import repro.claim4_runtime as claim4_runtime
import repro.claim4_full as claim4_full
from repro.claim4_runtime import (
    _require_claim4_model_sample_rate,
    _require_claim4_canary_run_name,
    _load_claim4_frozen_canary_config,
    _require_regular_head_file,
    _validate_claim4_canary_receipt_payload,
    _validate_claim4_remote_complete,
    _transcript_for_filename,
    audit_claim4_wav_hashes,
    claim4_assets_for_item,
    claim4_target_and_reference_text,
    load_claim4_vectors,
    materialize_claim4_wav_pair,
    parse_claim4_settings,
    run_seeded_claim4_arms,
)
from repro.config import load_config
from repro.contracts import ContractError, sha256_file
from repro.cremad_selection import load_audio_vote_rows
from repro.hub_artifacts import ARTIFACT_SCHEMA_VERSION, artifact_destination


REPO = Path(__file__).resolve().parents[2]


def test_claim4_runtime_config_is_fully_pinned() -> None:
    config = load_config(REPO / "configs/claim4-cremad-mixed-directional.yaml")
    settings = parse_claim4_settings(config)
    assert settings.actor_count == 60
    assert settings.expected_rendered_wavs == 420
    assert settings.audio_quality["sample_rate_hz"] == 24000
    assert settings.license_attestations["public_artifact_policy"] == {
        "source_audio": "excluded",
        "generated_audio": "excluded",
        "model_weights": "excluded",
        "public_evidence": "derived_metrics_and_hashes_only",
        "generated_audio_redistribution": "not_relied_on",
    }
    assert settings.license_attestations["components"]["cremad_source"]["license_expression"] == (
        "ODbL-1.0 database; DbCL-1.0 contents"
    )
    assert all(
        component["acknowledged"] is True
        for component in settings.license_attestations["components"].values()
    )
    assert settings.canary_prerequisite["receipt_path"] == "receipts/claim4-runtime-canary.json"
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
    assert sha256_file(REPO / "configs/claim4-cremad-mixed-directional.yaml") == (
        load_config(REPO / "configs/claim4-cremad-mixed-canary.yaml").raw["claim4_canary"]["base_config_sha256"]
    )


def test_claim4_canary_rejects_full_config_hash_drift() -> None:
    config = load_config(REPO / "configs/claim4-cremad-mixed-canary.yaml")
    raw = deepcopy(config.raw)
    raw["claim4_canary"]["base_config_sha256"] = "0" * 64
    altered = config.__class__(source=config.source, raw=raw, digest=config.digest)
    with pytest.raises(ContractError, match="full-config hash mismatch"):
        parse_claim4_settings(altered)


def test_claim4_loaded_model_sample_rate_must_match_frozen_config() -> None:
    assert _require_claim4_model_sample_rate(
        model=type("Model", (), {"sample_rate": 24000})(), expected_sample_rate=24000
    ) == 24000
    with pytest.raises(ContractError, match="differs"):
        _require_claim4_model_sample_rate(
            model=type("Model", (), {"sample_rate": 22050})(), expected_sample_rate=24000
        )


def test_claim4_license_attestation_schema_rejects_policy_drift(tmp_path: Path) -> None:
    config = load_config(REPO / "configs/claim4-cremad-mixed-directional.yaml")
    raw = deepcopy(config.raw)
    raw["claim4"]["license_attestations"]["public_artifact_policy"]["generated_audio"] = "included"
    altered = config.__class__(source=tmp_path / "altered.yaml", raw=raw, digest=config.digest)
    with pytest.raises(ContractError, match="public artifact and redistribution policy"):
        parse_claim4_settings(altered)


def test_claim4_canary_receipt_requires_complete_pinned_configs_and_remote_complete() -> None:
    full = REPO / "configs/claim4-cremad-mixed-directional.yaml"
    canary = REPO / "configs/claim4-cremad-mixed-canary.yaml"
    pending = json.loads((REPO / "receipts/claim4-runtime-canary.json").read_text(encoding="utf-8"))
    kwargs = {
        "full_config_sha256": sha256_file(full),
        "canary_config_sha256": sha256_file(canary),
        "canary_config_digest": load_config(canary).digest,
        "required_status": "complete",
    }
    with pytest.raises(ContractError, match="blocked until the GPU canary receipt is complete"):
        _validate_claim4_canary_receipt_payload(
            receipt_bytes=json.dumps(pending).encode("utf-8"), **kwargs
        )

    complete = {
        **pending,
        "status": "complete",
        "canary_config_digest": load_config(canary).digest,
        "canary_repository_commit": "a" * 40,
        "canary_run_name": "claim4-cremad-mixed-runtime-canary",
        "terminal_job_url": "https://openresearch.sh/runs/claim4-canary",
        "remote_hub_commit_oid": "b" * 40,
        "remote_complete_url": artifact_destination(
            commit="a" * 40,
            config_digest=kwargs["canary_config_digest"],
            run_name="claim4-cremad-mixed-runtime-canary",
        ).at_revision("b" * 40).complete_url,
        "remote_complete_sha256": "a" * 64,
    }
    assert _validate_claim4_canary_receipt_payload(
        receipt_bytes=json.dumps(complete).encode("utf-8"), **kwargs
    )["status"] == "complete"

    wrong_config = {**complete, "full_config_sha256": "0" * 64}
    with pytest.raises(ContractError, match="does not bind the current full configuration"):
        _validate_claim4_canary_receipt_payload(
            receipt_bytes=json.dumps(wrong_config).encode("utf-8"), **kwargs
        )
    no_remote_completion = {**complete, "remote_complete_url": None}
    with pytest.raises(ContractError, match="remote_complete_url"):
        _validate_claim4_canary_receipt_payload(
            receipt_bytes=json.dumps(no_remote_completion).encode("utf-8"), **kwargs
        )


def test_claim4_canary_remote_complete_must_be_immutable_and_bind_run_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_commit = "a" * 40
    hub_commit_oid = "b" * 40
    canary_config_sha256 = "c" * 64
    run_name = "claim4-cremad-mixed-runtime-canary"
    destination = artifact_destination(
        commit=repository_commit,
        config_digest=canary_config_sha256,
        run_name=run_name,
    )
    complete = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "artifact_manifest_path": "hub_artifact_manifest.json",
        "artifact_manifest_sha256": "d" * 64,
        "files_uploaded": ["summary.json"],
        "destination": destination.as_dict(),
        "remote_prefix": destination.remote_prefix,
        "atomic": True,
    }
    payload = (json.dumps(complete, sort_keys=True) + "\n").encode("utf-8")
    receipt = {
        "canary_repository_commit": repository_commit,
        "canary_config_sha256": canary_config_sha256,
        "canary_config_digest": canary_config_sha256,
        "canary_run_name": run_name,
        "remote_hub_commit_oid": hub_commit_oid,
        "remote_complete_url": destination.at_revision(hub_commit_oid).complete_url,
        "remote_complete_sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setattr(claim4_runtime, "fetch_url_bytes", lambda _: payload)
    assert _validate_claim4_remote_complete(receipt=receipt)["destination"] == destination.as_dict()

    mutable = {**receipt, "remote_complete_url": destination.complete_url}
    with pytest.raises(ContractError, match="required immutable run URL"):
        _validate_claim4_remote_complete(receipt=mutable)

    wrong_identity = {**complete, "remote_prefix": "runs/other"}
    wrong_payload = (json.dumps(wrong_identity, sort_keys=True) + "\n").encode("utf-8")
    monkeypatch.setattr(claim4_runtime, "fetch_url_bytes", lambda _: wrong_payload)
    with pytest.raises(ContractError, match="does not bind the canary run identity"):
        _validate_claim4_remote_complete(
            receipt={**receipt, "remote_complete_sha256": hashlib.sha256(wrong_payload).hexdigest()}
        )


def test_claim4_full_and_canary_configs_must_match_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    configs = repo / "configs"
    configs.mkdir(parents=True)
    full = configs / "full.yaml"
    canary = configs / "canary.yaml"
    full.write_text("full: frozen\n", encoding="utf-8")
    canary.write_text("canary: frozen\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "configs"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", "freeze configs"],
        cwd=repo,
        check=True,
    )
    for path, label in ((full, "Claim 4 full configuration"), (canary, "Claim 4 frozen canary configuration")):
        relative, payload = _require_regular_head_file(repo=repo, path=path, label=label)
        assert relative == Path("configs") / path.name
        assert payload == path.read_bytes()
    canary.write_text("canary: changed\n", encoding="utf-8")
    with pytest.raises(ContractError, match="uncommitted content"):
        _require_regular_head_file(
            repo=repo, path=canary, label="Claim 4 frozen canary configuration"
        )


def test_claim4_canary_canonical_digest_differs_from_file_hash_and_binds_run_name(tmp_path: Path) -> None:
    canary = tmp_path / "canary.yaml"
    canary.write_text(
        "run:\n  name: committed-canary\n  seed: 1\n  stage: claim4\nclaims: [4]\n",
        encoding="utf-8",
    )
    frozen = _load_claim4_frozen_canary_config(canary_config_path=canary)
    assert frozen.run_name == "committed-canary"
    assert frozen.digest != sha256_file(canary)
    destination = artifact_destination(
        commit="a" * 40, config_digest=frozen.digest, run_name=frozen.run_name
    )
    assert frozen.digest in destination.complete_url
    assert sha256_file(canary) not in destination.complete_url
    _require_claim4_canary_run_name(
        receipt={"canary_run_name": "committed-canary"}, canary_config=frozen
    )
    with pytest.raises(ContractError, match="does not bind the frozen canary configuration"):
        _require_claim4_canary_run_name(
            receipt={"canary_run_name": "tampered-run-name"}, canary_config=frozen
        )


def test_claim4_full_gate_runs_before_cuda_source_or_model_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = load_config(REPO / "configs/claim4-cremad-mixed-directional.yaml")

    def stop_before_work(**_: object) -> object:
        raise ContractError("canary gate reached")

    monkeypatch.setattr(claim4_full, "_require_claim4_full_canary_receipt", stop_before_work)
    monkeypatch.setattr(
        claim4_full, "_atomic_verified_download",
        lambda **_: pytest.fail("source work must not begin before the canary receipt gate"),
    )
    with pytest.raises(ContractError, match="canary gate reached"):
        claim4_full.run_claim4_runtime_full(config, repo=REPO, run_root=tmp_path)


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


def test_frozen_asset_manifest_has_exact_selected_target_reference_coverage() -> None:
    config = load_config(REPO / "configs/claim4-cremad-mixed-directional.yaml")
    settings = parse_claim4_settings(config)
    rows = load_audio_vote_rows(
        REPO / ".cache" / "cremad" / settings.source["revision"] / "tabulatedVotes.csv"
    )
    manifest = build_claim4_manifest(rows, actor_count=60, seed=20260728)
    manifest_path = REPO / settings.source["asset_manifest_path"]
    assets = load_frozen_claim4_asset_manifest(
        manifest_path,
        expected_sha256=settings.source["asset_manifest_sha256"],
        revision=settings.source["revision"],
    )
    require_claim4_asset_coverage(manifest, assets)
    assert len(assets) == 120
    selected_target, selected_reference = claim4_assets_for_item(assets, manifest[0])
    assert selected_target.filename == f"{manifest[0].file_name}.wav"
    assert selected_reference.filename == f"{manifest[0].reference_file_name}.wav"
    assert all(asset.download_url.startswith("https://media.githubusercontent.com/media/") for asset in assets.values())
    assert all("api.github.com" not in asset.download_url for asset in assets.values())


def test_frozen_asset_manifest_rejects_tampering_wrong_revision_and_filename_coverage(tmp_path: Path) -> None:
    config = load_config(REPO / "configs/claim4-cremad-mixed-directional.yaml")
    settings = parse_claim4_settings(config)
    source_path = REPO / settings.source["asset_manifest_path"]
    payload = json.loads(source_path.read_text(encoding="utf-8"))

    payload["assets"][0]["size_bytes"] = 0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="LFS size is invalid"):
        load_frozen_claim4_asset_manifest(
            tampered,
            expected_sha256=sha256_file(tampered),
            revision=settings.source["revision"],
        )

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["source"]["revision"] = "0" * 40
    wrong_revision = tmp_path / "wrong-revision.json"
    wrong_revision.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="repository or revision mismatch"):
        load_frozen_claim4_asset_manifest(
            wrong_revision,
            expected_sha256=sha256_file(wrong_revision),
            revision=settings.source["revision"],
        )

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    first = payload["assets"][0]
    replacement = "0999_IOM_HAP_XX.wav"
    first["filename"] = replacement
    first["repository_path"] = f"AudioWAV/{replacement}"
    first["download_url"] = (
        "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/"
        f"{settings.source['revision']}/AudioWAV/{replacement}"
    )
    wrong_filename = tmp_path / "wrong-filename.json"
    wrong_filename.write_text(json.dumps(payload), encoding="utf-8")
    assets = load_frozen_claim4_asset_manifest(
        wrong_filename,
        expected_sha256=sha256_file(wrong_filename),
        revision=settings.source["revision"],
    )
    rows = load_audio_vote_rows(
        REPO / ".cache" / "cremad" / settings.source["revision"] / "tabulatedVotes.csv"
    )
    with pytest.raises(ContractError, match="asset coverage mismatch"):
        require_claim4_asset_coverage(
            build_claim4_manifest(rows, actor_count=60, seed=20260728), assets
        )


def test_runtime_materializes_only_predeclared_media_urls_and_has_no_metadata_resolver(tmp_path: Path) -> None:
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
    payloads = {
        "target": b"target frozen wav payload",
        "reference": b"reference frozen wav payload",
    }
    revision = "1658cd342dff90010aa843eaeebd53610a08b1dc"
    assets = {}
    for file_name, payload_key in ((item.file_name, "target"), (item.reference_file_name, "reference")):
        filename = f"{file_name}.wav"
        payload = payloads[payload_key]
        url = (
            "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/"
            f"{revision}/AudioWAV/{filename}"
        )
        assets[filename] = FrozenClaim4Asset(
            filename=filename,
            repository_path=f"AudioWAV/{filename}",
            git_blob_sha1="0" * 40,
            lfs_oid_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            download_url=url,
        )
    requested_urls: list[str] = []

    def fetcher(url: str) -> bytes:
        requested_urls.append(url)
        return payloads["target" if "IEO_HAP" in url else "reference"]

    target, reference, _, _ = materialize_claim4_wav_pair(
        cache_root=tmp_path, item=item, assets=assets, fetch_content=fetcher
    )
    assert target.read_bytes() == payloads["target"]
    assert reference.read_bytes() == payloads["reference"]
    assert len(requested_urls) == 2
    assert all(url.startswith("https://media.githubusercontent.com/media/") for url in requested_urls)
    assert all("api.github.com" not in url for url in requested_urls)
    runtime_source = inspect.getsource(claim4_runtime)
    assert "api.github.com" not in runtime_source
    assert not hasattr(claim4_runtime, "_json_url")
    assert not hasattr(claim4_runtime, "_lfs_pointer_and_download_url")


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
