"""Frozen contracts for the public Claim 5 semantic-conflict proxy.

The protocol deliberately separates deterministic public-data construction
from GPU synthesis.  It is a categorical text-target proxy rather than an
exact Table 3 reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from repro.config import ReproConfig, load_config
from repro.contracts import ContractError, sha256_file
from repro.ravdess_acquisition import RavdessAcquisitionSettings, parse_ravdess_acquisition_settings
from repro.ravdess_manifest import build_claim1_groups


TARGETS = ("angry", "happy", "sad", "surprise")
CLAIM5_ARMS = (
    "no_steer",
    "instruction_target",
    "cocoemo_alpha3",
    "cocoemo_alpha6",
    "wrong_target_alpha6",
    "negative_target_alpha6",
    "random_norm_matched_alpha6",
    "flow_side_emotional_reference",
)
WRONG_TARGET = {"angry": "happy", "happy": "sad", "sad": "surprise", "surprise": "angry"}
GROUPS = (("01", "01"), ("01", "02"), ("02", "01"), ("02", "02"))


@dataclass(frozen=True)
class Claim5Settings:
    mode: str
    actor_ids: tuple[str, ...]
    ravdess: RavdessAcquisitionSettings
    catalog_path: str
    catalog_sha256: str
    vector_contract: dict[str, Any]
    model: dict[str, str]
    evaluators: dict[str, Any]
    audio_quality: dict[str, Any]
    decisions: dict[str, Any]
    expected_cells: int
    expected_rendered_wavs: int


@dataclass(frozen=True)
class Claim5Cell:
    cell_id: str
    actor_id: str
    target: str
    statement: str
    repetition: str
    content_id: str
    text: str
    neutral_reference: str
    emotional_reference: str


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a mapping")
    return value


def _keys(value: Mapping[str, Any], label: str, expected: set[str]) -> None:
    if set(value) != expected:
        raise ContractError(f"{label} keys changed: missing={sorted(expected - set(value))}, unexpected={sorted(set(value) - expected)}")


def _positive(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ContractError(f"{label} must be a finite number")
    return float(value)


def parse_claim5_settings(config: ReproConfig) -> Claim5Settings:
    """Validate the full protocol or derive its hash-locked one-actor canary."""

    canary = config.raw.get("claim5_canary")
    if canary is not None:
        return _parse_canary(config, canary)
    value = _mapping(config.raw.get("claim5"), "claim5")
    _keys(value, "claim5", {
        "protocol", "mode", "catalog", "ravdess", "targets", "statements", "repetitions", "selection",
        "arms", "expected_cells", "expected_rendered_wavs", "model", "vector_contract",
        "evaluators", "audio_quality", "decisions",
    })
    if value["protocol"] != "ravdess_text_target_categorical_semantic_conflict_proxy_v1":
        raise ContractError("claim5.protocol must be the frozen public categorical semantic-conflict proxy")
    if value["mode"] != "full":
        raise ContractError("Claim 5 full config must set mode: full")
    if tuple(value["targets"]) != TARGETS or tuple(value["statements"]) != ("01", "02") or tuple(value["repetitions"]) != ("01", "02"):
        raise ContractError("Claim 5 targets/statements/repetitions drifted from the frozen public scope")
    if tuple(value["arms"]) != CLAIM5_ARMS:
        raise ContractError("Claim 5 arms must be exactly the eight frozen arms in order")
    catalog = _mapping(value["catalog"], "claim5.catalog")
    _keys(catalog, "claim5.catalog", {"path", "sha256", "schema_version", "author_owned"})
    if not isinstance(catalog["path"], str) or Path(catalog["path"]).is_absolute() or ".." in Path(catalog["path"]).parts:
        raise ContractError("claim5.catalog.path must be a safe relative path")
    if not isinstance(catalog["sha256"], str) or len(catalog["sha256"]) != 64 or catalog["schema_version"] != 1 or catalog["author_owned"] is not True:
        raise ContractError("Claim 5 catalog must pin the author-owned schema-v1 payload")
    selection = _mapping(value["selection"], "claim5.selection")
    _keys(selection, "claim5.selection", {"actor_count", "cell_assignment", "neutral_reference", "emotional_reference", "content_family_count", "actors_per_content_family"})
    if selection != {
        "actor_count": 24,
        "cell_assignment": "group_index_equals_actor_index_plus_target_index_modulo_four",
        "neutral_reference": "same_actor_statement_repetition_neutral_intensity_01",
        "emotional_reference": "same_actor_statement_repetition_target_intensity_01",
        "content_family_count": 16,
        "actors_per_content_family": 6,
    }:
        raise ContractError("Claim 5 RAVDESS selection/balance contract changed")
    cells = _positive(value["expected_cells"], "claim5.expected_cells")
    wavs = _positive(value["expected_rendered_wavs"], "claim5.expected_rendered_wavs")
    if cells != 96 or wavs != cells * len(CLAIM5_ARMS):
        raise ContractError("Claim 5 full accounting must be 96 cells and 768 WAVs")
    vector_contract = _mapping(value["vector_contract"], "claim5.vector_contract")
    _keys(vector_contract, "claim5.vector_contract", {"backbone", "operation", "layers", "shape", "paths", "sha256"})
    if vector_contract["backbone"] != "cosyvoice2" or vector_contract["operation"] != "attn_output" or tuple(vector_contract["layers"]) != (17, 14) or vector_contract["shape"] != [1, 896]:
        raise ContractError("Claim 5 vectors must use CosyVoice2 attn_output layers [17, 14] shape [1, 896]")
    for field in ("paths", "sha256"):
        entries = _mapping(vector_contract[field], f"claim5.vector_contract.{field}")
        if tuple(entries) != TARGETS or any(not isinstance(entries[target], str) or not entries[target] for target in TARGETS):
            raise ContractError(f"Claim 5 vector {field} must cover every target in frozen order")
    if any(len(vector_contract["sha256"][target]) != 64 for target in TARGETS):
        raise ContractError("Claim 5 vector SHA-256 pins must be 64 hexadecimal characters")
    model = _mapping(value["model"], "claim5.model")
    if model != {
        "id": "FunAudioLLM/CosyVoice2-0.5B", "revision": "eec1ae6c79877dbd9379285cf8789c9e0879293d",
        "source_url": "https://github.com/FunAudioLLM/CosyVoice.git", "source_revision": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
    }:
        raise ContractError("Claim 5 CosyVoice2 model pins changed")
    evaluators = _mapping(value["evaluators"], "claim5.evaluators")
    _keys(evaluators, "claim5.evaluators", {"emotion2vec", "wavlm", "whisper"})
    emotion = _mapping(evaluators["emotion2vec"], "claim5.evaluators.emotion2vec")
    if emotion.get("id") != "emotion2vec/emotion2vec_plus_large" or emotion.get("revision") != "6c303ba987b86b93193de93e34bb2b077a6bedc4" or tuple(emotion.get("canonical_labels", ())) != ("angry", "happy", "neutral", "sad", "surprise") or not isinstance(emotion.get("raw_label_map"), dict):
        raise ContractError("Claim 5 Emotion2Vec evaluator pin/labels changed")
    if evaluators["wavlm"] != {"id": "microsoft/wavlm-base-sv", "revision": "0a23162ffc49adcf42bdf836a00cb2eb45af3601"}:
        raise ContractError("Claim 5 WavLM evaluator pin changed")
    if evaluators["whisper"] != {"id": "openai/whisper-large-v3", "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1", "language": "en"}:
        raise ContractError("Claim 5 Whisper evaluator pin changed")
    decisions = _mapping(value["decisions"], "claim5.decisions")
    _keys(decisions, "claim5.decisions", {"bootstrap_method", "bootstrap_replicates", "bootstrap_seed", "min_primary_effect", "min_wavlm_change", "max_wer_increase"})
    if decisions["bootstrap_method"] != "actor_cluster_observed_cell_mean_v1":
        raise ContractError("Claim 5 bootstrap method must remain actor-clustered over observed balanced cells")
    if _positive(decisions["bootstrap_replicates"], "claim5.decisions.bootstrap_replicates") != 10_000 or _positive(decisions["bootstrap_seed"], "claim5.decisions.bootstrap_seed") != 20260728:
        raise ContractError("Claim 5 must use the predeclared 10k actor-cluster bootstrap")
    if (_finite(decisions["min_primary_effect"], "claim5.decisions.min_primary_effect") != 0.0 or _finite(decisions["min_wavlm_change"], "claim5.decisions.min_wavlm_change") != -0.02 or _finite(decisions["max_wer_increase"], "claim5.decisions.max_wer_increase") != 0.02):
        raise ContractError("Claim 5 decision thresholds changed")
    quality = _mapping(value["audio_quality"], "claim5.audio_quality")
    if quality != {"sample_rate_hz": 24000, "max_clipping_fraction": 0.001}:
        raise ContractError("Claim 5 audio integrity settings changed")
    evidence = _mapping(config.raw.get("evidence"), "evidence")
    if evidence.get("claim_evidence") is not True:
        raise ContractError("Claim 5 full protocol must explicitly be marked claim evidence")
    return Claim5Settings("full", tuple(f"{value:02d}" for value in range(1, 25)), parse_ravdess_acquisition_settings(_mapping(value["ravdess"], "claim5.ravdess")), catalog["path"], catalog["sha256"], vector_contract, model, evaluators, quality, decisions, cells, wavs)


def _parse_canary(config: ReproConfig, raw: object) -> Claim5Settings:
    value = _mapping(raw, "claim5_canary")
    _keys(value, "claim5_canary", {"base_config", "base_config_sha256", "actor_id", "expected_cells", "expected_rendered_wavs"})
    base_name = value["base_config"]
    if not isinstance(base_name, str) or Path(base_name).name != base_name or not isinstance(value["base_config_sha256"], str) or len(value["base_config_sha256"]) != 64:
        raise ContractError("Claim 5 canary must pin a sibling full-config filename and SHA-256")
    base_path = config.source.parent / base_name
    if not base_path.is_file() or sha256_file(base_path) != value["base_config_sha256"]:
        raise ContractError("Claim 5 canary full-config hash mismatch")
    base = parse_claim5_settings(load_config(base_path))
    actor = value["actor_id"]
    if not isinstance(actor, str) or actor not in base.actor_ids or value["expected_cells"] != 4 or value["expected_rendered_wavs"] != 32:
        raise ContractError("Claim 5 canary must be one configured actor x four targets x eight arms")
    evidence = _mapping(config.raw.get("evidence"), "evidence")
    if evidence.get("claim_evidence") is not False:
        raise ContractError("Claim 5 canary must set evidence.claim_evidence: false")
    return replace(base, mode="canary", actor_ids=(actor,), expected_cells=4, expected_rendered_wavs=32)


def load_claim5_catalog(*, repo: Path, settings: Claim5Settings) -> dict[str, tuple[dict[str, str], ...]]:
    path = repo / settings.catalog_path
    if not path.is_file() or sha256_file(path) != settings.catalog_sha256:
        raise ContractError("Claim 5 text catalog is missing or its SHA-256 pin changed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError("Claim 5 text catalog is invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("author_owned") is not True or not isinstance(payload.get("texts"), list):
        raise ContractError("Claim 5 text catalog schema/ownership contract failed")
    result: dict[str, list[dict[str, str]]] = {target: [] for target in TARGETS}
    ids: set[str] = set()
    for item in payload["texts"]:
        if not isinstance(item, dict) or set(item) != {"target", "text_id", "text"} or item["target"] not in TARGETS:
            raise ContractError("Claim 5 text catalog contains malformed target/text entries")
        text_id, text = item["text_id"], item["text"]
        if not isinstance(text_id, str) or not isinstance(text, str) or not text.strip() or text_id in ids:
            raise ContractError("Claim 5 text catalog has duplicate/empty text entries")
        ids.add(text_id)
        result[item["target"]].append({"text_id": text_id, "text": text})
    if any(len(result[target]) != 4 for target in TARGETS):
        raise ContractError("Claim 5 catalog must contain exactly four texts per target")
    return {target: tuple(result[target]) for target in TARGETS}


def build_claim5_cells(groups: Sequence[Mapping[str, object]], *, catalog: Mapping[str, Sequence[Mapping[str, str]]], actor_ids: Sequence[str]) -> list[Claim5Cell]:
    """Build actor-target cells using the frozen modulo-four group allocation."""

    by_group = {str(group.get("group_id")): group for group in groups}
    cells: list[Claim5Cell] = []
    for actor_index, actor_id in enumerate(actor_ids):
        for target_index, target in enumerate(TARGETS):
            statement, repetition = GROUPS[(actor_index + target_index) % 4]
            group_id = f"actor_{actor_id}_statement_{statement}_repetition_{repetition}"
            group = by_group.get(group_id)
            if group is None:
                raise ContractError(f"Claim 5 required RAVDESS group is missing: {group_id}")
            references = group.get("references")
            if not isinstance(references, Mapping):
                raise ContractError(f"Claim 5 group has no references: {group_id}")
            neutral, emotional = references.get("neutral"), references.get(target)
            if not isinstance(neutral, Mapping) or not isinstance(emotional, Mapping) or not isinstance(neutral.get("path"), str) or not isinstance(emotional.get("path"), str):
                raise ContractError(f"Claim 5 group lacks same-group neutral/emotional references: {group_id}/{target}")
            text_item = catalog[target][(actor_index + target_index) % 4]
            cells.append(Claim5Cell(
                cell_id=f"actor_{actor_id}__{target}__{text_item['text_id']}", actor_id=actor_id, target=target,
                statement=statement, repetition=repetition, content_id=text_item["text_id"], text=text_item["text"],
                neutral_reference=neutral["path"], emotional_reference=emotional["path"],
            ))
    _validate_cell_balance(cells, actor_ids=actor_ids)
    return cells


def _validate_cell_balance(cells: Sequence[Claim5Cell], *, actor_ids: Sequence[str]) -> None:
    if len(cells) != len(actor_ids) * len(TARGETS) or len({cell.cell_id for cell in cells}) != len(cells):
        raise ContractError("Claim 5 cell construction has duplicate or incomplete cells")
    families: dict[str, set[str]] = {}
    for cell in cells:
        families.setdefault(cell.content_id, set()).add(cell.actor_id)
        if cell.neutral_reference == cell.emotional_reference:
            raise ContractError("Claim 5 requires distinct neutral and emotional same-group references")
    if len(actor_ids) == 24:
        if len(families) != 16 or any(len(actors) != 6 for actors in families.values()):
            raise ContractError("Claim 5 content-family balance must be 16 families x six actors in full scope")
    elif len(cells) != len(actor_ids) * len(TARGETS) or any(len(actors) != 1 for actors in families.values()):
        raise ContractError("Claim 5 canary content assignment is incomplete")


def load_claim5_vectors(*, repo: Path, settings: Claim5Settings) -> dict[str, dict[int, np.ndarray]]:
    try:
        import torch
    except ImportError as exc:
        raise ContractError("Claim 5 vector loading requires torch") from exc
    result: dict[str, dict[int, np.ndarray]] = {}
    operation = settings.vector_contract["operation"]
    for target in TARGETS:
        path = repo / settings.vector_contract["paths"][target]
        if not path.is_file() or sha256_file(path) != settings.vector_contract["sha256"][target]:
            raise ContractError(f"Claim 5 released vector hash/file mismatch for {target}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        by_layer: dict[int, np.ndarray] = {}
        for layer in settings.vector_contract["layers"]:
            try:
                array = checkpoint["steering_vectors"][operation][layer].detach().cpu().numpy()
            except (KeyError, AttributeError, TypeError) as exc:
                raise ContractError(f"Claim 5 vector lacks {operation} layer {layer}: {target}") from exc
            if array.shape != tuple(settings.vector_contract["shape"]) or not np.all(np.isfinite(array)):
                raise ContractError(f"Claim 5 vector shape/finiteness contract failed: {target}/layer{layer}")
            by_layer[layer] = np.asarray(array, dtype=np.float32)
        result[target] = by_layer
    return result


def claim5_random_seed(*, base_seed: int, cell_id: str, layer: int) -> int:
    payload = f"{base_seed}\0claim5-random-v1\0{cell_id}\0{layer}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _norm_match(random: np.ndarray, target: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(random.reshape(-1)))
    target_norm = float(np.linalg.norm(target.reshape(-1)))
    if norm == 0.0 or target_norm == 0.0:
        raise ContractError("Claim 5 norm-matched control received a zero vector")
    return (random * (target_norm / norm)).astype(np.float32)


def build_claim5_arms(vectors: Mapping[str, Mapping[int, np.ndarray]], *, target: str, base_seed: int, cell_id: str) -> dict[str, dict[str, Any]]:
    if target not in TARGETS or set(vectors) != set(TARGETS):
        raise ContractError("Claim 5 arm construction requires all target vector families")
    layers = tuple(vectors[target])
    if layers != (17, 14) or any(tuple(vectors[emotion]) != layers for emotion in TARGETS):
        raise ContractError("Claim 5 vector layers must remain [17, 14] in order")
    target_vectors = {layer: np.asarray(vectors[target][layer], dtype=np.float32) for layer in layers}
    random_vectors = {
        layer: _norm_match(np.random.Generator(np.random.PCG64(claim5_random_seed(base_seed=base_seed, cell_id=cell_id, layer=layer))).standard_normal(target_vectors[layer].shape).astype(np.float32), target_vectors[layer])
        for layer in layers
    }
    result = {
        "no_steer": {"kind": "native_no_steer"},
        "instruction_target": {
            "kind": "native_instruction",
            "instruction": f"Say it in {'an' if target == 'angry' else 'a'} {'surprised' if target == 'surprise' else target} tone.",
        },
        "cocoemo_alpha3": {"kind": "steered", "vectors": target_vectors, "alpha": 3.0, "vector_target": target},
        "cocoemo_alpha6": {"kind": "steered", "vectors": target_vectors, "alpha": 6.0, "vector_target": target},
        "wrong_target_alpha6": {"kind": "steered", "vectors": {layer: np.asarray(vectors[WRONG_TARGET[target]][layer], dtype=np.float32) for layer in layers}, "alpha": 6.0, "vector_target": WRONG_TARGET[target]},
        "negative_target_alpha6": {"kind": "steered", "vectors": {layer: -target_vectors[layer] for layer in layers}, "alpha": 6.0, "vector_target": target},
        "random_norm_matched_alpha6": {"kind": "steered", "vectors": random_vectors, "alpha": 6.0, "vector_target": "random_norm_matched"},
        "flow_side_emotional_reference": {"kind": "flow_side_emotional_reference"},
    }
    if tuple(result) != CLAIM5_ARMS:
        raise ContractError("Claim 5 arm order drifted")
    for name, arm in result.items():
        if "vectors" in arm:
            if tuple(arm["vectors"]) != layers or any(array.shape != (1, 896) or not np.all(np.isfinite(array)) for array in arm["vectors"].values()):
                raise ContractError(f"Claim 5 arm vector contract failed: {name}")
    return result
