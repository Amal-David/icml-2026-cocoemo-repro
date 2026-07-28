"""Pure paired-vector and speaker-disjoint site-analysis logic for Claims 2/3."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from repro.claim23_extraction import COSYVOICE_HOOK_TYPES, LAYERS
from repro.claim23_manifest import CLAIM23_EMOTIONS, TARGET_EMOTIONS
from repro.contracts import ContractError


Site = tuple[str, int]


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    low: float
    high: float


def _site_key(record: Mapping[str, Any]) -> Site:
    operation, layer = record.get("operation"), record.get("layer")
    if operation not in COSYVOICE_HOOK_TYPES or not isinstance(layer, int) or layer not in LAYERS:
        raise ContractError("Claim 2/3 activation record has an invalid frozen operation/layer")
    return operation, layer


def _vector(record: Mapping[str, Any]) -> torch.Tensor:
    raw = record.get("activation")
    value = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32)
    if value.ndim != 1 or value.numel() == 0 or not torch.isfinite(value).all():
        raise ContractError("Claim 2/3 activation must be a finite non-empty rank-one vector")
    return value.float().cpu()


def index_activations(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, Site], torch.Tensor]:
    indexed: dict[tuple[str, Site], torch.Tensor] = {}
    for record in records:
        clip_id = record.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise ContractError("Claim 2/3 activation record has no clip_id")
        key = (clip_id, _site_key(record))
        if key in indexed:
            raise ContractError(f"duplicate Claim 2/3 activation record: {key}")
        indexed[key] = _vector(record)
    return indexed


def require_complete_activation_grid(records: Iterable[Mapping[str, Any]], *, clip_ids: Iterable[str]) -> None:
    rows = list(records)
    expected = {(clip_id, operation, layer) for clip_id in clip_ids for operation in COSYVOICE_HOOK_TYPES for layer in LAYERS}
    actual = {(record.get("clip_id"), record.get("operation"), record.get("layer")) for record in rows}
    if actual != expected:
        raise ContractError(
            "Claim 2/3 activation grid is incomplete or has unexpected rows: "
            f"missing={len(expected - actual)}, unexpected={len(actual - expected)}"
        )


def paired_mean_difference_vectors(records: Iterable[Mapping[str, Any]], pairs: Iterable[Mapping[str, Any]], *, split: str = "train") -> tuple[dict[str, dict[Site, torch.Tensor]], dict[str, list[str]]]:
    """Implement Eq. 6 as a mean of exact per-pair deltas over train actors."""

    indexed = index_activations(records)
    selected = [pair for pair in pairs if pair.get("split") == split]
    expected_per_emotion = 48 if split == "train" else None
    by_emotion: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in selected:
        emotion = pair.get("emotion")
        if emotion not in TARGET_EMOTIONS:
            raise ContractError("Claim 2/3 pair ledger contains an unsupported emotion")
        by_emotion[str(emotion)].append(pair)
    if set(by_emotion) != set(TARGET_EMOTIONS):
        raise ContractError("Claim 2/3 vector construction requires all four target emotions")
    vectors: dict[str, dict[Site, torch.Tensor]] = {}
    ledger: dict[str, list[str]] = {}
    for emotion in TARGET_EMOTIONS:
        emotion_pairs = sorted(by_emotion[emotion], key=lambda item: str(item.get("pair_id")))
        if expected_per_emotion is not None and len(emotion_pairs) != expected_per_emotion:
            raise ContractError(f"Claim 2/3 train vector requires 48 exact {emotion} pairs, found {len(emotion_pairs)}")
        sites: dict[Site, torch.Tensor] = {}
        for site in ((operation, layer) for operation in COSYVOICE_HOOK_TYPES for layer in LAYERS):
            deltas: list[torch.Tensor] = []
            for pair in emotion_pairs:
                emotional_id, neutral_id = pair.get("emotional_clip_id"), pair.get("neutral_clip_id")
                if not isinstance(emotional_id, str) or not isinstance(neutral_id, str):
                    raise ContractError("Claim 2/3 pair lacks both clip IDs")
                try:
                    delta = indexed[(emotional_id, site)] - indexed[(neutral_id, site)]
                except KeyError as exc:
                    raise ContractError(f"Claim 2/3 pair activation missing at {site}") from exc
                deltas.append(delta)
            shapes = {tuple(delta.shape) for delta in deltas}
            if len(shapes) != 1:
                raise ContractError(f"Claim 2/3 paired activation shape mismatch at {site}")
            vector = torch.stack(deltas).mean(dim=0)
            if not torch.isfinite(vector).all():
                raise ContractError(f"Claim 2/3 vector is non-finite at {site}")
            sites[site] = vector
        vectors[emotion] = sites
        ledger[emotion] = [str(pair["pair_id"]) for pair in emotion_pairs]
    return vectors, ledger


def cosine_alignment(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ContractError("Claim 2/3 cosine alignment has a zero or non-finite norm")
    result = float(torch.dot(left, right) / denominator)
    if not math.isfinite(result):
        raise ContractError("Claim 2/3 cosine alignment is non-finite")
    return result


def actor_cluster_bootstrap(values: Iterable[Mapping[str, Any]], *, value_key: str, actor_key: str = "actor", replicates: int = 10_000, seed: int = 20260728) -> BootstrapInterval:
    rows = list(values)
    if replicates < 1 or not rows:
        raise ContractError("Claim 2/3 bootstrap requires positive replicates and non-empty rows")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        actor, value = row.get(actor_key), row.get(value_key)
        if not isinstance(actor, str) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ContractError("Claim 2/3 bootstrap row is malformed")
        grouped[actor].append(float(value))
    actor_means = np.asarray([np.mean(grouped[actor]) for actor in sorted(grouped)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(actor_means, size=(replicates, len(actor_means)), replace=True).mean(axis=1)
    return BootstrapInterval(float(actor_means.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)))


def claim2_directional_verdict(results: Mapping[str, Mapping[str, BootstrapInterval | float]]) -> str:
    """Apply the preregistered proxy decision, never paper-table thresholds."""

    for emotion in TARGET_EMOTIONS:
        outcome = results.get(emotion)
        if outcome is None:
            raise ContractError("Claim 2 verdict requires every target emotion")
        matched = outcome.get("matched")
        mismatched = outcome.get("mismatched")
        shuffled = outcome.get("shuffled")
        random_99 = outcome.get("random_99")
        if not isinstance(matched, BootstrapInterval) or not isinstance(mismatched, BootstrapInterval) or not isinstance(shuffled, BootstrapInterval) or not isinstance(random_99, (int, float)):
            raise ContractError("Claim 2 verdict inputs are malformed")
        if matched.high < min(mismatched.low, shuffled.low, float(random_99)):
            return "proxy_failed"
    if all(
        isinstance(outcome.get("matched"), BootstrapInterval)
        and isinstance(outcome.get("mismatched"), BootstrapInterval)
        and isinstance(outcome.get("shuffled"), BootstrapInterval)
        and outcome["matched"].low > outcome["mismatched"].high
        and outcome["matched"].low > outcome["shuffled"].high
        and outcome["matched"].low > float(outcome["random_99"])
        for outcome in results.values()
    ):
        return "proxy_supported"
    return "proxy_inconclusive"


def shuffled_train_vectors(
    records: Iterable[Mapping[str, Any]], pairs: Iterable[Mapping[str, Any]], *, site: Site, replicates: int = 100, seed: int = 20260728
) -> dict[str, list[torch.Tensor]]:
    """Generate 100 within-group train-label permutation vectors per emotion."""

    if replicates != 100:
        raise ContractError("Claim 2 requires exactly 100 within-group label permutations")
    indexed = index_activations(records)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if pair.get("split") == "train":
            grouped[str(pair.get("pair_id", "")).rsplit("_emotion_", 1)[0]].append(pair)
    if len(grouped) != 48 or any({row.get("emotion") for row in group} != set(TARGET_EMOTIONS) for group in grouped.values()):
        raise ContractError("Claim 2 shuffled-label control needs 48 complete train emotion groups")
    deltas_by_group: dict[str, dict[str, torch.Tensor]] = {}
    for group_id, group in grouped.items():
        deltas_by_group[group_id] = {}
        for pair in group:
            emotion = str(pair["emotion"])
            deltas_by_group[group_id][emotion] = indexed[(str(pair["emotional_clip_id"]), site)] - indexed[(str(pair["neutral_clip_id"]), site)]
    rng = np.random.default_rng(seed)
    output: dict[str, list[torch.Tensor]] = {emotion: [] for emotion in TARGET_EMOTIONS}
    for _ in range(replicates):
        assigned: dict[str, list[torch.Tensor]] = {emotion: [] for emotion in TARGET_EMOTIONS}
        for group_id in sorted(deltas_by_group):
            source = [deltas_by_group[group_id][emotion] for emotion in TARGET_EMOTIONS]
            for label, delta in zip(rng.permutation(TARGET_EMOTIONS), source):
                assigned[str(label)].append(delta)
        for emotion in TARGET_EMOTIONS:
            output[emotion].append(torch.stack(assigned[emotion]).mean(dim=0))
    return output


def norm_matched_random_vectors(vector: torch.Tensor, *, count: int = 100, seed: int = 20260728) -> list[torch.Tensor]:
    if count != 100:
        raise ContractError("Claim 2 requires exactly 100 norm-matched random vectors")
    norm = float(torch.linalg.vector_norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ContractError("Claim 2 cannot norm-match a zero or non-finite vector")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = []
    for _ in range(count):
        random_vector = torch.randn(vector.shape, generator=generator, dtype=vector.dtype)
        output.append(random_vector * (norm / torch.linalg.vector_norm(random_vector)))
    return output


def claim2_heldout_controls(
    records: Iterable[Mapping[str, Any]], pairs: Iterable[Mapping[str, Any]], *, target_emotion: str, site: Site,
    target_vector: torch.Tensor, shuffled_vectors: Iterable[torch.Tensor], random_vectors: Iterable[torch.Tensor],
    shuffled_seed: int, random_seed: int, split: str = "test",
) -> dict[str, Any]:
    """Evaluate exact test deltas against matched, mismatch, shuffle, and random controls."""

    if target_emotion not in TARGET_EMOTIONS:
        raise ContractError("Claim 2 held-out controls require a target emotion")
    shuffled, random = list(shuffled_vectors), list(random_vectors)
    if len(shuffled) != 100 or len(random) != 100:
        raise ContractError("Claim 2 held-out controls require 100 shuffled and 100 random vectors")
    indexed = index_activations(records)
    all_pairs = list(pairs)
    selected = [pair for pair in all_pairs if pair.get("split") == split and pair.get("emotion") == target_emotion]
    if len(selected) != 28:
        raise ContractError(f"Claim 2 held-out controls require 28 test {target_emotion} pairs")
    neutral_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in all_pairs:
        if pair.get("split") == split:
            neutral_by_cell[(str(pair.get("statement")), str(pair.get("repetition")))].append(pair)
    matched_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    random_scores: list[float] = []
    control_ledger: list[dict[str, Any]] = []
    for pair in selected:
        emotional_id, neutral_id = str(pair["emotional_clip_id"]), str(pair["neutral_clip_id"])
        delta = indexed[(emotional_id, site)] - indexed[(neutral_id, site)]
        actor = str(pair["actor"])
        candidates = sorted(
            {str(candidate["neutral_clip_id"]): candidate for candidate in neutral_by_cell[(str(pair["statement"]), str(pair["repetition"]))]
             if str(candidate["actor"]) != actor}.items()
        )
        if not candidates:
            raise ContractError("Claim 2 mismatch control has no same-text/repetition different-speaker neutral")
        selection = int.from_bytes(hashlib.sha256(str(pair["pair_id"]).encode("ascii")).digest()[:8], "big") % len(candidates)
        mismatch_neutral_id = candidates[selection][0]
        mismatch_delta = indexed[(emotional_id, site)] - indexed[(mismatch_neutral_id, site)]
        matched_alignment = cosine_alignment(delta, target_vector)
        mismatch_alignment = cosine_alignment(mismatch_delta, target_vector)
        shuffled_alignment = float(np.mean([cosine_alignment(delta, vector) for vector in shuffled]))
        per_pair_random = [cosine_alignment(delta, vector) for vector in random]
        matched_rows.append({"actor": actor, "value": matched_alignment})
        mismatch_rows.append({"actor": actor, "value": mismatch_alignment})
        shuffled_rows.append({"actor": actor, "value": shuffled_alignment})
        random_scores.extend(per_pair_random)
        control_ledger.append({
            "pair_id": str(pair["pair_id"]), "actor": actor, "matched_neutral_clip_id": neutral_id,
            "mismatched_neutral_clip_id": mismatch_neutral_id, "mismatch_candidate_index": selection,
            "mismatch_candidate_count": len(candidates), "matched_alignment": matched_alignment,
            "mismatched_alignment": mismatch_alignment, "shuffled_alignment_mean": shuffled_alignment,
            "random_alignment_99_per_pair": float(np.quantile(np.asarray(per_pair_random), 0.99)),
        })
    random_99 = float(np.quantile(np.asarray(random_scores), 0.99))
    return {
        "matched": actor_cluster_bootstrap(matched_rows, value_key="value"),
        "mismatched": actor_cluster_bootstrap(mismatch_rows, value_key="value"),
        "shuffled": actor_cluster_bootstrap(shuffled_rows, value_key="value"),
        "random_99": random_99,
        "pair_count": len(selected),
        "control_recompute": {
            "site": f"{site[0]}@{site[1]}", "split": split, "shuffled_draw_count": len(shuffled),
            "shuffled_seed": shuffled_seed, "random_draw_count": len(random), "random_seed": random_seed,
            "random_global_99": random_99,
        },
        "control_ledger": sorted(control_ledger, key=lambda row: row["pair_id"]),
    }


def paired_delta_alignment_controls(
    records: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
    *,
    target_emotion: str,
    site: Site,
    target_vector: torch.Tensor,
    shuffled_vectors: Iterable[torch.Tensor],
    random_vectors: Iterable[torch.Tensor],
    split: str = "test",
) -> dict[str, Any]:
    """Score held-out exact deltas against preregistered vector controls.

    ``shuffled_vectors`` must be produced by 100 within-group train-label
    permutations and ``random_vectors`` by 100 independent norm-matched draws.
    Keeping vector construction outside this evaluator makes the testable
    accounting boundary explicit and prevents a re-pairing placebo that would
    algebraically preserve the global mean difference.
    """

    if target_emotion not in TARGET_EMOTIONS:
        raise ContractError("Claim 2 alignment needs a non-neutral target emotion")
    shuffled = list(shuffled_vectors)
    random = list(random_vectors)
    if len(shuffled) != 100 or len(random) != 100:
        raise ContractError("Claim 2 alignment requires exactly 100 shuffled and 100 random controls")
    indexed = index_activations(records)
    test_pairs = [pair for pair in pairs if pair.get("split") == split and pair.get("emotion") == target_emotion]
    expected = 28 if split == "test" else 20 if split == "validation" else 48
    if len(test_pairs) != expected:
        raise ContractError(f"Claim 2 {split} alignment needs {expected} exact {target_emotion} pairs")
    rows: list[dict[str, Any]] = []
    shuffled_scores: list[float] = []
    random_scores: list[float] = []
    for pair in test_pairs:
        emotional_id, neutral_id = pair.get("emotional_clip_id"), pair.get("neutral_clip_id")
        if not isinstance(emotional_id, str) or not isinstance(neutral_id, str):
            raise ContractError("Claim 2 alignment pair lacks clip IDs")
        delta = indexed[(emotional_id, site)] - indexed[(neutral_id, site)]
        rows.append({"actor": pair.get("actor"), "matched": cosine_alignment(delta, target_vector)})
        shuffled_scores.extend(cosine_alignment(delta, vector) for vector in shuffled)
        random_scores.extend(cosine_alignment(delta, vector) for vector in random)
    if any(not isinstance(row["actor"], str) for row in rows):
        raise ContractError("Claim 2 alignment pair lacks actor identity")
    return {
        "matched": actor_cluster_bootstrap(rows, value_key="matched"),
        "shuffled_mean": float(np.mean(shuffled_scores)),
        "random_99": float(np.quantile(np.asarray(random_scores), 0.99)),
        "pair_count": len(rows),
    }


def _records_for_site(records: Iterable[Mapping[str, Any]], site: Site) -> list[Mapping[str, Any]]:
    return [record for record in records if _site_key(record) == site]


def fit_five_class_centroids(records: Iterable[Mapping[str, Any]], *, site: Site, split: str = "train") -> dict[str, torch.Tensor]:
    rows = [row for row in _records_for_site(records, site) if row.get("split") == split]
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in rows:
        emotion = row.get("emotion")
        if emotion not in CLAIM23_EMOTIONS:
            raise ContractError("Claim 3 centroid input has an invalid emotion")
        grouped[str(emotion)].append(_vector(row))
    if set(grouped) != set(CLAIM23_EMOTIONS):
        raise ContractError(f"Claim 3 centroid fit needs all five classes at {site}")
    if any(len(grouped[emotion]) != 48 for emotion in CLAIM23_EMOTIONS) and split == "train":
        raise ContractError(f"Claim 3 train centroid fit requires 48 clips per class at {site}")
    shapes = {tuple(vector.shape) for values in grouped.values() for vector in values}
    if len(shapes) != 1:
        raise ContractError(f"Claim 3 centroid fit has incompatible activation shapes at {site}")
    return {emotion: torch.stack(grouped[emotion]).mean(dim=0) for emotion in CLAIM23_EMOTIONS}


def nearest_centroid_predictions(records: Iterable[Mapping[str, Any]], *, centroids: Mapping[str, torch.Tensor], site: Site, split: str) -> list[dict[str, Any]]:
    if tuple(centroids) != CLAIM23_EMOTIONS:
        raise ContractError("Claim 3 nearest-centroid classifier must contain exactly five ordered classes")
    outputs: list[dict[str, Any]] = []
    for row in _records_for_site(records, site):
        if row.get("split") != split:
            continue
        vector = _vector(row)
        scores = {emotion: float(torch.sum((vector - centroids[emotion]) ** 2)) for emotion in CLAIM23_EMOTIONS}
        prediction = min(CLAIM23_EMOTIONS, key=lambda emotion: (scores[emotion], emotion))
        outputs.append({
            "clip_id": row["clip_id"], "actor": row["actor"], "group_id": row.get("group_id"),
            "truth": row["emotion"], "prediction": prediction, "correct": prediction == row["emotion"],
        })
    expected = 100 if split == "validation" else 140 if split == "test" else 240
    if len(outputs) != expected:
        raise ContractError(f"Claim 3 {split} prediction accounting mismatch at {site}: expected={expected}, actual={len(outputs)}")
    return outputs


def accuracy(predictions: Iterable[Mapping[str, Any]]) -> float:
    rows = list(predictions)
    if not rows:
        raise ContractError("Claim 3 accuracy requires predictions")
    return sum(bool(row.get("correct")) for row in rows) / len(rows)


def score_all_sites(records: Iterable[Mapping[str, Any]], *, split: str = "validation") -> dict[Site, float]:
    rows = list(records)
    scores: dict[Site, float] = {}
    for operation in COSYVOICE_HOOK_TYPES:
        for layer in LAYERS:
            site = (operation, layer)
            centroids = fit_five_class_centroids(rows, site=site)
            scores[site] = accuracy(nearest_centroid_predictions(rows, centroids=centroids, site=site, split=split))
    return scores


def select_top_sites(validation_scores: Mapping[Site, float], *, count: int = 2) -> list[Site]:
    expected = {(operation, layer) for operation in COSYVOICE_HOOK_TYPES for layer in LAYERS}
    if set(validation_scores) != expected or count != 2:
        raise ContractError("Claim 3 selection must score all 240 sites and select exactly two")
    return sorted(validation_scores, key=lambda site: (-validation_scores[site], site[0], site[1]))[:count]


def max_t_permutation_pvalue(records: Iterable[Mapping[str, Any]], *, observed_max: float, permutations: int = 200, seed: int = 20260728) -> tuple[float, list[float]]:
    """Within-group label permutations with a maximum-over-240-sites null."""

    if permutations != 200:
        raise ContractError("Claim 3 must use exactly 200 max-T permutations")
    rows = list(records)
    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    if len(train) != 240 * 240 or len(validation) != 100 * 240:
        raise ContractError("Claim 3 max-T requires a complete all-site train/validation activation grid")
    # One shared deterministic label shuffle per group.  The implementation is
    # vectorized over all 200 labelings at each site: copying 115k records and
    # refitting 240 Python classifiers per permutation would turn this bounded
    # null into a multi-hour bookkeeping task rather than an analysis.
    rng = np.random.default_rng(seed)
    indexed = index_activations(rows)
    by_clip = {str(row["clip_id"]): row for row in train if _site_key(row) == ("attn_output", 0)}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in by_clip.values():
        groups[str(row["group_id"])].append(row)
    if len(groups) != 48 or any(len(group) != 5 for group in groups.values()):
        raise ContractError("Claim 3 train labels must form 48 complete five-class groups")
    group_rows = [sorted(group, key=lambda row: CLAIM23_EMOTIONS.index(str(row["emotion"]))) for _, group in sorted(groups.items())]
    validation_rows = sorted(
        [row for row in validation if _site_key(row) == ("attn_output", 0)], key=lambda row: str(row["clip_id"])
    )
    if len(validation_rows) != 100:
        raise ContractError("Claim 3 validation labels must contain 100 clips")
    label_ids = torch.tensor([CLAIM23_EMOTIONS.index(str(row["emotion"])) for row in validation_rows], dtype=torch.long)
    permutation_ids = np.empty((permutations, len(group_rows), len(CLAIM23_EMOTIONS)), dtype=np.int64)
    for replicate in range(permutations):
        for group_index in range(len(group_rows)):
            permutation_ids[replicate, group_index] = rng.permutation(len(CLAIM23_EMOTIONS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = torch.nn.functional.one_hot(torch.from_numpy(permutation_ids), num_classes=len(CLAIM23_EMOTIONS)).to(device=device, dtype=torch.float32)
    max_scores = torch.zeros(permutations, device=device)
    for operation in COSYVOICE_HOOK_TYPES:
        for layer in LAYERS:
            site = (operation, layer)
            train_tensor = torch.stack([
                torch.stack([indexed[(str(row["clip_id"]), site)] for row in group]) for group in group_rows
            ]).to(device)
            validation_tensor = torch.stack([indexed[(str(row["clip_id"]), site)] for row in validation_rows]).to(device)
            centroids = torch.einsum("rgct,gcd->rtd", assignment, train_tensor) / len(group_rows)
            distances = (
                validation_tensor.square().sum(dim=1)[None, :, None]
                + centroids.square().sum(dim=2)[:, None, :]
                - 2.0 * torch.einsum("vd,rcd->rvc", validation_tensor, centroids)
            )
            predictions = distances.argmin(dim=2)
            site_scores = (predictions == label_ids.to(device)[None, :]).float().mean(dim=1)
            max_scores = torch.maximum(max_scores, site_scores)
    null = [float(value) for value in max_scores.cpu()]
    p_value = (1 + sum(value >= observed_max for value in null)) / (permutations + 1)
    return p_value, null


def claim3_site_verdict(*, selected_site: Site, max_t_p_value: float, selected_test: BootstrapInterval, attention_minus_median: BootstrapInterval) -> str:
    if not 0.0 <= max_t_p_value <= 1.0:
        raise ContractError("Claim 3 max-T p-value is invalid")
    if selected_test.high < 0.20 or attention_minus_median.high < 0.0:
        return "proxy_failed"
    if selected_site[0] == "attn_output" and 10 <= selected_site[1] <= 17 and max_t_p_value <= 0.05 and selected_test.low > 0.20 and attention_minus_median.low > 0.0:
        return "proxy_supported"
    return "proxy_inconclusive"


def vector_sha256(vector: torch.Tensor) -> str:
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ContractError("Claim 2 vector hash requires a finite rank-one vector")
    return hashlib.sha256(vector.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
