"""Deterministic audit of neutral mass in the released mixed-emotion path.

The paper's mixed-label target is a five-way distribution, but the released
``scripts/synthesize.py`` only loads four non-neutral steering directions.  This
module does not change that implementation.  It makes the resulting algebra and
its deviation from non-neutral renormalisation explicit for reproducible review.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from repro.contracts import ContractError, sha256_file


EMOTIONS = ("angry", "happy", "sad", "surprise", "neutral")
RELEASED_MIXING_EMOTIONS = EMOTIONS[:-1]
PROBE_DISTRIBUTION = {
    "p_angry": 0.35,
    "p_happy": 0.15,
    "p_sad": 0.10,
    "p_surprise": 0.00,
    "p_neutral": 0.40,
}


def _probabilities(distribution: Mapping[str, float]) -> dict[str, float]:
    expected = {f"p_{emotion}" for emotion in EMOTIONS}
    unexpected = sorted(set(distribution) - expected)
    missing = sorted(expected - set(distribution))
    if missing or unexpected:
        raise ContractError(f"invalid emotion distribution: missing={missing}, unexpected={unexpected}")

    result = {key: float(distribution[key]) for key in sorted(expected)}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ContractError("emotion proportions must be finite and non-negative")
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError(f"emotion proportions must sum to 1.0, found {total:.12g}")
    return result


def _weighted_sum(
    vectors: Mapping[str, torch.Tensor], weights: Mapping[str, float]
) -> torch.Tensor:
    if not vectors:
        raise ContractError("at least one non-neutral steering vector is required")
    missing = sorted(set(weights) - set(vectors))
    if missing:
        raise ContractError(f"weights refer to unavailable vectors: {missing}")
    first = next(iter(vectors.values()))
    if any(vector.shape != first.shape for vector in vectors.values()):
        raise ContractError("steering vectors must have identical shapes")
    return sum((weights[emotion] * vectors[emotion] for emotion in weights), torch.zeros_like(first))


def compare_neutral_semantics(
    vectors: Mapping[str, torch.Tensor], distribution: Mapping[str, float]
) -> dict[str, Any]:
    """Compare released, zero-neutral, and non-neutral-renormalised mixtures.

    ``vectors`` deliberately has no neutral direction.  The released helper
    filters label columns to available vector names, making it equivalent to an
    explicit zero neutral vector.  Renormalisation is included only as an audit
    counterfactual, never as a replacement for released behavior.
    """

    probabilities = _probabilities(distribution)
    loaded_emotions = tuple(vectors)
    if "neutral" in loaded_emotions:
        raise ContractError("this audit requires a missing neutral steering vector")
    unknown = sorted(set(loaded_emotions) - set(RELEASED_MIXING_EMOTIONS))
    if unknown:
        raise ContractError(f"unsupported released steering emotions: {unknown}")

    released_weights = {emotion: probabilities[f"p_{emotion}"] for emotion in loaded_emotions}
    retained_mass = sum(released_weights.values())
    released = _weighted_sum(vectors, released_weights)

    # A zero neutral vector contributes exactly zero to the released weighted sum.
    explicit_zero_neutral = released.clone()
    result: dict[str, Any] = {
        "distribution": probabilities,
        "loaded_emotions": list(loaded_emotions),
        "missing_positive_weight_emotions": [
            emotion
            for emotion in EMOTIONS
            if emotion not in loaded_emotions and probabilities[f"p_{emotion}"] > 0
        ],
        "released_semantics": "unrenormalized_non_neutral",
        "explicit_zero_neutral_semantics": "identical_to_released",
        "retained_non_neutral_mass": retained_mass,
        "released_vs_explicit_zero_max_abs_diff": float(
            (released - explicit_zero_neutral).abs().max().item()
        ),
    }

    if retained_mass == 0:
        result.update(
            {
                "renormalized_non_neutral_semantics": "undefined_all_neutral",
                "released_vs_renormalized_l2": None,
                "released_to_renormalized_norm_ratio": None,
            }
        )
        return result

    renormalized = released / retained_mass
    renorm_norm = float(torch.linalg.vector_norm(renormalized).item())
    released_norm = float(torch.linalg.vector_norm(released).item())
    result.update(
        {
            "renormalized_non_neutral_semantics": "counterfactual",
            "released_vs_renormalized_l2": float(
                torch.linalg.vector_norm(released - renormalized).item()
            ),
            "released_to_renormalized_norm_ratio": (
                None if renorm_norm == 0 else released_norm / renorm_norm
            ),
        }
    )
    return result


def _released_synthesis_contract(script: Path) -> dict[str, Any]:
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    emotions: tuple[str, ...] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EMOTIONS" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                break
            emotions = tuple(value)
            break
    if emotions is None:
        raise ContractError(f"could not parse EMOTIONS from {script}")

    neutral_references = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "p_neutral":
            neutral_references.append(f"string:{node.lineno}")
        elif isinstance(node, ast.Name) and node.id == "p_neutral":
            neutral_references.append(f"name:{node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr == "p_neutral":
            neutral_references.append(f"attribute:{node.lineno}")

    if emotions != RELEASED_MIXING_EMOTIONS:
        raise ContractError(
            "released mixed-emotion source changed: "
            f"expected {RELEASED_MIXING_EMOTIONS}, found {emotions}"
        )
    if neutral_references:
        raise ContractError(
            "released mixed-emotion source directly references p_neutral: "
            + ", ".join(neutral_references)
        )
    return {
        "script_emotions": list(emotions),
        "p_neutral_direct_references": neutral_references,
        "p_neutral_read_by_synthesis_script": bool(neutral_references),
    }


def audit_released_cosyvoice_neutral_mixing(repo: Path) -> dict[str, Any]:
    """Audit a fixed, source-pinned neutral-mass probe against released vectors."""

    synthesis_script = repo / "scripts" / "synthesize.py"
    source_contract = _released_synthesis_contract(synthesis_script)
    script_emotions = tuple(source_contract["script_emotions"])

    vectors: dict[str, torch.Tensor] = {}
    vector_files: dict[str, str] = {}
    for emotion in script_emotions:
        path = repo / "steering_vectors" / "cosyvoice2" / f"{emotion}_neutral_attn_output.pt"
        if not path.is_file():
            raise ContractError(f"released steering vector is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        try:
            vectors[emotion] = checkpoint["steering_vectors"]["attn_output"][17]
        except KeyError as exc:
            raise ContractError(f"released vector lacks attn_output layer 17: {path}") from exc
        vector_files[emotion] = sha256_file(path)

    comparison = compare_neutral_semantics(vectors, PROBE_DISTRIBUTION)
    return {
        "audit": "claim4_neutral_weight_mixing_v1",
        "claim_evidence": False,
        "verdict": "released_mixing_omits_neutral_mass_from_vector_composition",
        "source": {
            "synthesis_script": "scripts/synthesize.py",
            "synthesis_script_sha256": sha256_file(synthesis_script),
            **source_contract,
        },
        "probe": {
            "backbone": "cosyvoice2",
            "operation": "attn_output",
            "layer": 17,
            "vector_sha256": vector_files,
            **comparison,
        },
        "limitations": [
            "This is a CPU release-conformance audit, not a mixed-emotion TTS quality result.",
            "The fixed probe demonstrates algebraic semantics; empirical effect size requires generated speech and frozen evaluators.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit released CoCoEmo neutral-mass mixing")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_released_cosyvoice_neutral_mixing(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
