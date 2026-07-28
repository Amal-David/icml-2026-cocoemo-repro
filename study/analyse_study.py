"""Preregistered balance checks and crossed participant/content bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping

from study.protocol import ProtocolError, assert_private_directory


def _jsd(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    def kl(first: Mapping[str, float], second: Mapping[str, float]) -> float:
        return sum(value * math.log2(value / second[key]) for key, value in first.items() if value)

    midpoint = {key: (left[key] + right[key]) / 2 for key in left}
    return (kl(left, midpoint) + kl(right, midpoint)) / 2


def _allocation(row: Mapping[str, str], prefix: str = "") -> dict[str, float]:
    return {emotion: float(row[f"{prefix}{emotion}"]) / 100 for emotion in ("angry", "happy", "sad", "surprised", "neutral")}


def _target_emotion(row: Mapping[str, str]) -> str:
    target = _allocation(row, "target_")
    highest = max(target.values())
    labels = [emotion for emotion, value in target.items() if value == highest]
    if len(labels) != 1:
        raise ProtocolError("mismatch target allocation must have one dominant target emotion")
    return labels[0]


def _contrast(rows: Iterable[Mapping[str, str]], metric: Callable[[Mapping[str, str]], float], *, track: str, baseline: str) -> float:
    values = defaultdict(dict)
    for row in rows:
        if row["track"] == track:
            values[row["item_id"]].setdefault(row["condition"], []).append(metric(row))
    deltas = [
        mean(cell["cocoemo"]) - mean(cell[baseline])
        for cell in values.values()
        if "cocoemo" in cell and baseline in cell
    ]
    if not deltas:
        raise ProtocolError(f"no paired {track} observations for cocoemo versus {baseline}")
    return mean(deltas)


def _crossed_bootstrap(
    rows: list[Mapping[str, str]],
    statistic: Callable[[Iterable[Mapping[str, str]]], float],
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float, float]:
    participants = sorted({row["study_id_hmac"] for row in rows})
    contents = sorted({row["item_id"] for row in rows})
    if not participants or not contents:
        raise ProtocolError("cannot bootstrap an empty analysis set")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        participant_weight = Counter(rng.choice(participants) for _ in participants)
        content_weight = Counter(rng.choice(contents) for _ in contents)
        weighted: list[Mapping[str, str]] = []
        for row in rows:
            copies = participant_weight[row["study_id_hmac"]] * content_weight[row["item_id"]]
            weighted.extend([row] * copies)
        samples.append(statistic(weighted))
    samples.sort()
    lower = samples[max(0, math.floor(0.025 * (replicates - 1)))]
    upper = samples[min(replicates - 1, math.ceil(0.975 * (replicates - 1)))]
    return statistic(rows), lower, upper


def load_complete_export(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retained = [row for row in rows if row.get("retained") == "True"]
    participants_by_group = Counter()
    for participant, group_rows in _group_by(retained, key=lambda row: (row["study_id_hmac"], row["group"])).items():
        participants_by_group[participant[1]] += 1
        if len(group_rows) != 24:
            raise ProtocolError("each retained participant must have 24 experimental rows")
    if participants_by_group != Counter({"0": 10, "1": 10, "2": 10}):
        raise ProtocolError("final analysis requires exactly ten retained participants per group")
    expected_cells = Counter((row["stimulus_id"] for row in retained))
    if len(expected_cells) != 72 or any(count != 10 for count in expected_cells.values()):
        raise ProtocolError("final analysis requires exactly ten ratings per frozen stimulus")
    return retained


def _group_by(rows: Iterable[Mapping[str, str]], key: Callable[[Mapping[str, str]], Any]):
    grouped: dict[Any, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def analyse(rows: list[Mapping[str, str]], *, seed: int, replicates: int) -> dict[str, Any]:
    if replicates < 1000:
        raise ProtocolError("at least 1000 bootstrap replicates are required")
    mixed_jsd = lambda row: _jsd(_allocation(row), _allocation(row, "target_"))
    mismatch_target = lambda row: _allocation(row).get(_target_emotion(row), 0.0)
    mismatch_hit = lambda row: float(row["dominant_emotion"] == _target_emotion(row))
    naturalness = lambda row: float(row["naturalness_score"])
    definitions = {
        "mixed_jsd_cocoemo_minus_alpha0_lower_is_better": lambda values: _contrast(values, mixed_jsd, track="mixed", baseline="alpha0"),
        "mismatch_target_allocation_cocoemo_minus_alpha0_higher_is_better": lambda values: _contrast(values, mismatch_target, track="mismatch", baseline="alpha0"),
        "mismatch_dominant_hit_cocoemo_minus_alpha0_higher_is_better": lambda values: _contrast(values, mismatch_hit, track="mismatch", baseline="alpha0"),
        "naturalness_cocoemo_minus_instruction_noninferiority_margin_minus_0_35": lambda values: _contrast(values, naturalness, track="mixed", baseline="instruction"),
    }
    return {
        "status": "analysis_only_no_interpretive_verdict",
        "participants": len({row["study_id_hmac"] for row in rows}),
        "experimental_rows": len(rows),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "contrasts": {
            label: {
                "estimate": estimate,
                "ci95": [lower, upper],
                "naturalness_noninferior_at_margin_minus_0_35": lower >= -0.35 if label.startswith("naturalness") else None,
            }
            for label, statistic in definitions.items()
            for estimate, lower, upper in [_crossed_bootstrap(rows, statistic, seed=seed, replicates=replicates)]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    assert_private_directory(args.trials_csv.parent)
    args.output.parent.mkdir(mode=0o700, exist_ok=True)
    assert_private_directory(args.output.parent)
    result = analyse(load_complete_export(args.trials_csv), seed=args.seed, replicates=args.replicates)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(json.dumps({"output": str(args.output), "participants": result["participants"]}, sort_keys=True))


if __name__ == "__main__":
    main()
