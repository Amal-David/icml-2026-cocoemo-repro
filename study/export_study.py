"""Export immutable private study records to a private, analysis-ready CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from study.protocol import ProtocolError, assert_private_directory, load_manifest, manifest_hash
from study.storage import load_submissions


def _retention(records: list[Mapping[str, Any]]) -> dict[str, bool]:
    """Select the earliest valid ten per group; priority resolves timestamp ties."""

    selected: dict[str, bool] = {}
    for group in range(3):
        candidates = [
            row
            for row in records
            if row.get("group") == group and row.get("valid_under_preregistered_exclusions") is True
        ]
        candidates.sort(key=lambda row: (str(row.get("server_timestamp", "")), int(row.get("priority", -1))))
        for row in candidates[:10]:
            selected[str(row["study_id_hmac"])] = True
    return selected


def _manifest_rows(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in manifest["experimental_stimuli"]:
        rows[(str(row["item_id"]), str(row["condition"]))] = row
    return rows


def export(*, storage_root: Path, manifest: Mapping[str, Any], output: Path) -> dict[str, Any]:
    records = load_submissions(storage_root)
    frozen_hash = manifest_hash(manifest)
    item_ids = sorted({str(row["item_id"]) for row in manifest["experimental_stimuli"]})
    rows_by_item_condition = _manifest_rows(manifest)
    retained = _retention(records)
    output.parent.mkdir(mode=0o700, exist_ok=True)
    assert_private_directory(output.parent)
    fields = [
        "study_id_hmac",
        "server_timestamp",
        "group",
        "priority",
        "retained",
        "valid_under_preregistered_exclusions",
        "exclusion_reasons",
        "item_id",
        "track",
        "condition",
        "stimulus_id",
        "audio_sha256",
        "naturalness_score",
        "dominant_emotion",
        "angry",
        "happy",
        "sad",
        "surprised",
        "neutral",
        "target_angry",
        "target_happy",
        "target_sad",
        "target_surprised",
        "target_neutral",
    ]
    written = 0
    with output.open("x", encoding="utf-8", newline="") as handle:
        handle.chmod(0o600)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if record.get("protocol_version") != manifest["protocol_version"] or record.get("manifest_hash") != frozen_hash:
                raise ProtocolError("a submission does not match the frozen manifest and protocol")
            group = record.get("group")
            if group not in {0, 1, 2}:
                raise ProtocolError("a submission has an invalid assigned group")
            expected_trials = {f"experiment-{item_id}" for item_id in item_ids}
            response_by_trial = {row.get("trial_id"): row for row in record.get("trials", [])}
            if not expected_trials.issubset(response_by_trial):
                raise ProtocolError("a submission is missing an experimental response")
            for item_index, item_id in enumerate(item_ids):
                condition = ("alpha0", "instruction", "cocoemo")[(group + (item_index % 3)) % 3]
                manifest_row = rows_by_item_condition[(item_id, condition)]
                response = response_by_trial[f"experiment-{item_id}"]
                if response.get("audio_sha256") != manifest_row["audio_sha256"]:
                    raise ProtocolError("response audio does not match the assigned manifest row")
                allocation = response["emotion_allocation"]
                target = manifest_row["target_allocation"]
                writer.writerow(
                    {
                        "study_id_hmac": record["study_id_hmac"],
                        "server_timestamp": record["server_timestamp"],
                        "group": group,
                        "priority": record["priority"],
                        "retained": str(record["study_id_hmac"]) in retained,
                        "valid_under_preregistered_exclusions": record["valid_under_preregistered_exclusions"],
                        "exclusion_reasons": json.dumps(record["exclusion_reasons"], separators=(",", ":")),
                        "item_id": item_id,
                        "track": manifest_row["track"],
                        "condition": condition,
                        "stimulus_id": manifest_row["stimulus_id"],
                        "audio_sha256": manifest_row["audio_sha256"],
                        "naturalness_score": response["naturalness_score"],
                        "dominant_emotion": response["dominant_emotion"],
                        **allocation,
                        **{f"target_{emotion}": target[emotion] for emotion in ("angry", "happy", "sad", "surprised", "neutral")},
                    }
                )
                written += 1
    output.chmod(0o600)
    by_group = Counter(record["group"] for record in records if retained.get(str(record.get("study_id_hmac"))))
    return {
        "manifest_hash": frozen_hash,
        "submissions_total": len(records),
        "retained_by_group": dict(by_group),
        "experimental_rows_written": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_private_directory(args.storage_root)
    manifest = load_manifest(args.manifest)
    print(json.dumps(export(storage_root=args.storage_root, manifest=manifest, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
