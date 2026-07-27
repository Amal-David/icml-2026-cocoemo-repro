from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

from repro.config import ReproConfig
from repro.contracts import ContractError, sha256_file
from repro.cremad_selection import load_audio_vote_rows, selection_variants, target_distribution


def _download_verified(*, url: str, expected_sha256: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or sha256_file(target) != expected_sha256:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        target.write_bytes(payload)
    actual = sha256_file(target)
    if actual != expected_sha256:
        raise ContractError(f"download hash mismatch: expected={expected_sha256}, actual={actual}")
    return target


def run_claim4_metadata(config: ReproConfig, *, repo: Path, run_root: Path) -> dict[str, Any]:
    settings = config.raw.get("claim4_metadata")
    if not isinstance(settings, dict):
        raise ContractError("config.claim4_metadata must be a mapping")
    source = _download_verified(
        url=str(settings["votes_url"]),
        expected_sha256=str(settings["votes_sha256"]),
        target=repo / ".cache" / "cremad" / str(settings["source_revision"]) / "tabulatedVotes.csv",
    )
    rows = load_audio_vote_rows(source)
    variants = selection_variants(rows)
    counts = {name: len(items) for name, items in variants.items()}
    expected = {name: int(value) for name, value in settings["expected_counts"].items()}
    if counts != expected:
        raise ContractError(f"CREMA-D selection counts changed: expected={expected}, actual={counts}")

    audit = {
        "source_url": settings["votes_url"],
        "source_revision": settings["source_revision"],
        "source_sha256": sha256_file(source),
        "paper_reported_count": int(settings["paper_reported_count"]),
        "selection_counts": counts,
        "paper_count_reproduced": int(settings["paper_reported_count"]) in counts.values(),
    }
    (run_root / "selection_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_root / "selection_variants.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "count"])
        writer.writerows(sorted(counts.items()))

    proxy_rows = variants["closest_supported_disagreement"]
    with (run_root / "cremad_audio_vote_proxy_v1.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "source_vote_row_id",
            "sample_id",
            "provided_label",
            "majority_vote",
            "n_raters",
            "count_angry",
            "count_happy",
            "count_neutral",
            "count_sad",
            "p_angry",
            "p_happy",
            "p_sad",
            "p_surprise",
            "p_neutral",
            "selection_variant",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in proxy_rows:
            writer.writerow(
                {
                    "source_vote_row_id": row.row_id,
                    "sample_id": row.file_name,
                    "provided_label": row.provided_label,
                    "majority_vote": ":".join(row.majority_labels),
                    "n_raters": row.num_responses,
                    "count_angry": row.counts["A"],
                    "count_happy": row.counts["H"],
                    "count_neutral": row.counts["N"],
                    "count_sad": row.counts["S"],
                    **target_distribution(row),
                    "selection_variant": "closest_supported_disagreement",
                }
            )

    return {
        "stage": "claim4",
        "claim_evidence": False,
        "verdict": "paper_selection_count_not_reproducible",
        "paper_reported_count": int(settings["paper_reported_count"]),
        "selection_counts": counts,
        "source_revision": settings["source_revision"],
        "source_sha256": sha256_file(source),
        "limitations": [
            "This CPU audit verifies the released selection protocol, not mixed-emotion TTS quality.",
            "The paper does not release its 772-item manifest or filtering code.",
            "CREMA-D has no surprise category; proxy p_surprise is fixed to zero.",
        ],
    }
