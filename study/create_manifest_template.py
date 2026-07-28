"""Create a non-runnable 72+6 frozen-manifest template without media files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from study.protocol import CONDITIONS, EMOTIONS, PROTOCOL_VERSION


_ZERO = "0" * 64
_PROVENANCE = {
    "backbone": "REPLACE_WITH_PINNED_BACKBONE",
    "model_revision": "REPLACE_WITH_IMMUTABLE_REVISION",
    "vector_id": "REPLACE_WITH_VECTOR_ID",
    "vector_sha256": _ZERO,
    "reference_voice_id": "REPLACE_WITH_NON_IDENTIFYING_CONSENTED_VOICE_ID",
    "reference_voice_release_id": "REPLACE_WITH_PRIVATE_RELEASE_RECEIPT_ID",
    "generator_commit": "REPLACE_WITH_40_CHAR_GIT_COMMIT",
    "license_basis": "written consent for research redistribution",
}


def template_manifest() -> dict[str, object]:
    experimental = []
    for track, alpha, index_offset in (("mixed", 5, 0), ("mismatch", 6, 12)):
        for index in range(12):
            item_id = f"{track}-{index + 1:02d}"
            for condition in CONDITIONS:
                experimental.append(
                    {
                        "stimulus_id": f"{item_id}-{condition}",
                        "item_id": item_id,
                        "track": track,
                        "condition": condition,
                        "audio_path": f"REPLACE_WITH_PRIVATE_AUDIO/{item_id}-{condition}.wav",
                        "audio_sha256": _ZERO,
                        "duration_s": 1.0,
                        "transcript": "REPLACE_WITH_LICENSE_CLEARED_ORIGINAL_TEXT",
                        "target_allocation": {emotion: 20 for emotion in EMOTIONS},
                        "alpha": 0 if condition == "alpha0" else alpha,
                        "provenance": dict(_PROVENANCE),
                        "template_index": index_offset + index,
                    }
                )
    quality = []
    duplicate_sources = ("mixed-01", "mismatch-01")
    for index, kind in enumerate(("instruction", "instruction", "duplicate", "duplicate", "calibration", "calibration")):
        row: dict[str, object] = {
            "trial_id": f"quality-{index + 1:02d}",
            "kind": kind,
        }
        if kind == "duplicate":
            row["source_item_id"] = duplicate_sources[index - 2]
        else:
            row.update(
                {
                    "audio_path": f"REPLACE_WITH_PRIVATE_AUDIO/quality-{index + 1:02d}.wav",
                    "audio_sha256": _ZERO,
                    "duration_s": 1.0,
                    "expected_dominant_emotion": EMOTIONS[index % len(EMOTIONS)],
                    "provenance": dict(_PROVENANCE),
                }
            )
        quality.append(row)
    return {
        "template_notice": "Replace every REPLACE_WITH value and zero digest. This template contains no audio and must fail media verification.",
        "protocol_version": PROTOCOL_VERSION,
        "manifest_version": "REPLACE_WITH_FROZEN_TAG",
        "study_id": "REPLACE_WITH_STUDY_ID",
        "generated_at": "2026-01-01T00:00:00Z",
        "experimental_stimuli": experimental,
        "quality_trials": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("study/manifest.example.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(template_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
