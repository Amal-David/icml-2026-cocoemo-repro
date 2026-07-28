from __future__ import annotations

import pytest

from study.analyse_study import analyse, load_complete_export
from study.protocol import ProtocolError


def _complete_rows():
    rows = []
    conditions = ("alpha0", "instruction", "cocoemo")
    for group in range(3):
        for rater in range(10):
            participant = f"p-{group}-{rater}"
            for index in range(24):
                track = "mixed" if index < 12 else "mismatch"
                condition = conditions[(group + (index % 3)) % 3]
                target = {"angry": 0, "happy": 100, "sad": 0, "surprised": 0, "neutral": 0}
                allocation = target if condition == "cocoemo" else {"angry": 20, "happy": 20, "sad": 20, "surprised": 20, "neutral": 20}
                rows.append(
                    {
                        "study_id_hmac": participant,
                        "group": str(group),
                        "retained": "True",
                        "item_id": f"{track}-{index % 12 + 1:02d}",
                        "track": track,
                        "condition": condition,
                        "stimulus_id": f"{track}-{index % 12 + 1:02d}-{condition}",
                        "naturalness_score": "4",
                        "dominant_emotion": "happy" if condition == "cocoemo" else "neutral",
                        **{key: str(value) for key, value in allocation.items()},
                        **{f"target_{key}": str(value) for key, value in target.items()},
                    }
                )
    return rows


def test_crossed_bootstrap_analysis_requires_and_handles_complete_design() -> None:
    rows = _complete_rows()
    result = analyse(rows, seed=1, replicates=1000)
    assert result["participants"] == 30
    assert result["experimental_rows"] == 720
    assert set(result["contrasts"]) == {
        "mixed_jsd_cocoemo_minus_alpha0_lower_is_better",
        "mismatch_target_allocation_cocoemo_minus_alpha0_higher_is_better",
        "mismatch_dominant_hit_cocoemo_minus_alpha0_higher_is_better",
        "naturalness_cocoemo_minus_instruction_noninferiority_margin_minus_0_35",
    }


def test_analysis_rejects_early_or_imbalanced_export(tmp_path) -> None:
    path = tmp_path / "trials.csv"
    path.write_text("study_id_hmac,group,retained\np,0,True\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="24 experimental"):
        load_complete_export(path)
