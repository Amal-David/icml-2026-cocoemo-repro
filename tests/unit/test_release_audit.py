from pathlib import Path

from repro.release_audit import audit_release


REPO = Path(__file__).resolve().parents[2]


def test_released_cosyvoice_vectors_match_backbone_dimension() -> None:
    report = audit_release(REPO)
    cosy = [item for item in report["vectors"] if "/cosyvoice2/" in item["path"]]

    assert len(cosy) == 4
    assert all(item["observed_hidden_dims"] == [896] for item in cosy)
    assert all(item["compatible"] for item in cosy)


def test_release_audit_exposes_indextts2_contract_defects() -> None:
    report = audit_release(REPO)
    index = [item for item in report["vectors"] if "/indextts2/" in item["path"]]

    assert len(index) == 4
    assert all(item["expected_hidden_dim"] == 1024 for item in index)
    assert all(item["observed_hidden_dims"] == [1280] for item in index)
    assert all(not item["compatible"] for item in index)
    assert "released_vector_hidden_dimension_mismatch" in report["defects"]
    assert "indextts2_extraction_helper_mismatch" in report["defects"]


def test_release_audit_exposes_cosyvoice_operator_deviation() -> None:
    report = audit_release(REPO)

    assert report["cosyvoice_injection"]["paper_operator"] == "norm_preserve_steer_op_"
    assert report["cosyvoice_injection"]["released_active_operator"] == "translation_op_"
    assert not report["cosyvoice_injection"]["matches_paper"]
    assert "cosyvoice_injection_operator_deviation" in report["defects"]
