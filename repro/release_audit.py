from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import torch

from repro.contracts import sha256_file
from repro.claim4_mixing_audit import audit_released_cosyvoice_neutral_mixing


EXPECTED_VECTOR_DIMS = {"cosyvoice2": 896, "indextts2": 1024}


def _vector_shapes(checkpoint: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    for operation, layers in checkpoint["steering_vectors"].items():
        result[operation] = {str(layer): list(tensor.shape) for layer, tensor in layers.items()}
    return result


def audit_vector(path: Path, *, backbone: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    shapes = _vector_shapes(checkpoint)
    expected_dim = EXPECTED_VECTOR_DIMS[backbone]
    dimensions = sorted({shape[-1] for layers in shapes.values() for shape in layers.values()})
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "expected_hidden_dim": expected_dim,
        "observed_hidden_dims": dimensions,
        "compatible": dimensions == [expected_dim],
        "recommended_layers": checkpoint.get("recommended_layers", {}),
        "metadata": checkpoint.get("metadata", {}),
        "shapes": shapes,
    }


def imported_names(path: Path, module_suffix: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith(module_suffix):
            names.update(alias.name for alias in node.names)
    return names


def function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                break
            return segment
    raise ValueError(f"function not found: {function_name} in {path}")


def audit_release(repo: Path) -> dict[str, Any]:
    vectors = []
    for backbone in EXPECTED_VECTOR_DIMS:
        for path in sorted((repo / "steering_vectors" / backbone).glob("*.pt")):
            report = audit_vector(path, backbone=backbone)
            report["path"] = path.relative_to(repo).as_posix()
            vectors.append(report)

    helper_reports = {}
    for relative in ("scripts/extract.py", "scripts/discriminability.py"):
        path = repo / relative
        names = imported_names(path, "_core_indextts")
        helper_reports[relative] = {
            "imported": sorted(names),
            "expected_helper": "extract_with_hooks_audio_indextts2",
            "compatible": "extract_with_hooks_audio_indextts2" in names,
        }

    injection_path = repo / "cocoemo" / "steering" / "_core_cosyvoice.py"
    injection_source = function_source(injection_path, "prepare_steering_injection_config")
    injection = {
        "paper_operator": "norm_preserve_steer_op_",
        "released_active_operator": "translation_op_"
        if "'inject_op': translation_op_" in injection_source
        else "unknown",
    }
    injection["matches_paper"] = injection["released_active_operator"] == injection["paper_operator"]

    defects = []
    if any(not report["compatible"] for report in vectors):
        defects.append("released_vector_hidden_dimension_mismatch")
    if any(not report["compatible"] for report in helper_reports.values()):
        defects.append("indextts2_extraction_helper_mismatch")
    if not injection["matches_paper"]:
        defects.append("cosyvoice_injection_operator_deviation")

    mixing = audit_released_cosyvoice_neutral_mixing(repo)
    if mixing["verdict"] == "released_mixing_omits_neutral_mass_from_vector_composition":
        defects.append("claim4_neutral_weight_omitted")

    return {
        "vectors": vectors,
        "indextts2_helpers": helper_reports,
        "cosyvoice_injection": injection,
        "claim4_neutral_mixing": mixing,
        "defects": defects,
    }
