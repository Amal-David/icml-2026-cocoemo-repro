from __future__ import annotations

import platform
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from repro.contracts import sha256_file


UPSTREAM_COMMIT = "dcc319148dd2e0e0f3039e06e81e0f770d108f08"
UPSTREAM_URL = "https://github.com/wsssy/CoCoEmo"


_EXECUTION_ENVIRONMENT = {
    "hf_job_id": "HF_JOB_ID",
    "hf_job_url": "HF_JOB_URL",
    "orx_run_id": "ORX_RUN_ID",
    "orx_project_id": "ORX_PROJECT_ID",
    "orx_node_id": "ORX_NODE_ID",
    "orx_job_id": "ORX_JOB_ID",
    "backend": "REPRO_BACKEND",
    "hardware": "REPRO_HARDWARE",
    "image": "REPRO_IMAGE",
    "hf_job_hardware": "HF_JOB_HARDWARE",
    "hf_job_image": "HF_JOB_IMAGE",
    "orx_hardware": "ORX_HARDWARE",
    "orx_image": "ORX_IMAGE",
}


def _safe_execution_value(value: str | None, *, is_url: bool = False) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value or len(value) > 512:
        return None
    # Job identifiers, hardware names, and image tags never need an auth token.
    # Drop anything that resembles one rather than risking provenance leakage.
    lowered = value.lower()
    if any(marker in lowered for marker in ("hf_", "bearer ", "token=", "apikey", "api_key", "secret")):
        return None
    if is_url:
        if not value.startswith("https://") or "@" in value.split("//", 1)[1].split("/", 1)[0]:
            return None
        value = value.split("?", 1)[0].split("#", 1)[0]
    return value


def execution_context() -> dict[str, Any]:
    """Capture an allow-listed job context without reading secret variables."""

    values = {
        key: _safe_execution_value(
            os.environ.get(environment), is_url=key.endswith("_url")
        )
        for key, environment in _EXECUTION_ENVIRONMENT.items()
    }
    if not values["backend"]:
        values["backend"] = (
            "huggingface_jobs" if values["hf_job_id"] else "openresearch" if values["orx_run_id"] else "local"
        )
    values["hardware"] = (
        values["hardware"]
        or values["hf_job_hardware"]
        or values["orx_hardware"]
        or _safe_execution_value(os.environ.get("CUDA_VISIBLE_DEVICES"))
        or "cpu"
    )
    values["image"] = (
        values["image"] or values["hf_job_image"] or values["orx_image"] or None
    )
    return values


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    # Do not persist porcelain lines: they reveal local filenames (which can
    # include licensed-data locations).  The digest still binds the observed
    # state for diagnostics without exposing those paths.
    status_lines = status.splitlines()
    state_bytes = status.encode("utf-8") + b"\0" + diff
    return {
        "dirty": bool(status_lines),
        "dirty_file_count": len(status_lines),
        "dirty_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
    }


def require_clean_worktree(repo: Path) -> None:
    state = git_state(repo)
    if state["dirty"]:
        raise RuntimeError(
            "non-preflight evidence runs require a clean git worktree; "
            f"observed {state['dirty_file_count']} changed path(s)"
        )


def collect_provenance(*, repo: Path, config_path: Path, config_digest: str) -> dict[str, Any]:
    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_info = {"version": None, "cuda_available": False, "cuda_version": None, "gpu": None}

    return {
        "repository": {
            "url": "https://github.com/Amal-David/icml-2026-cocoemo-repro",
            "commit": git_commit(repo),
            **git_state(repo),
        },
        "upstream": {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT},
        "config": {
            "path": config_path.relative_to(repo).as_posix(),
            "sha256": config_digest,
            "file_sha256": sha256_file(config_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch_info,
        },
        "execution": execution_context(),
    }
