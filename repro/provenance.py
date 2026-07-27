from __future__ import annotations

import platform
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from repro.contracts import sha256_file


UPSTREAM_COMMIT = "dcc319148dd2e0e0f3039e06e81e0f770d108f08"
UPSTREAM_URL = "https://github.com/wsssy/CoCoEmo"


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
    return {
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


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
    }
