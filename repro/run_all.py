from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from repro.config import ReproConfig, load_config
from repro.contracts import ContractError, sha256_file
from repro.hub_artifacts import (
    artifact_destination,
    persist_run_artifacts,
    preflight_remote_write_permission,
    require_hf_token,
    safe_upload_error,
    write_public_artifact_manifest,
)
from repro.provenance import collect_provenance, require_clean_worktree
from repro.release_audit import audit_release


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight(config: ReproConfig) -> dict[str, Any]:
    required = [
        REPO / "cocoemo" / "steering" / "__init__.py",
        REPO / "scripts" / "synthesize.py",
        REPO / "scripts" / "evaluate.py",
        REPO / "steering_vectors" / "cosyvoice2" / "angry_neutral_attn_output.pt",
    ]
    missing = [path.relative_to(REPO).as_posix() for path in required if not path.is_file()]
    if missing:
        raise ContractError(f"release preflight is missing files: {', '.join(missing)}")
    release_audit = audit_release(REPO)
    return {
        "stage": "preflight",
        "claim_evidence": False,
        "verdict": "not_evaluated",
        "checks": {"required_release_files": "pass", "missing": []},
        "release_audit": release_audit,
        "limitations": [
            "No synthesis or empirical claim was evaluated by this preflight.",
            "Exact ESD and IEMOCAP reproductions require separately licensed data.",
            "Naturalness requires ratings from blinded human listeners.",
        ],
    }


def run_stage(config: ReproConfig, *, run_root: Path) -> dict[str, Any]:
    if config.stage == "preflight":
        return preflight(config)
    if config.stage == "baseline":
        from repro.gpu_baseline import run_gpu_baseline

        return run_gpu_baseline(config, repo=REPO, run_root=run_root)
    if config.stage == "claim1":
        from repro.claim1_gpu import run_claim1_gpu

        return run_claim1_gpu(config, repo=REPO, run_root=run_root)
    if config.stage == "claim2" and (
        "claim23" in config.raw or "claim23_canary" in config.raw
    ):
        from repro.claim23_gpu import run_claim23_gpu

        return run_claim23_gpu(config, repo=REPO, run_root=run_root)
    if config.stage == "claim4" and "claim4_metadata" in config.raw:
        from repro.claim4_metadata import run_claim4_metadata

        return run_claim4_metadata(config, repo=REPO, run_root=run_root)
    if config.stage == "claim4" and "claim4_canary" in config.raw:
        from repro.claim4_runtime import run_claim4_runtime_canary

        return run_claim4_runtime_canary(config, repo=REPO, run_root=run_root)
    if config.stage == "claim4" and "claim4" in config.raw:
        from repro.claim4_full import run_claim4_runtime_full

        return run_claim4_runtime_full(config, repo=REPO, run_root=run_root)
    if config.stage == "claim5" and (
        "claim5" in config.raw or "claim5_canary" in config.raw
    ):
        from repro.claim5_full import run_claim5_runtime

        return run_claim5_runtime(config, repo=REPO, run_root=run_root)
    raise ContractError(
        f"stage {config.stage!r} has no implemented evidence runner; refusing to emit a claim result"
    )


def render_eval(config: ReproConfig, provenance: dict[str, Any], summary: dict[str, Any]) -> str:
    limitations = "\n".join(f"- {item}" for item in summary.get("limitations", []))
    return (
        f"# {config.run_name}\n\n"
        f"- Stage: `{config.stage}`\n"
        f"- Claims requested: `{list(config.claims)}`\n"
        f"- Claim evidence: `{summary['claim_evidence']}`\n"
        f"- Verdict: `{summary['verdict']}`\n"
        f"- Repository commit: `{provenance['repository']['commit']}`\n"
        f"- Config digest: `{config.digest}`\n\n"
        f"- Evidence destination: `{provenance['artifacts']['remote_prefix']}`\n"
        f"- Completion rule: accept evidence only when one atomic remote commit contains this report, its manifest, and `COMPLETE.json`.\n\n"
        f"## Limitations\n\n{limitations}\n"
    )


def _write_local_ready(
    path: Path, *, config: ReproConfig, provenance: dict[str, Any], artifact_manifest_sha256: str
) -> None:
    write_json(
        path,
        {
            "status": "local_ready_for_remote_persistence",
            "repository_commit": provenance["repository"]["commit"],
            "config_sha256": config.digest,
            "run_name": config.run_name,
            "artifact_manifest_sha256": artifact_manifest_sha256,
        },
    )


def _load_existing_local_result(
    *, run_root: Path, config: ReproConfig, provenance: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    ready_path = run_root / "LOCAL_READY.json"
    summary_path = run_root / "summary.json"
    provenance_path = run_root / "provenance.json"
    if not ready_path.is_file() or not summary_path.is_file() or not provenance_path.is_file():
        raise ContractError(
            "run root is nonempty without a complete local-ready record; refusing to mix retry state with a new run"
        )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    expected = {
        "repository_commit": provenance["repository"]["commit"],
        "config_sha256": config.digest,
        "run_name": config.run_name,
    }
    if any(ready.get(key) != value for key, value in expected.items()):
        raise ContractError("local-ready run identity does not match this immutable config and commit")
    return (
        json.loads(summary_path.read_text(encoding="utf-8")),
        json.loads(provenance_path.read_text(encoding="utf-8")),
    )


def _validate_frozen_artifacts(run_root: Path) -> None:
    """Reject retry state whose scientific output differs from its ledger."""

    manifest_path = run_root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ContractError("local-ready run is missing its frozen artifact manifest")
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContractError("local-ready artifact manifest is malformed") from exc
    by_path = {entry.get("path"): entry for entry in entries if isinstance(entry, dict)}
    required = {"summary.json", "provenance.json", "EVAL.md", "resolved_config.json"}
    if not required <= set(by_path):
        raise ContractError("local-ready artifact manifest does not bind all frozen scientific reports")
    try:
        ready = json.loads((run_root / "LOCAL_READY.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("local-ready record is malformed") from exc
    if ready.get("artifact_manifest_sha256") != sha256_file(manifest_path):
        raise ContractError("local-ready artifact manifest hash mismatch")
    for relative, entry in by_path.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ContractError("local-ready artifact manifest contains an unsafe path")
        path = run_root / relative
        expected = entry.get("sha256")
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise ContractError(f"local-ready artifact hash mismatch: {relative}")


def _append_persistence_attempt(run_root: Path, payload: dict[str, Any]) -> None:
    """Keep mutable remote-attempt telemetry separate from scientific output."""

    path = run_root / "PERSISTENCE_ATTEMPTS.json"
    if path.exists():
        try:
            attempts = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractError("persistence diagnostics are malformed") from exc
        if not isinstance(attempts, list):
            raise ContractError("persistence diagnostics must be a JSON list")
    else:
        attempts = []
    attempts.append({"at": dt.datetime.now(dt.timezone.utc).isoformat(), **payload})
    write_json(path, attempts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict CoCoEmo reproduction runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    # Fail before repository/model preparation for evidence stages. A preflight
    # remains useful without credentials but immediately persists when a Job
    # supplies its token.
    token = require_hf_token() if config.stage != "preflight" else os.environ.get("HF_TOKEN")
    provenance = collect_provenance(repo=REPO, config_path=config.source, config_digest=config.digest)
    destination = artifact_destination(
        commit=provenance["repository"]["commit"],
        config_digest=config.digest,
        run_name=config.run_name,
    )
    provenance["artifacts"] = destination.as_dict()
    # A claim runner can incur GPU cost before it emits its first artifact.  Do
    # not let it start when it has no way to durably preserve its evidence.
    commit = provenance["repository"]["commit"][:12]
    name = re.sub(r"[^a-z0-9]+", "-", config.run_name.lower()).strip("-")
    run_root = REPO / ".openresearch" / "artifacts" / f"{commit}-{config.digest[:12]}-{name}"
    completed = run_root / "COMPLETE.json"
    if completed.exists():
        raise ContractError(f"completed run is immutable: {run_root}")
    ready = run_root / "LOCAL_READY.json"
    permission_preflight: dict[str, Any] | None = None
    if config.stage != "preflight":
        require_clean_worktree(REPO)
        # Do this before synthesis/evaluation.  A retry has no expensive stage
        # to protect and instead verifies its frozen local artifacts below.
        if not ready.exists():
            permission_preflight = preflight_remote_write_permission(
                destination=destination, token=token
            )
    if ready.exists():
        summary, persisted_provenance = _load_existing_local_result(
            run_root=run_root, config=config, provenance=provenance
        )
        _validate_frozen_artifacts(run_root)
        # The result provenance, summary, and EVAL report stay immutable.  A
        # retry may only append non-scientific persistence diagnostics.
        provenance = persisted_provenance
    else:
        if run_root.exists() and any(run_root.iterdir()):
            raise ContractError(
                "run root is nonempty without LOCAL_READY.json; refusing to overwrite an interrupted run"
            )
        run_root.mkdir(parents=True, exist_ok=True)
        summary = run_stage(config, run_root=run_root)
        write_json(run_root / "resolved_config.json", config.raw)

    if not ready.exists():
        summary["artifact_persistence"] = {
            "status": "pending_atomic_hub_commit" if token else "local_ready_pending_hf_token",
            "destination": destination.as_dict(),
            "acceptance": "This summary remains pending until one atomic Hub commit contains its manifest and COMPLETE.json.",
        }
        provenance["artifacts"] = destination.as_dict()
        write_json(run_root / "provenance.json", provenance)
        write_json(run_root / "summary.json", summary)
        eval_text = render_eval(config, provenance, summary)
        (run_root / "EVAL.md").write_text(eval_text, encoding="utf-8")
        (REPO / "EVAL.md").write_text(eval_text, encoding="utf-8")
    else:
        eval_text = (run_root / "EVAL.md").read_text(encoding="utf-8")

    if not ready.exists():
        manifest_path = write_public_artifact_manifest(run_root)
        # This is intentionally the final local write before any remote call:
        # it certifies a complete, hash-accounted local evidence set for retry.
        _write_local_ready(
            ready,
            config=config,
            provenance=provenance,
            artifact_manifest_sha256=sha256_file(manifest_path),
        )

    if permission_preflight is not None:
        _append_persistence_attempt(run_root, {"kind": "write_permission_preflight", **permission_preflight})

    if token is None:
        print(eval_text)
        return

    try:
        remote_result = persist_run_artifacts(
            run_root=run_root, destination=destination, token=token
        )
    except Exception as exc:
        # Keep the local ready marker for a clean fresh-job retry, but never
        # mutate the frozen scientific reports or manufacture completion.
        _append_persistence_attempt(
            run_root,
            {"kind": "atomic_upload", "status": "failed", "error": safe_upload_error(exc)},
        )
        raise

    write_json(
        completed,
        {
            "status": "complete",
            "hub_artifact_manifest_sha256": remote_result["artifact_manifest_sha256"],
            "remote": remote_result,
        },
    )
    _append_persistence_attempt(
        run_root,
        {
            "kind": "atomic_upload",
            "status": "succeeded",
            "hub_commit_oid": remote_result["hub_commit_oid"],
            "hub_commit_url": remote_result["hub_commit_url"],
            "immutable_destination": remote_result["immutable_destination"],
        },
    )
    print(eval_text)


if __name__ == "__main__":
    main()
