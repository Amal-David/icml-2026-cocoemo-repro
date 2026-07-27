from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from repro.config import ReproConfig, load_config
from repro.contracts import ArtifactLedger, ContractError
from repro.provenance import collect_provenance
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


def run_stage(config: ReproConfig) -> dict[str, Any]:
    if config.stage == "preflight":
        return preflight(config)
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
        f"## Limitations\n\n{limitations}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict CoCoEmo reproduction runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    provenance = collect_provenance(repo=REPO, config_path=config.source, config_digest=config.digest)
    commit = provenance["repository"]["commit"][:12]
    name = re.sub(r"[^a-z0-9]+", "-", config.run_name.lower()).strip("-")
    run_root = REPO / ".openresearch" / "artifacts" / f"{commit}-{config.digest[:12]}-{name}"
    completed = run_root / "COMPLETE.json"
    if completed.exists():
        raise ContractError(f"completed run is immutable: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    summary = run_stage(config)
    write_json(run_root / "resolved_config.json", config.raw)
    write_json(run_root / "provenance.json", provenance)
    write_json(run_root / "summary.json", summary)
    eval_text = render_eval(config, provenance, summary)
    (run_root / "EVAL.md").write_text(eval_text, encoding="utf-8")
    (REPO / "EVAL.md").write_text(eval_text, encoding="utf-8")

    ledger = ArtifactLedger(run_root)
    for filename, kind in (
        ("resolved_config.json", "config"),
        ("provenance.json", "provenance"),
        ("summary.json", "metrics"),
        ("EVAL.md", "report"),
    ):
        ledger.add(run_root / filename, kind=kind)
    manifest_path = ledger.write()
    write_json(
        completed,
        {
            "status": "complete",
            "artifact_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    print(eval_text)


if __name__ == "__main__":
    main()
