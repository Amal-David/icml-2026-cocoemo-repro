from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download

from repro.config import ReproConfig
from repro.contracts import ContractError, require_complete_counts, sha256_file


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _prepare_cosyvoice(repo: Path, settings: dict[str, Any]) -> Path:
    revision = str(settings["source_revision"])
    target = repo / ".cache" / "cosyvoice" / revision
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", str(settings["source_url"]), str(target)])
    _run(["git", "fetch", "origin", revision], cwd=target)
    _run(["git", "checkout", "--detach", revision], cwd=target)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=target)
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=target, check=True, capture_output=True, text=True
    ).stdout.strip()
    if actual != revision:
        raise ContractError(f"CosyVoice revision mismatch: expected={revision}, actual={actual}")
    return target


def _prepare_model(repo: Path, settings: dict[str, Any]) -> Path:
    revision = str(settings["model_revision"])
    target = repo / ".cache" / "models" / f"cosyvoice2-{revision}"
    snapshot_download(
        repo_id=str(settings["model_id"]),
        revision=revision,
        local_dir=target,
    )
    required = ["cosyvoice2.yaml", "llm.pt", "flow.pt", "hift.pt", "speech_tokenizer_v2.onnx"]
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise ContractError(f"model snapshot is incomplete: {', '.join(missing)}")
    return target


def _audio_record(path: Path, *, alpha: float) -> dict[str, Any]:
    audio, sample_rate = torchaudio.load(path)
    if audio.numel() == 0 or sample_rate <= 0:
        raise ContractError(f"invalid audio output: {path}")
    duration = audio.shape[-1] / sample_rate
    rms = float(torch.sqrt(torch.mean(audio.float().square())).item())
    clipping_fraction = float((audio.abs() >= 0.999).float().mean().item())
    if duration <= 0.05 or rms <= 1e-5:
        raise ContractError(
            f"degenerate audio output: path={path}, duration={duration}, rms={rms}"
        )
    return {
        "alpha": alpha,
        "path": path.name,
        "sha256": sha256_file(path),
        "sample_rate": sample_rate,
        "frames": audio.shape[-1],
        "duration_seconds": duration,
        "rms": rms,
        "clipping_fraction": clipping_fraction,
    }


def run_gpu_baseline(config: ReproConfig, *, repo: Path, run_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ContractError("CosyVoice2 baseline requires CUDA; CPU execution is not claim evidence")
    settings = config.raw.get("baseline")
    if not isinstance(settings, dict):
        raise ContractError("config.baseline must be a mapping")

    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cosyvoice_root = _prepare_cosyvoice(repo, settings)
    model_dir = _prepare_model(repo, settings)
    os.environ["COSYVOICE_ROOT"] = str(cosyvoice_root)

    import cocoemo.backbones.cosyvoice2 as cv2
    from cocoemo.steering import load_steering_vectors

    vector_path = repo / str(settings["steering_vector"])
    reference_audio = repo / str(settings["reference_audio"])
    for required in (vector_path, reference_audio):
        if not required.is_file():
            raise ContractError(f"baseline input does not exist: {required}")

    model = cv2.load_model(str(model_dir))
    vectors = load_steering_vectors(str(vector_path))
    alphas = [float(value) for value in settings["alphas"]]
    output_dir = run_root / "synthesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for alpha in alphas:
        output_path = output_dir / f"alpha-{alpha:g}.wav"
        cv2.generate_steered_speech(
            model=model,
            text=str(settings["text"]),
            reference_audio_path=str(reference_audio),
            steering_vectors=vectors,
            layers=[int(value) for value in settings["layers"]],
            alpha=alpha,
            operations=[str(settings["operation"])],
            prompt_text=str(settings.get("prompt_text", "")),
            output_path=str(output_path),
            verbose=True,
        )
        records.append(_audio_record(output_path, alpha=alpha))

    require_complete_counts(requested=len(alphas), generated=len(records), evaluated=len(records))
    if len({record["sha256"] for record in records}) != len(records):
        raise ContractError("steering conditions produced byte-identical WAV outputs")

    per_sample = run_root / "per_sample.jsonl"
    per_sample.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    return {
        "stage": "baseline",
        "claim_evidence": False,
        "verdict": "runtime_baseline_pass",
        "requested": len(alphas),
        "generated": len(records),
        "evaluated": len(records),
        "records": records,
        "source_revision": settings["source_revision"],
        "model_id": settings["model_id"],
        "model_revision": settings["model_revision"],
        "steering_vector_sha256": sha256_file(vector_path),
        "implementation_operator": "translation_op_",
        "limitations": [
            "This run proves released CosyVoice2 steering executes; it does not verify a paper metric.",
            "The released implementation uses translation_op_ rather than paper Equation 8 norm preservation.",
        ],
    }
