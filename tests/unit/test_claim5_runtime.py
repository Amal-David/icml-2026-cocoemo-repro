from __future__ import annotations

import sys
import types
import hashlib
from pathlib import Path

import numpy as np
import pytest

from repro.claim5_runtime import flow_side_emotional_reference, native_instruction, native_no_steer
from repro.contracts import ContractError


class _Audio:
    def numel(self) -> int:
        return 1


def _install_fake_cosyvoice(monkeypatch) -> None:
    module = types.ModuleType("cosyvoice")
    utils = types.ModuleType("cosyvoice.utils")
    files = types.ModuleType("cosyvoice.utils.file_utils")
    files.load_wav = lambda path, rate: (path, rate)
    monkeypatch.setitem(sys.modules, "cosyvoice", module)
    monkeypatch.setitem(sys.modules, "cosyvoice.utils", utils)
    monkeypatch.setitem(sys.modules, "cosyvoice.utils.file_utils", files)


def test_native_arms_call_only_their_direct_cosyvoice_apis(monkeypatch, tmp_path) -> None:
    _install_fake_cosyvoice(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []

    class Model:
        def inference_zero_shot(self, **kwargs):
            calls.append(("zero", kwargs))
            yield {"tts_speech": _Audio()}

        def inference_instruct2(self, **kwargs):
            calls.append(("instruct", kwargs))
            yield {"tts_speech": _Audio()}

    model = Model()
    assert native_no_steer(model, text="x", reference_audio=tmp_path / "ref.wav", prompt_text="prompt").numel() == 1
    assert native_instruction(model, text="x", reference_audio=tmp_path / "ref.wav", instruction="Say it in a surprised tone.").numel() == 1
    assert [name for name, _ in calls] == ["zero", "instruct"]
    assert calls[0][1]["prompt_text"] == "prompt"
    assert calls[1][1]["instruct_text"] == "Say it in a surprised tone."
    assert "steering_vectors" not in calls[0][1]
    assert "steering_vectors" not in calls[1][1]


def test_flow_side_control_reuses_claim1_frontend_merge_and_trace(monkeypatch, tmp_path) -> None:
    import repro.claim5_runtime as runtime

    calls: list[object] = []
    monkeypatch.setattr(runtime, "_prepare_frontend_input", lambda model, **kwargs: {"reference": kwargs["reference_audio"]})

    def fake_merge(neutral, emotional, **kwargs):
        calls.append((neutral, emotional, kwargs))
        return {"merged": True}

    monkeypatch.setattr(runtime, "merge_cross_conditioned_inputs", fake_merge)
    monkeypatch.setattr(runtime, "synthesize_cross_conditioned_with_llm_trace", lambda model, inputs: ("audio", 1))

    class Model:
        model = object()

    assert flow_side_emotional_reference(Model(), text="target", neutral_reference=tmp_path / "n.wav", emotional_reference=tmp_path / "e.wav", prompt_text="prompt") == ("audio", 1)
    assert calls[0][2]["condition"] == "flow_driven"


def test_run_all_routes_claim5_canary(monkeypatch, tmp_path) -> None:
    from repro.config import load_config
    from repro.run_all import run_stage
    import repro.claim5_full as claim5_full

    config = load_config(Path(__file__).resolve().parents[2] / "configs/claim5-ravdess-semantic-conflict-canary.yaml")
    expected = {"stage": "claim5", "verdict": "not_evaluated"}
    monkeypatch.setattr(claim5_full, "run_claim5_runtime", lambda config, repo, run_root: expected)
    assert run_stage(config, run_root=tmp_path) == expected


def _runtime_groups(tmp_path: Path) -> list[dict[str, object]]:
    from repro.claim5_protocol import TARGETS

    groups: list[dict[str, object]] = []
    for actor in range(1, 25):
        for statement in ("01", "02"):
            for repetition in ("01", "02"):
                references: dict[str, dict[str, str]] = {}
                for emotion in ("neutral", *TARGETS):
                    path = tmp_path / f"{actor:02d}-{statement}-{repetition}-{emotion}.wav"
                    path.write_bytes(path.name.encode("ascii"))
                    references[emotion] = {"path": str(path)}
                groups.append({
                    "group_id": f"actor_{actor:02d}_statement_{statement}_repetition_{repetition}",
                    "references": references,
                })
    return groups


def test_mocked_canary_renders_32_wavs_and_uses_full_reference_bank(monkeypatch, tmp_path) -> None:
    import torch
    import cocoemo.backbones as backbones
    import repro.claim5_full as full
    import repro.gpu_baseline as baseline
    from repro.claim5_protocol import TARGETS
    from repro.config import load_config

    config = load_config(Path(__file__).resolve().parents[2] / "configs/claim5-ravdess-semantic-conflict-canary.yaml")
    groups = _runtime_groups(tmp_path)
    renders: list[tuple[str, int]] = []

    class FakeCosyVoice:
        sample_rate = 24000

        class model:
            @staticmethod
            def eval() -> None:
                return None

    class FakeEvaluator:
        def emotion_embedding(self, path: Path) -> np.ndarray:
            return np.array([1.0, 0.5, 0.25], dtype=np.float32)

        def evaluate(self, **kwargs: object) -> dict[str, object]:
            return {
                "emotion_probabilities": {target: 0.2 for target in (*TARGETS, "neutral")},
                "wavlm_speaker_similarity": 0.8,
                "whisper_hypothesis": "mock transcript",
                "whisper_wer": 0.0,
            }

    def fake_render(**kwargs: object) -> dict[str, object]:
        output = kwargs["output_path"]
        assert isinstance(output, Path)
        rate = kwargs["sample_rate"]
        assert rate == 24000
        output.write_bytes(output.name.encode("utf-8"))
        renders.append((output.name, rate))
        return {
            "path": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sample_rate": rate, "frames": 100, "duration_seconds": 1.0, "rms": 0.1, "clipping_fraction": 0.0,
        }

    fake_adapter = types.SimpleNamespace(load_model=lambda _: FakeCosyVoice())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(full, "ensure_ravdess_dataset", lambda **kwargs: tmp_path)
    monkeypatch.setattr(full, "build_claim1_groups", lambda _: groups)
    monkeypatch.setattr(full, "load_claim5_vectors", lambda **kwargs: {
        target: {17: np.ones((1, 896), dtype=np.float32), 14: np.ones((1, 896), dtype=np.float32)}
        for target in TARGETS
    })
    monkeypatch.setattr(full, "load_frozen_claim4_evaluators", lambda _: FakeEvaluator())
    monkeypatch.setattr(full, "render_claim5_arm", fake_render)
    monkeypatch.setattr(full, "reset_claim5_generation_seed", lambda _: None)
    monkeypatch.setattr(baseline, "_prepare_cosyvoice", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(baseline, "_prepare_model", lambda *args, **kwargs: tmp_path)
    monkeypatch.setitem(sys.modules, "cocoemo.backbones.cosyvoice2", fake_adapter)
    monkeypatch.setattr(backbones, "cosyvoice2", fake_adapter, raising=False)

    summary = full.run_claim5_runtime(config, repo=Path(__file__).resolve().parents[2], run_root=tmp_path / "run")
    assert len(renders) == 32
    assert summary["claim_evidence"] is False
    assert summary["verdict"] == "not_evaluated"
    assert summary["reference_embedding_bank"] == {
        "source_cells": 96, "source_embeddings": 96,
        "purpose": "frozen full 24-actor leave-one-actor-out prototype bank",
    }


def test_loaded_and_generated_sample_rates_must_match_the_frozen_contract() -> None:
    from repro.claim5_full import _require_generated_sample_rate, _require_sample_rate

    assert _require_sample_rate(configured_rate=24000, cosyvoice=types.SimpleNamespace(sample_rate=24000)) == 24000
    with pytest.raises(ContractError, match="differs"):
        _require_sample_rate(configured_rate=24000, cosyvoice=types.SimpleNamespace(sample_rate=22050))
    _require_generated_sample_rate(audio={"sample_rate": 24000}, expected_rate=24000)
    with pytest.raises(ContractError, match="generated WAV sample rate differs"):
        _require_generated_sample_rate(audio={"sample_rate": 22050}, expected_rate=24000)


def test_runtime_refuses_a_nonempty_incomplete_run_root_before_loading_models(tmp_path) -> None:
    import repro.claim5_full as full
    from repro.config import load_config

    config = load_config(Path(__file__).resolve().parents[2] / "configs/claim5-ravdess-semantic-conflict-canary.yaml")
    partial = tmp_path / "partial-run"
    partial.mkdir()
    (partial / "claim5_generation.jsonl").write_text('{"stale": true}\n', encoding="utf-8")
    with pytest.raises(ContractError, match="refusing to append stale"):
        full.run_claim5_runtime(config, repo=Path(__file__).resolve().parents[2], run_root=partial)
