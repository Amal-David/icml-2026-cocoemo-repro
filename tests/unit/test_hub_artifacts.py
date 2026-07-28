from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from repro import run_all
from repro.hub_artifacts import (
    HubArtifactError,
    artifact_destination,
    collect_upload_candidates,
    persist_run_artifacts,
    preflight_remote_write_permission,
    require_hf_token,
    safe_upload_error,
    write_public_artifact_manifest,
)
from repro.provenance import execution_context


class RecordingHubApi:
    def __init__(
        self,
        *,
        fail_commit: bool = False,
        already_complete: bool = False,
        before_create=None,
        write_access: bool | None = None,
        raise_after_commit: bool = False,
    ) -> None:
        self.fail_commit = fail_commit
        self.already_complete = already_complete
        self.before_create = before_create
        self.write_access = write_access
        self.raise_after_commit = raise_after_commit
        self.file_exists_calls: list[dict[str, object]] = []
        self.commit_calls: list[dict[str, object]] = []
        self.head = "a" * 40
        self.remote_files: dict[str, bytes] = {}
        self.last_commit_by_path: dict[str, str] = {}

    def repo_info(self, **_kwargs: object) -> object:
        return SimpleNamespace(sha=self.head)

    def file_exists(self, **kwargs: object) -> bool:
        self.file_exists_calls.append(kwargs)
        filename = str(kwargs["filename"])
        return filename in self.remote_files or (
            self.already_complete and filename.endswith("/COMPLETE.json")
        )

    def get_paths_info(self, **kwargs: object) -> list[object]:
        path = str(kwargs["paths"][0])
        if path not in self.remote_files:
            return []
        return [SimpleNamespace(last_commit=SimpleNamespace(oid=self.last_commit_by_path[path]))]

    def read_file(self, **kwargs: object) -> bytes:
        return self.remote_files[str(kwargs["filename"])]

    def create_commit(self, **kwargs: object) -> object:
        if self.before_create:
            self.before_create()
            self.before_create = None
        self.commit_calls.append(kwargs)
        if self.fail_commit:
            header = "Author" + "ization: Bearer "
            raise RuntimeError(header + "hf_" + "0123456789abcdefghijklmnop")
        self.head = f"{len(self.commit_calls):040x}"
        for operation in kwargs["operations"]:
            if operation.__class__.__name__ == "CommitOperationDelete":
                self.remote_files.pop(operation.path_in_repo, None)
                self.last_commit_by_path.pop(operation.path_in_repo, None)
            else:
                self.remote_files[operation.path_in_repo] = operation.path_or_fileobj
                self.last_commit_by_path[operation.path_in_repo] = self.head
        if self.raise_after_commit:
            raise TimeoutError("simulated transport response loss")
        return SimpleNamespace(oid=self.head, commit_url=f"https://example.invalid/commit/{self.head}")

    def has_write_access(self, **_kwargs: object) -> bool | None:
        return self.write_access


def _write(root, name: str, value: bytes | str = "{}\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))
    return path


def _destination():
    return artifact_destination(commit="a" * 40, config_digest="b" * 64, run_name="Claim 1 Canary")


def _operation_paths(call: dict[str, object]) -> list[str]:
    return [operation.path_in_repo for operation in call["operations"]]


def test_collect_upload_candidates_uses_top_level_schema_and_filters_nested_sources(tmp_path) -> None:
    _write(tmp_path, "summary.json")
    _write(tmp_path, "claim1_per_sample.jsonl", "{\"score\": 1}\n")
    _write(tmp_path, "EVAL.md", "# evidence\n")
    _write(tmp_path, "figures/result.png", b"\x89PNG\r\n\x1a\nsmall")
    _write(tmp_path, "wavs/speech.wav", b"audio")
    _write(tmp_path, "cache/metrics.json")
    _write(tmp_path, "metrics/per_sample.jsonl", "{\"leak\": true}\n")
    _write(tmp_path, "credentials.json")
    _write(tmp_path, "notes.txt", "not evidence\n")

    assert [candidate.relative_path for candidate in collect_upload_candidates(tmp_path)] == [
        "EVAL.md",
        "claim1_per_sample.jsonl",
        "figures/result.png",
        "summary.json",
    ]


def test_collector_rejects_credentials_and_suspicious_binary_payloads(tmp_path) -> None:
    _write(tmp_path, "summary.json", '{"token": "not allowed"}\n')
    with pytest.raises(HubArtifactError, match="payload field"):
        collect_upload_candidates(tmp_path)


def test_public_ledger_never_exposes_nested_audio_or_cache_paths(tmp_path) -> None:
    _write(tmp_path, "summary.json")
    _write(tmp_path, "EVAL.md", "# report\n")
    _write(tmp_path, "wavs/private-speech.wav", b"audio")
    _write(tmp_path, "cache/private-model-state.json", "{}\n")
    manifest = write_public_artifact_manifest(tmp_path)
    body = manifest.read_text(encoding="utf-8")

    assert "private-speech.wav" not in body
    assert "private-model-state.json" not in body
    assert {entry["path"] for entry in json.loads(body)["artifacts"]} == {"EVAL.md", "summary.json"}


def test_collector_rejects_symlinked_figures_directory_and_escape(tmp_path) -> None:
    _write(tmp_path, "summary.json")
    outside = tmp_path.parent / "outside-figures"
    outside.mkdir(exist_ok=True)
    _write(outside, "result.png", b"\x89PNG\r\n\x1a\nsmall")
    (tmp_path / "figures").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HubArtifactError, match="symlinked figures directory"):
        collect_upload_candidates(tmp_path)


def test_collector_rejects_base64_like_json_payload(tmp_path) -> None:
    _write(tmp_path, "summary.json", '{"metric": "' + "A" * 256 + '"}\n')
    with pytest.raises(HubArtifactError, match="base64-like"):
        collect_upload_candidates(tmp_path)


def test_claim5_public_candidates_exclude_local_embeddings_and_keep_derived_reports(tmp_path) -> None:
    _write(tmp_path, "summary.json", '{"reference_embedding_bank": {"source_embeddings": 96}}\n')
    _write(tmp_path, "claim5_reference_embeddings.jsonl", '{"actor_id": "01", "embedding": [0.1, 0.2]}\n')
    _write(tmp_path, "claim5_reference_embedding_manifest.json", '{"source_embeddings": 96, "reference_embeddings_sha256": "a"}\n')
    _write(tmp_path, "claim5_per_sample.jsonl", '{"cell_id": "01", "target_emotion_probability": 0.5}\n')

    paths = [candidate.relative_path for candidate in collect_upload_candidates(tmp_path)]

    assert paths == [
        "claim5_per_sample.jsonl",
        "claim5_reference_embedding_manifest.json",
        "summary.json",
    ]
    manifest = write_public_artifact_manifest(tmp_path).read_text(encoding="utf-8")
    assert "claim5_reference_embeddings.jsonl" not in manifest
    assert '"embedding": [0.1, 0.2]' not in manifest


@pytest.mark.parametrize("raw_key", ["embedding", "embeddings", "activation", "activations", "tensor", "tensors"])
def test_collector_rejects_raw_representation_fields_in_eligible_reports(tmp_path, raw_key) -> None:
    _write(tmp_path, "claim5_per_sample.jsonl", json.dumps({raw_key: [0.1, 0.2]}) + "\n")

    with pytest.raises(HubArtifactError, match="raw representation field"):
        collect_upload_candidates(tmp_path)


def test_persistence_is_one_atomic_parent_guarded_commit_and_pins_revision(tmp_path) -> None:
    _write(tmp_path, "summary.json", '{"artifact_persistence": {"status": "pending_atomic_hub_commit"}}\n')
    _write(tmp_path, "EVAL.md", "# evidence\n")
    api = RecordingHubApi()

    result = persist_run_artifacts(
        run_root=tmp_path, destination=_destination(), token="hf_test_token_value", api_factory=lambda _: api
    )

    assert len(api.commit_calls) == 1
    call = api.commit_calls[0]
    assert call["parent_commit"] == "a" * 40
    paths = _operation_paths(call)
    assert paths[-1].endswith("/COMPLETE.json")
    assert any(path.endswith("/hub_artifact_manifest.json") for path in paths)
    assert result["status"] == "uploaded_atomically"
    assert result["hub_commit_oid"] == "0" * 39 + "1"
    assert result["immutable_destination"]["revision"] == result["hub_commit_oid"]
    assert result["immutable_destination"]["complete_url"].split("/")[7] == result["hub_commit_oid"]
    assert api.file_exists_calls[0]["revision"] == "a" * 40


def test_snapshot_survives_tamper_between_hash_and_commit(tmp_path) -> None:
    summary = _write(tmp_path, "summary.json", '{"verdict": "original"}\n')
    original = summary.read_bytes()
    api = RecordingHubApi(before_create=lambda: summary.write_text('{"verdict": "tampered"}\n'))

    persist_run_artifacts(
        run_root=tmp_path, destination=_destination(), token="hf_test_token_value", api_factory=lambda _: api
    )

    summary_operation = next(
        item for item in api.commit_calls[0]["operations"] if item.path_in_repo.endswith("/summary.json")
    )
    assert summary_operation.path_or_fileobj == original
    assert summary.read_bytes() != original


def test_existing_remote_complete_reconciles_only_an_exact_snapshot(tmp_path) -> None:
    _write(tmp_path, "summary.json")
    existing = RecordingHubApi()
    first = persist_run_artifacts(run_root=tmp_path, destination=_destination(), token="hf_test_token_value", api_factory=lambda _: existing)
    reconciled = persist_run_artifacts(run_root=tmp_path, destination=_destination(), token="hf_test_token_value", api_factory=lambda _: existing)
    assert reconciled["status"] == "reconciled_remote_atomic_commit"
    assert reconciled["hub_commit_oid"] == first["hub_commit_oid"]
    assert len(existing.commit_calls) == 1


def test_lost_response_reconciles_remote_atomic_commit_and_retry(tmp_path) -> None:
    _write(tmp_path, "summary.json", '{"verdict": "frozen"}\n')
    api = RecordingHubApi(raise_after_commit=True)
    result = persist_run_artifacts(run_root=tmp_path, destination=_destination(), token="hf_test_token_value", api_factory=lambda _: api)
    retry = persist_run_artifacts(run_root=tmp_path, destination=_destination(), token="hf_test_token_value", api_factory=lambda _: api)
    assert result["status"] == "reconciled_remote_atomic_commit"
    assert retry["status"] == "reconciled_remote_atomic_commit"
    assert len(api.commit_calls) == 1


def test_atomic_failure_does_not_create_partial_completion(tmp_path) -> None:
    _write(tmp_path, "summary.json")

    failed = RecordingHubApi(fail_commit=True)
    with pytest.raises(HubArtifactError, match="atomic evidence commit failed"):
        persist_run_artifacts(run_root=tmp_path, destination=_destination(), token="hf_test_token_value", api_factory=lambda _: failed)
    assert len(failed.commit_calls) == 1
    assert not (tmp_path / "COMPLETE.json").exists()
    fake_token = "hf_" + "0123456789abcdefghijklmnop"
    assert fake_token not in json.dumps(safe_upload_error(RuntimeError(f"HF_TOKEN={fake_token}")))


def test_permission_preflight_uses_safe_deterministic_canary_and_cleanup(tmp_path) -> None:
    del tmp_path
    api = RecordingHubApi()
    result = preflight_remote_write_permission(destination=_destination(), token="hf_test_token_value", api_factory=lambda _: api)

    assert result["method"] == "deterministic_commit_canary"
    assert len(api.commit_calls) == 2
    probe, cleanup = api.commit_calls
    assert _operation_paths(probe) == [result["probe_path"]]
    assert _operation_paths(cleanup) == [result["probe_path"]]
    assert cleanup["parent_commit"] == result["probe_commit_oid"]
    assert result["cleanup_commit_oid"] == "0" * 39 + "2"


def test_authoritative_permission_adapter_avoids_canary() -> None:
    api = RecordingHubApi(write_access=True)
    result = preflight_remote_write_permission(destination=_destination(), token="hf_test_token_value", api_factory=lambda _: api)
    assert result["method"] == "authoritative_non_mutating_check"
    assert not api.commit_calls


def test_non_preflight_token_gate_and_execution_context_do_not_serialize_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(HubArtifactError, match="HF_TOKEN is required"):
        require_hf_token()

    monkeypatch.setenv("HF_TOKEN", "test-token-value")
    monkeypatch.setenv("HF_JOB_ID", "job-123")
    monkeypatch.setenv("ORX_RUN_ID", "run-456")
    monkeypatch.setenv("REPRO_HARDWARE", "a10g-small")
    monkeypatch.setenv("REPRO_IMAGE", "python:3.11-cuda")
    serialized = json.dumps(execution_context())
    assert "test-token-value" not in serialized
    assert "job-123" in serialized


def test_runner_retry_preserves_frozen_reports_and_writes_only_attempt_diagnostics(tmp_path, monkeypatch) -> None:
    config_path = _write(tmp_path, "configs/preflight.yaml", "run:\n  name: persistence test\n  seed: 7\n  stage: preflight\nclaims: []\n")
    monkeypatch.setattr(run_all, "REPO", tmp_path)
    monkeypatch.setattr(run_all, "collect_provenance", lambda **_: {"repository": {"commit": "a" * 40}, "execution": {"backend": "local"}})
    monkeypatch.setattr(run_all, "run_stage", lambda *_args, **_kwargs: {"claim_evidence": False, "verdict": "not_evaluated", "limitations": []})
    monkeypatch.setenv("HF_TOKEN", "test-token-value")
    failures = [RuntimeError(f"Bearer {'hf_' + '0123456789abcdefghijklmnop'}"), {"hub_commit_oid": "c" * 40, "hub_commit_url": "https://example.invalid/commit/c", "immutable_destination": _destination().at_revision("c" * 40).as_dict(), "artifact_manifest_sha256": "d" * 64}]

    def _persist(**_kwargs):
        result = failures.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(run_all, "persist_run_artifacts", _persist)
    monkeypatch.setattr("sys.argv", ["run_all.py", "--config", str(config_path)])
    with pytest.raises(RuntimeError):
        run_all.main()
    root = next((tmp_path / ".openresearch" / "artifacts").iterdir())
    frozen = {name: (root / name).read_bytes() for name in ("summary.json", "provenance.json", "EVAL.md")}
    run_all.main()

    assert {name: (root / name).read_bytes() for name in frozen} == frozen
    assert (root / "COMPLETE.json").is_file()
    attempts = json.loads((root / "PERSISTENCE_ATTEMPTS.json").read_text())
    assert [attempt["status"] for attempt in attempts if attempt["kind"] == "atomic_upload"] == ["failed", "succeeded"]


def test_runner_rejects_nonpreflight_before_stage_without_token(tmp_path, monkeypatch) -> None:
    config_path = _write(tmp_path, "configs/claim1.yaml", "run:\n  name: gpu evidence\n  seed: 7\n  stage: claim1\nclaims: [1]\n")
    called = False

    def _unexpected_stage(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("stage should not run")

    monkeypatch.setattr(run_all, "REPO", tmp_path)
    monkeypatch.setattr(run_all, "run_stage", _unexpected_stage)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["run_all.py", "--config", str(config_path)])
    with pytest.raises(HubArtifactError, match="HF_TOKEN is required"):
        run_all.main()
    assert not called


def test_runner_checks_clean_tree_and_remote_write_before_nonpreflight_stage(tmp_path, monkeypatch) -> None:
    config_path = _write(tmp_path, "configs/claim1.yaml", "run:\n  name: gpu evidence\n  seed: 7\n  stage: claim1\nclaims: [1]\n")
    order: list[str] = []
    monkeypatch.setattr(run_all, "REPO", tmp_path)
    monkeypatch.setattr(run_all, "collect_provenance", lambda **_: {"repository": {"commit": "a" * 40}, "execution": {"backend": "local"}})
    monkeypatch.setattr(run_all, "require_clean_worktree", lambda *_: order.append("clean"))
    monkeypatch.setattr(
        run_all,
        "preflight_remote_write_permission",
        lambda **_: order.append("permission") or {"method": "mock", "status": "passed"},
    )
    monkeypatch.setattr(
        run_all,
        "run_stage",
        lambda *_args, **_kwargs: order.append("stage") or {"claim_evidence": False, "verdict": "not_evaluated", "limitations": []},
    )
    monkeypatch.setattr(
        run_all,
        "persist_run_artifacts",
        lambda **_: {"hub_commit_oid": "c" * 40, "hub_commit_url": "https://example.invalid/commit/c", "immutable_destination": _destination().at_revision("c" * 40).as_dict(), "artifact_manifest_sha256": "d" * 64},
    )
    monkeypatch.setenv("HF_TOKEN", "test-token-value")
    monkeypatch.setattr("sys.argv", ["run_all.py", "--config", str(config_path)])

    run_all.main()

    assert order == ["clean", "permission", "stage"]
