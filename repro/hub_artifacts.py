"""Atomic, fail-closed persistence of public reproduction evidence on the Hub."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import quote

from repro.contracts import ContractError


ARTIFACT_DATASET_REPO = "amal-david/cocoemo-repro-artifacts"
ARTIFACT_SCHEMA_VERSION = 2
MAX_TEXT_BYTES = 12 * 1024 * 1024
MAX_JSONL_BYTES = 48 * 1024 * 1024
MAX_FIGURE_BYTES = 20 * 1024 * 1024
MAX_JSONL_LINES = 1_000_000

# This is intentionally a schema, not an extension-based recursive walk.  A
# runner can retain source WAVs and model caches beside its reports locally;
# they are never candidates for a public commit.
FIXED_TOP_LEVEL_FILES = frozenset(
    {
        "summary.json",
        "provenance.json",
        "resolved_config.json",
        "artifact_manifest.json",
        "EVAL.md",
    }
)
DERIVED_FILE = re.compile(
    r"(?:"
    r"(?:claim[1-5](?:_[a-z0-9]+)*)|"
    r"(?:per_sample(?:_acoustic)?)|"
    r"(?:acoustic_summary|manifest|hook_shapes|pair_ledger|train_vector_manifest)|"
    r"(?:selection_audit|selection_variants|cremad_audio_vote_proxy_v1|failed_inputs)|"
    r"(?:metrics(?:_[a-z0-9]+)*|baseline(?:_[a-z0-9]+)*|[a-z0-9]+_summary)"
    r")\.(?:json|jsonl|md|csv|sha256|sha512)$"
)
# Raw evaluator representations remain in the local run root so retries can
# validate the scientific computation, but they never cross the public Hub
# boundary.  Their public counterpart is a hash/count-only manifest.
EXCLUDED_FILES = frozenset(
    {
        "COMPLETE.json",
        "LOCAL_READY.json",
        "hub_artifact_manifest.json",
        "claim5_reference_embeddings.jsonl",
    }
)
SENSITIVE_NAME = re.compile(r"(?:credential|secret|token|private[_-]?key|\.env)", re.IGNORECASE)
SENSITIVE_CONTENT = re.compile(
    r"(?:\b(?:HF_TOKEN|HUGGINGFACE_HUB_TOKEN)\s*=\s*[^\s,;]+|Authorization\s*:\s*Bearer\s+[^\s,;]+|\bhf_[A-Za-z0-9]{12,}|\bsk-[A-Za-z0-9]{12,})",
    re.IGNORECASE,
)
SUSPICIOUS_JSON_KEY = re.compile(
    r"(?:credential|secret|token|password|api[_-]?key|private[_-]?key|binary|base64|blob|payload|raw[_-]?(?:audio|bytes|data))",
    re.IGNORECASE,
)
RAW_REPRESENTATION_JSON_KEY = re.compile(
    r"^(?:raw_)?(?:embedding|embeddings|activation|activations|tensor|tensors)$",
    re.IGNORECASE,
)
BASE64ISH = re.compile(r"^[A-Za-z0-9+/=_-]{256,}$")
RUN_NAME = re.compile(r"[^a-z0-9]+")
HEX_OID = re.compile(r"[0-9a-f]{7,64}")


class HubArtifactError(ContractError):
    """Raised when evidence cannot be safely or durably persisted."""


class HubApi(Protocol):
    def file_exists(self, **kwargs: Any) -> bool: ...

    def repo_info(self, **kwargs: Any) -> Any: ...

    def create_commit(self, **kwargs: Any) -> Any: ...

    def get_paths_info(self, **kwargs: Any) -> Any: ...

    def hf_hub_download(self, **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class ArtifactDestination:
    repo_id: str
    remote_prefix: str
    directory_url: str
    complete_url: str
    revision: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {
            "repo_id": self.repo_id,
            "repo_type": "dataset",
            "remote_prefix": self.remote_prefix,
            "directory_url": self.directory_url,
            "complete_url": self.complete_url,
        }
        if self.revision:
            result["revision"] = self.revision
        return result

    def at_revision(self, revision: str) -> "ArtifactDestination":
        if not HEX_OID.fullmatch(revision):
            raise HubArtifactError("Hub commit oid is not a safe hexadecimal identifier")
        encoded_prefix = quote(self.remote_prefix, safe="/")
        dataset = quote(self.repo_id, safe="/")
        return ArtifactDestination(
            repo_id=self.repo_id,
            remote_prefix=self.remote_prefix,
            directory_url=f"https://huggingface.co/datasets/{dataset}/tree/{revision}/{encoded_prefix}",
            complete_url=f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{encoded_prefix}/COMPLETE.json",
            revision=revision,
        )


@dataclass(frozen=True)
class UploadCandidate:
    absolute_path: Path
    relative_path: str
    bytes: int
    sha256: str
    payload: bytes

    def as_manifest(self) -> dict[str, Any]:
        return {"path": self.relative_path, "bytes": self.bytes, "sha256": self.sha256}


def require_hf_token(token: str | None = None) -> str:
    value = token if token is not None else os.environ.get("HF_TOKEN")
    if not value or not value.strip():
        raise HubArtifactError("HF_TOKEN is required before a non-preflight evidence run can start")
    return value


def artifact_destination(*, commit: str, config_digest: str, run_name: str, repo_id: str = ARTIFACT_DATASET_REPO) -> ArtifactDestination:
    if not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        raise HubArtifactError("repository commit is not a safe immutable hexadecimal identifier")
    if not re.fullmatch(r"[0-9a-f]{64}", config_digest):
        raise HubArtifactError("config digest is not a SHA-256 hexadecimal digest")
    name = RUN_NAME.sub("-", run_name.lower()).strip("-")
    if not name:
        raise HubArtifactError("run name does not produce a safe artifact path")
    prefix = f"runs/{commit}-{config_digest}-{name}"
    encoded_prefix = quote(prefix, safe="/")
    dataset = quote(repo_id, safe="/")
    return ArtifactDestination(
        repo_id=repo_id,
        remote_prefix=prefix,
        directory_url=f"https://huggingface.co/datasets/{dataset}/tree/main/{encoded_prefix}",
        complete_url=f"https://huggingface.co/datasets/{dataset}/resolve/main/{encoded_prefix}/COMPLETE.json",
    )


def _is_allowed_top_level(name: str) -> bool:
    return name in FIXED_TOP_LEVEL_FILES or bool(DERIVED_FILE.fullmatch(name))


def _reject_text_content(content: str, *, path: Path) -> None:
    if "\x00" in content:
        raise HubArtifactError(f"candidate artifact contains binary content: {path.name}")
    if SENSITIVE_CONTENT.search(content):
        raise HubArtifactError(f"candidate artifact contains credential-like content: {path.name}")


def _reject_suspicious_json(value: Any, *, path: Path, key: str | None = None) -> None:
    if key and SUSPICIOUS_JSON_KEY.search(key):
        raise HubArtifactError(f"candidate artifact contains a disallowed payload field: {path.name}")
    if key and RAW_REPRESENTATION_JSON_KEY.fullmatch(key):
        raise HubArtifactError(f"candidate artifact contains a raw representation field: {path.name}")
    if isinstance(value, dict):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise HubArtifactError(f"candidate artifact has a non-string JSON key: {path.name}")
            _reject_suspicious_json(child, path=path, key=child_key)
    elif isinstance(value, list):
        for child in value:
            _reject_suspicious_json(child, path=path)
    elif isinstance(value, str):
        _reject_text_content(value, path=path)
        if BASE64ISH.fullmatch(value):
            raise HubArtifactError(f"candidate artifact contains a suspicious base64-like string: {path.name}")


def _validate_text_payload(path: Path, payload: bytes) -> None:
    limit = MAX_JSONL_BYTES if path.suffix.lower() == ".jsonl" else MAX_TEXT_BYTES
    if len(payload) > limit:
        raise HubArtifactError(f"candidate artifact exceeds the public size cap: {path.name}")
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HubArtifactError(f"candidate artifact is not valid UTF-8 text: {path.name}") from exc
    _reject_text_content(content, path=path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            _reject_suspicious_json(json.loads(content), path=path)
        except json.JSONDecodeError as exc:
            raise HubArtifactError(f"candidate JSON artifact is malformed: {path.name}") from exc
    elif suffix == ".jsonl":
        for index, line in enumerate(content.splitlines(), start=1):
            if index > MAX_JSONL_LINES:
                raise HubArtifactError(f"candidate JSONL artifact exceeds line cap: {path.name}")
            if not line.strip():
                continue
            try:
                _reject_suspicious_json(json.loads(line), path=path)
            except json.JSONDecodeError as exc:
                raise HubArtifactError(f"candidate JSONL artifact is malformed at line {index}: {path.name}") from exc


def _validate_figure(path: Path, payload: bytes) -> None:
    if len(payload) > MAX_FIGURE_BYTES:
        raise HubArtifactError(f"figure exceeds the public size cap: {path.name}")
    if path.suffix.lower() == ".png":
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HubArtifactError(f"figure is not a PNG payload: {path.name}")
        return
    try:
        svg = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HubArtifactError(f"SVG figure is not UTF-8: {path.name}") from exc
    _reject_text_content(svg, path=path)
    if not re.search(r"<svg(?:\s|>)", svg, re.IGNORECASE) or re.search(
        r"<(?:script|foreignObject|image)\b|(?:href|xlink:href)\s*=\s*['\"](?:https?:|data:)", svg, re.IGNORECASE
    ):
        raise HubArtifactError(f"SVG figure contains active or external content: {path.name}")


def _snapshot_candidate(path: Path, relative_path: str) -> UploadCandidate:
    if path.is_symlink() or not path.is_file():
        raise HubArtifactError(f"refusing non-regular artifact: {path}")
    payload = path.read_bytes()
    if relative_path.startswith("figures/"):
        _validate_figure(path, payload)
    else:
        _validate_text_payload(path, payload)
    return UploadCandidate(
        absolute_path=path.resolve(),
        relative_path=relative_path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


def collect_upload_candidates(run_root: Path) -> list[UploadCandidate]:
    """Return one immutable snapshot of the public, top-level artifact schema."""

    if run_root.is_symlink():
        raise HubArtifactError(f"refusing symlinked run root: {run_root}")
    root = run_root.resolve()
    if not root.is_dir():
        raise HubArtifactError(f"run root does not exist: {run_root}")
    candidates: list[UploadCandidate] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name in EXCLUDED_FILES or path.name.startswith("."):
            continue
        if path.is_dir():
            if path.name != "figures":
                continue
            if path.is_symlink():
                raise HubArtifactError(f"refusing symlinked figures directory: {path}")
            figures_root = path.resolve()
            try:
                figures_root.relative_to(root)
            except ValueError as exc:
                raise HubArtifactError(f"figures directory escapes run root: {path}") from exc
            for figure in sorted(path.iterdir(), key=lambda item: item.name):
                if figure.is_dir() or figure.name.startswith(".") or figure.suffix.lower() not in {".png", ".svg"}:
                    continue
                if figure.is_symlink():
                    raise HubArtifactError(f"refusing symlinked figure: {figure}")
                try:
                    figure.resolve().relative_to(figures_root)
                except ValueError as exc:
                    raise HubArtifactError(f"figure escapes figures directory: {figure}") from exc
                candidates.append(_snapshot_candidate(figure, f"figures/{figure.name}"))
            continue
        if SENSITIVE_NAME.search(path.name) or not _is_allowed_top_level(path.name):
            continue
        candidates.append(_snapshot_candidate(path, path.name))
    if not candidates:
        raise HubArtifactError("run produced no public-safe derived artifacts to persist")
    return candidates


def write_public_artifact_manifest(run_root: Path) -> Path:
    """Bind only the exact public candidate schema for local retry checks."""

    candidates = collect_upload_candidates(run_root)
    manifest_path = run_root / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "scope": "public_upload_candidates_only",
                "artifacts": [candidate.as_manifest() for candidate in candidates],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _manifest_candidate(candidates: list[UploadCandidate], destination: ArtifactDestination, run_root: Path) -> UploadCandidate:
    manifest_path = run_root / "hub_artifact_manifest.json"
    payload = (
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "destination": destination.as_dict(),
                "files": [candidate.as_manifest() for candidate in candidates],
                "snapshot": "candidate bytes were hashed before the atomic Hub commit was created",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(payload)
    return UploadCandidate(
        absolute_path=manifest_path.resolve(),
        relative_path=manifest_path.name,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


def _default_api_factory(token: str) -> HubApi:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency contract is exercised in jobs
        raise HubArtifactError("huggingface_hub is required to persist evidence") from exc
    return HfApi(token=token)


def _commit_add(path_in_repo: str, payload: bytes) -> Any:
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as exc:  # pragma: no cover
        raise HubArtifactError("huggingface_hub is required to create atomic evidence commits") from exc
    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=payload)


def _commit_delete(path_in_repo: str) -> Any:
    try:
        from huggingface_hub import CommitOperationDelete
    except ImportError as exc:  # pragma: no cover
        raise HubArtifactError("huggingface_hub is required to clean permission canaries") from exc
    return CommitOperationDelete(path_in_repo=path_in_repo)


def _repo_oid(api: HubApi, destination: ArtifactDestination) -> str:
    try:
        info = api.repo_info(repo_id=destination.repo_id, repo_type="dataset", revision="main")
    except Exception as exc:
        raise HubArtifactError("could not read the artifact dataset head; refusing an unguarded write") from exc
    oid = getattr(info, "sha", None) or getattr(info, "oid", None)
    if not isinstance(oid, str) or not HEX_OID.fullmatch(oid):
        raise HubArtifactError("artifact dataset head did not provide a usable commit oid")
    return oid


def _commit_oid(commit: Any) -> str:
    oid = getattr(commit, "oid", None) or getattr(commit, "sha", None)
    if not isinstance(oid, str) or not HEX_OID.fullmatch(oid):
        raise HubArtifactError("Hub create_commit did not return a usable commit oid")
    return oid


def _hub_commit_url(destination: ArtifactDestination, oid: str, commit: Any) -> str:
    returned = getattr(commit, "commit_url", None)
    if isinstance(returned, str) and returned.startswith("https://"):
        return returned.split("?", 1)[0].split("#", 1)[0]
    return f"https://huggingface.co/datasets/{quote(destination.repo_id, safe='/')}/commit/{oid}"


def _remote_bytes(api: HubApi, *, destination: ArtifactDestination, filename: str, revision: str) -> bytes:
    # Test adapters provide an in-memory reader.  Production uses the Hub API
    # cache and reads the exact revision-pinned file from it.
    reader = getattr(api, "read_file", None)
    if callable(reader):
        value = reader(
            repo_id=destination.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
        if isinstance(value, bytes):
            return value
        raise HubArtifactError("remote artifact reader returned a non-bytes payload")
    try:
        local_path = api.hf_hub_download(
            repo_id=destination.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
        return Path(local_path).read_bytes()
    except Exception as exc:
        raise HubArtifactError(f"could not fetch remote artifact for reconciliation: {filename}") from exc


def _complete_revision(api: HubApi, *, destination: ArtifactDestination, head: str) -> str:
    complete_path = f"{destination.remote_prefix}/COMPLETE.json"
    try:
        paths = api.get_paths_info(
            repo_id=destination.repo_id,
            paths=[complete_path],
            repo_type="dataset",
            revision=head,
        )
    except Exception:
        # The current head is still an immutable revision and is a safe
        # fallback when the server does not expose a file-level last commit.
        return head
    if not paths:
        return head
    last_commit = getattr(paths[0], "last_commit", None)
    oid = getattr(last_commit, "oid", None)
    return oid if isinstance(oid, str) and HEX_OID.fullmatch(oid) else head


def _reconcile_remote_artifacts(
    *,
    api: HubApi,
    destination: ArtifactDestination,
    head: str,
    candidates: list[UploadCandidate],
    manifest: UploadCandidate,
) -> dict[str, Any] | None:
    """Accept a lost response only after byte-for-byte remote reconciliation."""

    complete_path = f"{destination.remote_prefix}/COMPLETE.json"
    try:
        if not api.file_exists(
            repo_id=destination.repo_id,
            filename=complete_path,
            repo_type="dataset",
            revision=head,
        ):
            return None
    except Exception as exc:
        raise HubArtifactError("could not check remote completion during reconciliation") from exc
    revision = _complete_revision(api, destination=destination, head=head)
    complete = _remote_bytes(api, destination=destination, filename=complete_path, revision=revision)
    remote_manifest_bytes = _remote_bytes(
        api,
        destination=destination,
        filename=f"{destination.remote_prefix}/{manifest.relative_path}",
        revision=revision,
    )
    try:
        complete_body = json.loads(complete)
        remote_manifest = json.loads(remote_manifest_bytes)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HubArtifactError("remote completion or manifest is malformed") from exc
    if complete_body.get("status") != "complete" or complete_body.get("atomic") is not True:
        raise HubArtifactError("remote completion marker is not an atomic evidence commit")
    if complete_body.get("destination") != destination.as_dict():
        raise HubArtifactError("remote completion marker does not bind this run identity")
    if complete_body.get("artifact_manifest_sha256") != hashlib.sha256(remote_manifest_bytes).hexdigest():
        raise HubArtifactError("remote completion marker does not match its manifest")
    if remote_manifest_bytes != manifest.payload:
        raise HubArtifactError("remote Hub manifest does not match the frozen local snapshot")
    if remote_manifest.get("destination") != destination.as_dict():
        raise HubArtifactError("remote Hub manifest has a different evidence destination")
    expected = [candidate.as_manifest() for candidate in candidates]
    if remote_manifest.get("files") != expected:
        raise HubArtifactError("remote artifact manifest does not match the frozen local snapshot")
    if complete_body.get("files_uploaded") != [candidate.relative_path for candidate in candidates]:
        raise HubArtifactError("remote completion file list does not match the frozen local snapshot")
    # The manifest alone is insufficient after a lost response: fetch every
    # candidate at the immutable revision and verify its exact bytes and hash.
    for candidate in candidates:
        remote_payload = _remote_bytes(
            api,
            destination=destination,
            filename=f"{destination.remote_prefix}/{candidate.relative_path}",
            revision=revision,
        )
        if len(remote_payload) != candidate.bytes or hashlib.sha256(remote_payload).hexdigest() != candidate.sha256:
            raise HubArtifactError(f"remote artifact does not match frozen snapshot: {candidate.relative_path}")
    immutable_destination = destination.at_revision(revision)
    return {
        "status": "reconciled_remote_atomic_commit",
        "destination": destination.as_dict(),
        "immutable_destination": immutable_destination.as_dict(),
        "hub_commit_oid": revision,
        "hub_commit_url": f"https://huggingface.co/datasets/{quote(destination.repo_id, safe='/')}/commit/{revision}",
        "uploaded_files": [candidate.relative_path for candidate in candidates],
        "artifact_manifest_sha256": manifest.sha256,
        "expected_parent_commit": head,
    }


def preflight_remote_write_permission(
    *,
    destination: ArtifactDestination,
    token: str,
    api_factory: Callable[[str], HubApi] = _default_api_factory,
) -> dict[str, Any]:
    """Prove write access before a costly evidence stage starts.

    Hugging Face does not expose a repository-scoped non-mutating write check
    through the public API.  An adapter may provide ``has_write_access`` when
    it has authoritative information; otherwise this makes and immediately
    deletes a tiny deterministic canary under a dedicated namespace.
    """

    token = require_hf_token(token)
    api = api_factory(token)
    checker = getattr(api, "has_write_access", None)
    if callable(checker):
        allowed = checker(repo_id=destination.repo_id, repo_type="dataset")
        if allowed is True:
            return {"method": "authoritative_non_mutating_check", "status": "passed"}
        if allowed is False:
            raise HubArtifactError("token lacks write permission for the artifact dataset")

    parent = _repo_oid(api, destination)
    probe_digest = hashlib.sha256(f"{destination.repo_id}\n{destination.remote_prefix}".encode()).hexdigest()
    probe_path = f"permission-canaries/{probe_digest}.json"
    try:
        if api.file_exists(repo_id=destination.repo_id, filename=probe_path, repo_type="dataset", revision=parent):
            raise HubArtifactError("a prior permission canary remains remotely; refusing to overwrite it")
        payload = (json.dumps({"schema_version": ARTIFACT_SCHEMA_VERSION, "purpose": "write_permission_preflight", "target": destination.remote_prefix}, sort_keys=True) + "\n").encode("utf-8")
        probe = api.create_commit(
            repo_id=destination.repo_id,
            repo_type="dataset",
            operations=[_commit_add(probe_path, payload)],
            commit_message="CoCoEmo artifact write-permission preflight",
            parent_commit=parent,
        )
        probe_oid = _commit_oid(probe)
        cleanup = api.create_commit(
            repo_id=destination.repo_id,
            repo_type="dataset",
            operations=[_commit_delete(probe_path)],
            commit_message="Clean CoCoEmo artifact write-permission preflight",
            parent_commit=probe_oid,
        )
        cleanup_oid = _commit_oid(cleanup)
    except HubArtifactError:
        raise
    except Exception as exc:
        raise HubArtifactError("artifact dataset write-permission preflight failed") from exc
    return {
        "method": "deterministic_commit_canary",
        "status": "passed",
        "probe_path": probe_path,
        "probe_commit_oid": probe_oid,
        "cleanup_commit_oid": cleanup_oid,
    }


def persist_run_artifacts(
    *,
    run_root: Path,
    destination: ArtifactDestination,
    token: str,
    api_factory: Callable[[str], HubApi] = _default_api_factory,
) -> dict[str, Any]:
    """Persist an exact snapshot with one parent-guarded Hub commit.

    A remote COMPLETE marker is part of the same commit as the hashes and all
    candidates.  A concurrent head change rejects the operation rather than
    publishing a stale or partial evidence directory.
    """

    token = require_hf_token(token)
    candidates = collect_upload_candidates(run_root)
    candidates.sort(key=lambda item: item.relative_path)
    manifest = _manifest_candidate(candidates, destination, run_root)
    api = api_factory(token)
    parent = _repo_oid(api, destination)
    reconciled = _reconcile_remote_artifacts(
        api=api,
        destination=destination,
        head=parent,
        candidates=candidates,
        manifest=manifest,
    )
    if reconciled is not None:
        return reconciled

    complete_payload = (
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": "complete",
                "artifact_manifest_path": manifest.relative_path,
                "artifact_manifest_sha256": manifest.sha256,
                "files_uploaded": [candidate.relative_path for candidate in candidates],
                "destination": destination.as_dict(),
                "remote_prefix": destination.remote_prefix,
                "atomic": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    operations = [
        _commit_add(f"{destination.remote_prefix}/{candidate.relative_path}", candidate.payload)
        for candidate in [*candidates, manifest]
    ]
    operations.append(_commit_add(f"{destination.remote_prefix}/COMPLETE.json", complete_payload))
    try:
        commit = api.create_commit(
            repo_id=destination.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Persist CoCoEmo evidence atomically: {destination.remote_prefix}",
            parent_commit=parent,
        )
    except Exception as exc:
        # A transport timeout can happen after the server has committed the
        # request.  Re-read the current head and accept only an exact remote
        # byte-for-byte match; otherwise preserve the original failure.
        try:
            reconciled = _reconcile_remote_artifacts(
                api=api,
                destination=destination,
                head=_repo_oid(api, destination),
                candidates=candidates,
                manifest=manifest,
            )
        except HubArtifactError:
            reconciled = None
        if reconciled is not None:
            return reconciled
        raise HubArtifactError("atomic evidence commit failed; no completion state was accepted") from exc
    oid = _commit_oid(commit)
    immutable_destination = destination.at_revision(oid)
    return {
        "status": "uploaded_atomically",
        "destination": destination.as_dict(),
        "immutable_destination": immutable_destination.as_dict(),
        "hub_commit_oid": oid,
        "hub_commit_url": _hub_commit_url(destination, oid, commit),
        "uploaded_files": [candidate.relative_path for candidate in candidates],
        "artifact_manifest_sha256": manifest.sha256,
        "expected_parent_commit": parent,
    }


def safe_upload_error(error: BaseException) -> dict[str, str]:
    """Keep retry diagnostics useful without serializing an authentication secret."""

    message = SENSITIVE_CONTENT.sub("[redacted]", str(error))
    return {"type": type(error).__name__, "message": message[:500]}
