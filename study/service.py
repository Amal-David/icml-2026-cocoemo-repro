"""Stateful, server-validated study service used by the Gradio app."""

from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from study.protocol import (
    PROTOCOL_VERSION,
    Invitation,
    ProtocolError,
    Trial,
    build_experimental_trials,
    build_quality_trials,
    canonical_json,
    hmac_code,
    interleave_trials,
    manifest_hash,
    validate_completed_submission,
    validate_response,
)
from study.storage import write_immutable_submission


@dataclass(frozen=True)
class StudyService:
    manifest: Mapping[str, Any]
    invitations: Mapping[str, Invitation]
    hmac_secret: str
    order_secret: str
    storage_root: Path

    @property
    def frozen_manifest_hash(self) -> str:
        return manifest_hash(self.manifest)

    def _session_tag(self, state: Mapping[str, Any]) -> str:
        payload = canonical_json(
            {
                key: state[key]
                for key in (
                    "study_id_hmac",
                    "group",
                    "priority",
                    "started_unix_ms",
                    "nonce",
                    "index",
                    "responses",
                )
            }
        )
        return hmac.new(self.order_secret.encode("utf-8"), payload, "sha256").hexdigest()

    def _validate_state(self, state: Mapping[str, Any]) -> Invitation:
        required = {"study_id_hmac", "group", "priority", "started_unix_ms", "nonce", "session_tag", "responses", "index"}
        if not required.issubset(state):
            raise ProtocolError("study session is incomplete")
        study_id_hmac = state["study_id_hmac"]
        invitation = self.invitations.get(study_id_hmac)
        if invitation is None:
            raise ProtocolError("study invitation is not recognized")
        if state["group"] != invitation.group or state["priority"] != invitation.priority:
            raise ProtocolError("study session assignment is not recognized")
        if not isinstance(state["nonce"], str) or not isinstance(state["started_unix_ms"], int):
            raise ProtocolError("study session integrity data is invalid")
        expected_tag = self._session_tag(state)
        if not isinstance(state["session_tag"], str) or not hmac.compare_digest(expected_tag, state["session_tag"]):
            raise ProtocolError("study session integrity check failed")
        if not isinstance(state["responses"], list) or not isinstance(state["index"], int):
            raise ProtocolError("study session response state is invalid")
        return invitation

    def _schedule(self, state: Mapping[str, Any]) -> list[Trial]:
        invitation = self._validate_state(state)
        experiment = build_experimental_trials(
            self.manifest,
            group=invitation.group,
            study_id_hmac=str(state["study_id_hmac"]),
            order_secret=self.order_secret,
        )
        quality = build_quality_trials(
            self.manifest,
            group=invitation.group,
            study_id_hmac=str(state["study_id_hmac"]),
            order_secret=self.order_secret,
        )
        return interleave_trials(experiment, quality)

    def start(self, *, invitation_code: str, consent: bool, adult: bool) -> dict[str, Any]:
        if consent is not True or adult is not True:
            raise ProtocolError("adult consent is required before any study trial")
        study_id_hmac = hmac_code(invitation_code, self.hmac_secret)
        invitation = self.invitations.get(study_id_hmac)
        if invitation is None:
            raise ProtocolError("invitation code is not valid")
        state: dict[str, Any] = {
            "study_id_hmac": study_id_hmac,
            "group": invitation.group,
            "priority": invitation.priority,
            "started_unix_ms": round(time.time() * 1000),
            "nonce": secrets.token_hex(16),
            "responses": [],
            "index": 0,
        }
        state["session_tag"] = self._session_tag(state)
        self._schedule(state)
        return state

    def current_trial(self, state: Mapping[str, Any]) -> Trial:
        schedule = self._schedule(state)
        index = state["index"]
        if not 0 <= index < len(schedule):
            raise ProtocolError("study session has no remaining trial")
        return schedule[index]

    def record_trial(self, state: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
        schedule = self._schedule(state)
        index = state["index"]
        if not 0 <= index < len(schedule) or len(state["responses"]) != index:
            raise ProtocolError("study session trial sequence is invalid")
        trial = schedule[index]
        normalized = validate_response(
            response,
            trial=trial,
            protocol_version=PROTOCOL_VERSION,
            expected_manifest_hash=self.frozen_manifest_hash,
        )
        next_state = dict(state)
        next_state["responses"] = [*state["responses"], normalized]
        next_state["index"] = index + 1
        next_state["session_tag"] = self._session_tag(next_state)
        return next_state

    def finish(self, state: Mapping[str, Any], *, now_unix_ms: int | None = None) -> Path:
        invitation = self._validate_state(state)
        schedule = self._schedule(state)
        now = round(time.time() * 1000) if now_unix_ms is None else now_unix_ms
        elapsed_seconds = (now - state["started_unix_ms"]) / 1000
        if elapsed_seconds < 0:
            raise ProtocolError("server clock produced a negative completion time")
        submission = validate_completed_submission(
            study_id_hmac=str(state["study_id_hmac"]),
            group=invitation.group,
            priority=invitation.priority,
            schedule=schedule,
            responses=state["responses"],
            expected_manifest_hash=self.frozen_manifest_hash,
            elapsed_seconds=elapsed_seconds,
        )
        return write_immutable_submission(self.storage_root, submission)
