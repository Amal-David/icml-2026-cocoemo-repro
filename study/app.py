"""Private, blinded Gradio listening-study application.

Run only in a private Space/Bucket environment after the manifest and ethics
requirements in ``PREREGISTRATION.md`` have been frozen.  This module rejects
all missing configuration rather than opening a partially configured study.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from study.protocol import EMOTIONS, ProtocolError, assert_private_directory, load_invitations, load_manifest
from study.service import StudyService


@dataclass(frozen=True)
class StudyConfig:
    manifest_path: Path
    audio_root: Path
    storage_root: Path
    invitations_path: Path
    hmac_secret: str
    order_secret: str

    @classmethod
    def from_environment(cls) -> "StudyConfig":
        def required_path(name: str) -> Path:
            value = os.environ.get(name)
            if not value:
                raise ProtocolError(f"{name} must be configured")
            return Path(value).expanduser().resolve()

        def required_secret(name: str) -> str:
            value = os.environ.get(name)
            if not value or len(value) < 32:
                raise ProtocolError(f"{name} must be configured with at least 32 characters")
            return value

        return cls(
            manifest_path=required_path("STUDY_MANIFEST_PATH"),
            audio_root=required_path("STUDY_AUDIO_ROOT"),
            storage_root=required_path("STUDY_STORAGE_ROOT"),
            invitations_path=required_path("STUDY_INVITATIONS_PATH"),
            hmac_secret=required_secret("STUDY_HMAC_SECRET"),
            order_secret=required_secret("STUDY_ORDER_SECRET"),
        )

    def load_service(self) -> StudyService:
        if not self.manifest_path.is_file() or not self.audio_root.is_dir() or not self.invitations_path.is_file():
            raise ProtocolError("manifest, audio root, or invitation-code hash file is missing")
        assert_private_directory(self.storage_root)
        return StudyService(
            manifest=load_manifest(self.manifest_path, audio_root=self.audio_root),
            invitations=load_invitations(self.invitations_path),
            hmac_secret=self.hmac_secret,
            order_secret=self.order_secret,
            storage_root=self.storage_root,
        )


_PLAYBACK_GUARD = """
() => {
  const setPlayback = (milliseconds) => {
    const root = document.querySelector('#study-playback-ms');
    const input = root && root.querySelector('input, textarea');
    if (!input) return;
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    setter.call(input, String(Math.round(milliseconds)));
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  };
  document.addEventListener('ended', (event) => {
    const audio = event.target;
    if (!(audio instanceof HTMLAudioElement) || !audio.closest('#study-audio')) return;
    setPlayback(audio.duration * 1000);
  }, true);
}
"""


def build_app(service: StudyService, *, audio_root: Path):
    """Build an app without performing any deployment or recruitment action."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("install study/requirements.txt before launching the listening study") from exc

    def current_view(state: dict[str, Any]):
        trial = service.current_trial(state)
        return (
            gr.update(value=str((audio_root / trial.audio_path).resolve()), visible=True),
            gr.update(value=f"Trial {state['index'] + 1} of 30", visible=True),
            gr.update(value="", visible=False),
            gr.update(visible=True),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def start(code: str, consent: bool, adult: bool):
        try:
            state = service.start(invitation_code=code, consent=consent, adult=adult)
            updates = current_view(state)
            return (state, gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=True), *updates)
        except ProtocolError as exc:
            return (None, gr.update(value=str(exc), visible=True), gr.update(visible=True), gr.update(visible=False), *([gr.update()] * 13))

    def unlock(state: dict[str, Any], playback_ms: str):
        try:
            trial = service.current_trial(state)
            milliseconds = int(playback_ms)
            if milliseconds < 0:
                raise ProtocolError("browser playback telemetry is invalid")
            return gr.update(value="", visible=False), gr.update(visible=True)
        except (TypeError, ValueError, ProtocolError) as exc:
            return gr.update(value=str(exc), visible=True), gr.update(visible=False)

    def submit(
        state: dict[str, Any],
        playback_ms: str,
        naturalness: int,
        dominant: str,
        angry: float,
        happy: float,
        sad: float,
        surprised: float,
        neutral: float,
    ):
        try:
            trial = service.current_trial(state)
            raw_allocations = {
                "angry": angry,
                "happy": happy,
                "sad": sad,
                "surprised": surprised,
                "neutral": neutral,
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not float(value).is_integer()
                for value in raw_allocations.values()
            ):
                raise ProtocolError("each emotion allocation must be a whole number")
            allocations = {emotion: int(value) for emotion, value in raw_allocations.items()}
            response = {
                "protocol_version": service.manifest["protocol_version"],
                "manifest_hash": service.frozen_manifest_hash,
                "trial_id": trial.trial_id,
                "audio_sha256": trial.audio_sha256,
                "playback_duration_ms": int(playback_ms),
                "naturalness_score": naturalness,
                "dominant_emotion": dominant,
                "emotion_allocation": allocations,
            }
            next_state = service.record_trial(state, response)
            if next_state["index"] == 30:
                return (
                    next_state,
                    gr.update(value="All ratings are recorded. Complete the study to create the one immutable submission.", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(visible=False),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            updates = current_view(next_state)
            return (next_state, gr.update(value="", visible=False), *updates)
        except (TypeError, ValueError, ProtocolError) as exc:
            return (state, gr.update(value=str(exc), visible=True), *([gr.update()] * 13))

    def finish(state: dict[str, Any]):
        try:
            service.finish(state)
            return None, gr.update(value="Submission recorded. Thank you.", visible=True), gr.update(visible=False)
        except ProtocolError as exc:
            return state, gr.update(value=str(exc), visible=True), gr.update(visible=True)

    with gr.Blocks(js=_PLAYBACK_GUARD, title="Listening study") as app:
        session = gr.State(value=None)
        gr.Markdown("# Listening Study")
        status = gr.Markdown(visible=False)
        with gr.Column(visible=True) as entry:
            gr.Markdown((Path(__file__).with_name("consent.md")).read_text(encoding="utf-8"))
            code = gr.Textbox(label="Invitation code", type="password", autocomplete="off")
            consent = gr.Checkbox(label="I consent to participate in this study.")
            adult = gr.Checkbox(label="I confirm that I am at least 18 years old.")
            begin = gr.Button("Begin")
        with gr.Column(visible=False) as study:
            counter = gr.Markdown()
            audio = gr.Audio(label="Audio", type="filepath", interactive=False, elem_id="study-audio")
            playback_ms = gr.Textbox(value="", visible=False, elem_id="study-playback-ms")
            unlock_button = gr.Button("Unlock ratings")
            with gr.Column(visible=False) as rating:
                naturalness = gr.Radio([1, 2, 3, 4, 5], label="Naturalness", type="value")
                dominant = gr.Radio(list(EMOTIONS), label="Dominant emotion", type="value")
                with gr.Row():
                    angry = gr.Number(label="Angry", precision=0)
                    happy = gr.Number(label="Happy", precision=0)
                    sad = gr.Number(label="Sad", precision=0)
                    surprised = gr.Number(label="Surprised", precision=0)
                    neutral = gr.Number(label="Neutral", precision=0)
                next_button = gr.Button("Next")
            finish_button = gr.Button("Complete study", visible=False)

        begin.click(
            start,
            inputs=[code, consent, adult],
            outputs=[session, status, entry, study, audio, counter, playback_ms, unlock_button, naturalness, dominant, angry, happy, sad, surprised, neutral, rating, finish_button],
        )
        unlock_button.click(unlock, inputs=[session, playback_ms], outputs=[status, rating])
        next_button.click(
            submit,
            inputs=[session, playback_ms, naturalness, dominant, angry, happy, sad, surprised, neutral],
            outputs=[session, status, audio, counter, playback_ms, unlock_button, naturalness, dominant, angry, happy, sad, surprised, neutral, rating, finish_button],
        )
        finish_button.click(finish, inputs=[session], outputs=[session, status, finish_button])
    return app


def main() -> None:
    config = StudyConfig.from_environment()
    service = config.load_service()
    app = build_app(service, audio_root=config.audio_root)
    app.launch(allowed_paths=[str(config.audio_root)], show_error=False)


if __name__ == "__main__":
    main()
