from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from repro.contracts import ContractError


EMOTIONS = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry", "08": "surprise"}
STATEMENTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}


@dataclass(frozen=True)
class RavdessClip:
    path: str
    actor: str
    statement: str
    repetition: str
    emotion: str
    transcript: str


def parse_ravdess_clip(path: str | Path) -> RavdessClip | None:
    source = Path(path)
    parts = source.stem.split("-")
    if len(parts) != 7:
        raise ContractError(f"invalid RAVDESS filename: {source.name}")
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "03" or channel != "01" or intensity != "01":
        return None
    if emotion not in EMOTIONS:
        return None
    if statement not in STATEMENTS or repetition not in {"01", "02"}:
        raise ContractError(f"unsupported RAVDESS statement/repetition: {source.name}")
    return RavdessClip(
        path=source.as_posix(),
        actor=actor,
        statement=statement,
        repetition=repetition,
        emotion=EMOTIONS[emotion],
        transcript=STATEMENTS[statement],
    )


def build_claim1_groups(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, RavdessClip]] = {}
    for path in paths:
        clip = parse_ravdess_clip(path)
        if clip is None:
            continue
        key = (clip.actor, clip.statement, clip.repetition)
        emotions = grouped.setdefault(key, {})
        if clip.emotion in emotions:
            raise ContractError(f"duplicate RAVDESS emotion in group {key}: {clip.emotion}")
        emotions[clip.emotion] = clip

    required = set(EMOTIONS.values())
    results = []
    for key, emotions in sorted(grouped.items()):
        missing = sorted(required.difference(emotions))
        if missing:
            raise ContractError(f"incomplete RAVDESS group {key}: missing {', '.join(missing)}")
        actor, statement, repetition = key
        results.append(
            {
                "group_id": f"actor_{actor}_statement_{statement}_repetition_{repetition}",
                "actor": actor,
                "statement": statement,
                "repetition": repetition,
                "target_text": STATEMENTS[statement],
                "references": {emotion: asdict(emotions[emotion]) for emotion in sorted(required)},
            }
        )
    if not results:
        raise ContractError("no complete RAVDESS Claim 1 groups found")
    return results


def require_full_public_scope(groups: list[dict[str, object]]) -> None:
    if len(groups) != 96:
        raise ContractError(f"full public RAVDESS scope requires 96 independent groups, found {len(groups)}")
