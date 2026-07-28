from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from repro.claim4_protocol import (
    Claim4ManifestItem,
    LfsPointer,
    build_claim4_manifest,
    ensure_lfs_wav,
    is_claim4_eligible,
    parse_lfs_pointer,
)
from repro.contracts import ContractError
from repro.cremad_selection import VoteRow


def _row(
    actor: int, row_id: int, *, counts: dict[str, int] | None = None, label: str = "H", code: str = "HAP",
    majority: tuple[str, ...] = ("A", "N"),
) -> VoteRow:
    values = counts or {"A": 2, "D": 0, "F": 0, "H": 1, "N": 2, "S": 0}
    return VoteRow(
        row_id=row_id,
        file_name=f"{actor:04d}_IEO_{code}_XX",
        provided_label=label,
        counts=values,
        num_responses=sum(values.values()),
        majority_labels=majority,
    )


def test_manifest_is_deterministic_one_item_per_actor_and_order_independent() -> None:
    rows = [_row(actor, actor * 10) for actor in range(1001, 1005)]
    rows.extend(
        _row(actor, actor * 10 + 1, label="N", code="NEU", majority=("N",))
        for actor in range(1001, 1005)
    )

    first = build_claim4_manifest(rows, actor_count=3, seed=20260728)
    second = build_claim4_manifest(list(reversed(rows)), actor_count=3, seed=20260728)

    assert first == second
    assert len(first) == 3
    assert len({entry.actor_id for entry in first}) == 3
    assert all(sum(entry.distribution.values()) == pytest.approx(1.0) for entry in first)
    assert all(entry.reference_provided_label == "N" for entry in first)
    assert all(entry.reference_majority_labels == ("N",) for entry in first)
    assert all(entry.reference_file_name != entry.file_name for entry in first)


def test_manifest_rejects_fewer_eligible_actors() -> None:
    rows = [
        _row(1001, 1),
        _row(1001, 2, label="N", code="NEU", majority=("N",)),
        _row(1002, 2, counts={"A": 5, "D": 0, "F": 0, "H": 0, "N": 0, "S": 0}),
    ]
    with pytest.raises(ContractError, match="insufficient eligible CREMA-D actors"):
        build_claim4_manifest(rows, actor_count=2, seed=1)


def test_lfs_pointer_and_atomic_cache_integrity(tmp_path: Path) -> None:
    payload = b"valid wav bytes"
    pointer = parse_lfs_pointer(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
        f"size {len(payload)}\n"
    )
    item = Claim4ManifestItem(
        actor_id="1001",
        row_id=1,
        file_name="1001_IEO_HAP_XX",
        provided_label="H",
        distribution={"p_angry": 0.4, "p_happy": 0.2, "p_sad": 0.0, "p_surprise": 0.0, "p_neutral": 0.4},
        reference_row_id=2,
        reference_file_name="1001_IEO_NEU_XX",
        reference_provided_label="N",
        reference_majority_labels=("N",),
    )

    target = ensure_lfs_wav(
        cache_root=tmp_path,
        item=item,
        pointer=pointer,
        fetch_content=lambda: payload,
    )
    assert target.read_bytes() == payload
    assert ensure_lfs_wav(
        cache_root=tmp_path,
        item=item,
        pointer=pointer,
        fetch_content=lambda: pytest.fail("verified cache must not fetch again"),
    ) == target


def test_lfs_integrity_failure_never_promotes_cache(tmp_path: Path) -> None:
    item = Claim4ManifestItem(
        actor_id="1001",
        row_id=1,
        file_name="1001_IEO_HAP_XX",
        provided_label="H",
        distribution={"p_angry": 0.4, "p_happy": 0.2, "p_sad": 0.0, "p_surprise": 0.0, "p_neutral": 0.4},
        reference_row_id=2,
        reference_file_name="1001_IEO_NEU_XX",
        reference_provided_label="N",
        reference_majority_labels=("N",),
    )
    pointer = LfsPointer(oid_sha256="0" * 64, size_bytes=3)
    with pytest.raises(ContractError, match="sha256 mismatch"):
        ensure_lfs_wav(cache_root=tmp_path, item=item, pointer=pointer, fetch_content=lambda: b"bad")
    assert not list(tmp_path.rglob("*.wav"))


def test_lfs_cache_rejects_symlinked_roots_parents_targets_and_traversal(tmp_path: Path) -> None:
    payload = b"valid wav bytes"
    pointer = LfsPointer(hashlib.sha256(payload).hexdigest(), len(payload))
    item = Claim4ManifestItem(
        actor_id="1001", row_id=1, file_name="1001_IEO_HAP_XX", provided_label="H",
        distribution={"p_angry": 0.4, "p_happy": 0.2, "p_sad": 0.0, "p_surprise": 0.0, "p_neutral": 0.4},
        reference_row_id=2, reference_file_name="1001_IEO_NEU_XX", reference_provided_label="N",
        reference_majority_labels=("N",),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractError, match="cache root symlink"):
        ensure_lfs_wav(cache_root=linked_root, item=item, pointer=pointer, fetch_content=lambda: payload)

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractError, match="cache root symlink"):
        ensure_lfs_wav(cache_root=linked_parent / "cache", item=item, pointer=pointer, fetch_content=lambda: payload)

    with pytest.raises(ContractError, match="unsafe CREMA-D cache root"):
        ensure_lfs_wav(
            cache_root=tmp_path / "safe" / ".." / "escaped",
            item=item, pointer=pointer, fetch_content=lambda: payload,
        )

    cache_root = tmp_path / "cache"
    target_parent = cache_root / item.actor_id
    target_parent.mkdir(parents=True)
    (target_parent / item.wav_name).symlink_to(outside / "untrusted.wav")
    with pytest.raises(ContractError, match="cache target symlink"):
        ensure_lfs_wav(cache_root=cache_root, item=item, pointer=pointer, fetch_content=lambda: payload)


def test_lfs_pointer_rejects_non_pointer_content() -> None:
    with pytest.raises(ContractError, match="unexpected version header"):
        parse_lfs_pointer(b"not a pointer")


def test_eligibility_rejects_one_non_neutral_plus_neutral_and_tied_non_neutral() -> None:
    one_non_neutral = _row(1001, 1, counts={"A": 2, "D": 0, "F": 0, "H": 0, "N": 3, "S": 0})
    tied_non_neutral = _row(1001, 2, counts={"A": 2, "D": 0, "F": 0, "H": 2, "N": 1, "S": 0})
    assert not is_claim4_eligible(one_non_neutral)
    assert not is_claim4_eligible(tied_non_neutral)
