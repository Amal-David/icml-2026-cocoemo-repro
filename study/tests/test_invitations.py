from __future__ import annotations

import os

import pytest

from study.generate_invitations import generate, write_outputs
from study.protocol import ProtocolError, load_invitations


def test_frozen_seed_assigns_complete_nonsequential_group_ranks(monkeypatch, tmp_path) -> None:
    counter = iter(range(36))
    monkeypatch.setattr("study.generate_invitations.secrets.token_urlsafe", lambda _: f"code-{next(counter)}")
    codes, hashes = generate(secret="h" * 40, priority_seed="p" * 40)
    assert len(codes) == 36
    assert {row["priority_seed_commitment"] for row in hashes}
    first_group = [row["priority"] for row in hashes if row["group"] == 0]
    assert sorted(first_group) == list(range(12))
    # Rows are rank ordered, so compare priority in original code-generation order.
    priority_by_digest = {row["code_hmac"]: row["priority"] for row in hashes}
    from study.protocol import hmac_code

    generated_order = [priority_by_digest[hmac_code(code, "h" * 40)] for code in codes[:12]]
    assert generated_order != list(range(12))
    invitation_path = tmp_path / "invitations.json"
    invitation_path.write_text(__import__("json").dumps(hashes), encoding="utf-8")
    os.chmod(tmp_path, 0o700)
    assert len(load_invitations(invitation_path)) == 36


def test_invitation_outputs_are_private_exclusive_and_no_follow(tmp_path) -> None:
    os.chmod(tmp_path, 0o700)
    hashes = [{"code_hmac": "a" * 64, "group": 0, "priority": 0}]
    codes = ["one-time-code"]
    hashes_path = tmp_path / "hashes.json"
    codes_path = tmp_path / "codes.txt"
    write_outputs(hashes_output=hashes_path, codes_output=codes_path, codes=codes, hashes=hashes)
    assert hashes_path.stat().st_mode & 0o777 == 0o600
    assert codes_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ProtocolError, match="already exists"):
        write_outputs(hashes_output=hashes_path, codes_output=tmp_path / "new-codes.txt", codes=codes, hashes=hashes)
    target = tmp_path / "target.txt"
    target.write_text("keep", encoding="utf-8")
    symlink = tmp_path / "symlink-codes.txt"
    symlink.symlink_to(target)
    with pytest.raises(ProtocolError, match="already exists"):
        write_outputs(hashes_output=tmp_path / "new-hashes.json", codes_output=symlink, codes=codes, hashes=hashes)
    assert target.read_text(encoding="utf-8") == "keep"
