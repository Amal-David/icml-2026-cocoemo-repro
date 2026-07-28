"""Generate one-time invitation codes and a private HMAC-only lookup file."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from pathlib import Path

from study.protocol import ProtocolError, hmac_code
from study.storage import write_new_private_file


def _locked_priority(*, priority_seed: str, group: int, code_hmac: str) -> bytes:
    return hmac.new(priority_seed.encode("utf-8"), f"{group}:{code_hmac}".encode("utf-8"), hashlib.sha256).digest()


def generate(*, secret: str, priority_seed: str, per_group: int = 12) -> tuple[list[str], list[dict[str, object]]]:
    if per_group < 12:
        raise ProtocolError("the preregistration requires at least 12 codes per group")
    if len(priority_seed) < 32:
        raise ProtocolError("priority seed must contain at least 32 characters")
    codes: list[str] = []
    candidates: dict[int, list[tuple[str, str]]] = {0: [], 1: [], 2: []}
    commitment = hashlib.sha256(priority_seed.encode("utf-8")).hexdigest()
    for group in range(3):
        for _ in range(per_group):
            code = secrets.token_urlsafe(18)
            codes.append(code)
            candidates[group].append((code, hmac_code(code, secret)))
    hashes: list[dict[str, object]] = []
    for group, rows in candidates.items():
        for priority, (_, digest) in enumerate(
            sorted(rows, key=lambda row: _locked_priority(priority_seed=priority_seed, group=group, code_hmac=row[1]))
        ):
            hashes.append(
                {
                    "code_hmac": digest,
                    "group": group,
                    "priority": priority,
                    "priority_seed_commitment": commitment,
                }
            )
    return codes, hashes


def write_outputs(*, hashes_output: Path, codes_output: Path, codes: list[str], hashes: list[dict[str, object]]) -> None:
    """Persist invitation artifacts atomically without exposing or replacing files."""

    write_new_private_file(hashes_output, json.dumps(hashes, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    try:
        write_new_private_file(codes_output, ("\n".join(codes) + "\n").encode("utf-8"))
    except BaseException:
        # A partially created lookup would make future runs ambiguous. The code
        # file is never replaced; caller must choose fresh absent output paths.
        try:
            hashes_output.unlink()
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secret", required=True, help="not persisted by this command")
    parser.add_argument("--priority-seed", required=True, help="freeze privately, then destroy after recruitment closes")
    parser.add_argument("--hashes-output", type=Path, required=True)
    parser.add_argument("--codes-output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=12)
    args = parser.parse_args()
    codes, hashes = generate(secret=args.secret, priority_seed=args.priority_seed, per_group=args.per_group)
    write_outputs(hashes_output=args.hashes_output, codes_output=args.codes_output, codes=codes, hashes=hashes)


if __name__ == "__main__":
    main()
