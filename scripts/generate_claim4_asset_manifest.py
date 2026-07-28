#!/usr/bin/env python3
"""Freeze Git-LFS pointers for the selected Claim 4 CREMA-D WAV files.

This maintenance command is intentionally separate from the runtime.  It
retrieves only immutable raw pointer blobs at a pinned commit, computes the
Git blob SHA-1 locally, and writes no audio bytes.  GPU jobs consume the
committed output and never ask GitHub's Contents or Blob REST endpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from repro.claim4_protocol import (
    _CLAIM4_ASSET_SCHEMA,
    _CLAIM4_REPOSITORY,
    _expected_claim4_asset_url,
    _expected_claim4_pointer_url,
    build_claim4_manifest,
    git_blob_sha1,
    parse_lfs_pointer,
)
from repro.cremad_selection import load_audio_vote_rows


def _fetch_pointer(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "cocoemo-repro-manifest/1"})
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError as exc:
            last_error = exc
            if attempt == 6:
                break
            time.sleep(float(attempt))
    assert last_error is not None
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--votes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pointer-cache",
        type=Path,
        help="optional local cache for immutable pointer blobs; never stores audio",
    )
    args = parser.parse_args()

    rows = load_audio_vote_rows(args.votes)
    selection = build_claim4_manifest(rows, actor_count=60, seed=20260728)
    filenames = sorted(
        {f"{file_name}.wav" for item in selection for file_name in (item.file_name, item.reference_file_name)}
    )
    if len(filenames) != 120:
        raise SystemExit(f"expected 120 unique selected WAVs, found {len(filenames)}")
    assets = []
    for index, filename in enumerate(filenames, start=1):
        pointer_url = _expected_claim4_pointer_url(revision=args.revision, filename=filename)
        download_url = _expected_claim4_asset_url(revision=args.revision, filename=filename)
        cache_path = args.pointer_cache / filename if args.pointer_cache is not None else None
        if cache_path is not None and cache_path.is_file():
            pointer_payload = cache_path.read_bytes()
        else:
            pointer_payload = _fetch_pointer(pointer_url)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(pointer_payload)
        pointer = parse_lfs_pointer(pointer_payload)
        assets.append(
            {
                "filename": filename,
                "repository_path": f"AudioWAV/{filename}",
                "git_blob_sha1": git_blob_sha1(pointer_payload),
                "lfs_oid_sha256": pointer.oid_sha256,
                "size_bytes": pointer.size_bytes,
                "download_url": download_url,
            }
        )
        print(f"[{index}/{len(filenames)}] {filename}", file=sys.stderr)
    payload = {
        "schema_version": _CLAIM4_ASSET_SCHEMA,
        "source": {"repository": _CLAIM4_REPOSITORY, "revision": args.revision},
        "assets": assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
