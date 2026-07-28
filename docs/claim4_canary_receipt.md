# Claim 4 Canary Receipt

`receipts/claim4-runtime-canary.json` is intentionally committed as a pending
template on the base branch. Claim 4 full mode refuses to start CUDA, source
retrieval, or model preparation until that same path is replaced in an ORX
child commit with a terminal `complete` receipt.

Run the hash-locked canary configuration on a GPU Job first. After the Job has
persisted public-safe artifacts and its remote `COMPLETE.json`, update only the
receipt in the ORX child with:

- `status: complete`;
- the unchanged file SHA-256 values of the current full and canary configurations,
  plus the canary's canonical `ReproConfig.digest` used in the artifact path;
- the immutable repository commit that ran the canary and its exact run name;
- the terminal Hugging Face Job or OpenResearch Job URL;
- the immutable Hugging Face dataset commit oid and the exact public remote
  `COMPLETE.json` URL under `/resolve/<hub-oid>/runs/<repository-commit>-<canary-config-sha>-<run-name>/`;
- the SHA-256 of that remote completion marker.

Commit the receipt update. Full mode checks that the receipt bytes equal the
version at `HEAD`, verifies both configuration files and both configuration
hashes against `HEAD`, requires HTTPS terminal and immutable completion URLs,
fetches and SHA-256-verifies the remote marker before CUDA, data, or model
work, and validates its atomic-completion identity fields. It records the
receipt SHA-256 and commits in its summary.
The receipt intentionally does not enter the full configuration hash: a
terminal canary must be able to replace the pending template without
invalidating the canary's base-config lock.

Only derived metrics and hashes may be uploaded. Source CREMA-D audio,
generated audio, model files, and caches remain excluded, so this workflow does
not rely on redistributing generated speech.
