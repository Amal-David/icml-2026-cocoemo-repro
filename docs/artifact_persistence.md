# Durable Artifact Contract

Every harness invocation has a deterministic public destination:

```text
https://huggingface.co/datasets/amal-david/cocoemo-repro-artifacts/tree/<hub-commit-oid>/runs/<commit>-<config-sha256>-<run-name>
```

Only a strict top-level report schema is eligible for upload: fixed scientific
reports (`summary.json`, `provenance.json`, `resolved_config.json`,
`artifact_manifest.json`, and `EVAL.md`), known derived JSON/JSONL/Markdown/CSV
or hash reports, and validated figures directly beneath `figures/`. The
uploader does not recurse through run directories. It excludes source audio,
raw embeddings, activations, tensors, model weights, source datasets, caches,
credentials, and all other nested paths. Text and JSON are parsed for
credential-like strings, raw representation fields, and suspicious
binary/base64 payloads, and every public file is size-capped. Claim 5 retains
its frozen reference embeddings locally for retry integrity and publishes only
the corresponding count-and-hash manifest.

The uploader snapshots and hashes the candidate bytes once, then submits the
reports, `hub_artifact_manifest.json`, and remote `COMPLETE.json` in one
parent-guarded Hugging Face commit. A concurrent head change rejects the write;
there is no partial artifact directory. Consumers must use the commit-pinned
URLs recorded in the local `COMPLETE.json`, and accept a run only when that
same immutable commit contains a completion marker matching its manifest hash.
The summary remains `pending_atomic_hub_commit`; it is never rewritten after
the evidence result is frozen.

A failed transfer writes no local completion marker. `LOCAL_READY.json` allows
a fresh Job to retry the exact immutable scientific output without rerunning
the experiment. Before retrying, the runner validates every path and hash in
`artifact_manifest.json`. Mutable `PERSISTENCE_ATTEMPTS.json` is the only place
that records retry diagnostics and immutable Hub commit links.

Non-preflight stages require `HF_TOKEN`, a clean Git worktree, and a remote
write-permission preflight before synthesis or evaluation begins. When no
authoritative non-mutating permission check is available, the preflight writes
and immediately deletes a deterministic tiny canary under
`permission-canaries/`; both commit identifiers are recorded in persistence
diagnostics. A preflight can complete locally without a token and remains
`LOCAL_READY` until it is uploaded. Jobs should pass the token as a secret,
never as a logged environment value. Provenance records only a dirty-file count
and state hash, plus allow-listed job/OpenResearch backend, hardware, and image
fields; it never reads or serializes `HF_TOKEN` or local filenames.
