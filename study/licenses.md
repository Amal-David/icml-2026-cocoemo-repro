# Study media and licensing gate

No participant-facing media is included in this repository. The manifest
template contains paths and zero hashes only; it must fail media verification.

Before a private study deployment, every generated WAV must be listed in the
frozen manifest with:

- exact generated-file SHA-256 and decoded duration;
- original, license-cleared spoken text;
- pinned backbone and model revision;
- vector identifier and vector SHA-256;
- non-identifying reference-voice identifier and written-release receipt ID;
- immutable generator commit;
- a license basis explicitly documenting consent and research redistribution.

The validator rejects provenance that names ESD, IEMOCAP, CREMA-D, or RAVDESS,
and it rejects a license basis that lacks both consent and redistribution
language. This is a hard distribution boundary: no raw ESD, IEMOCAP, CREMA-D,
RAVDESS, or other evaluation-corpus recording may be placed in the study app,
manifest, Bucket, or public repository.

The written voice release and any legal/ethics records are not committed here.
Their private receipt identifiers are enough for the frozen manifest; the owner
must confirm that the actual release covers the intended study and deployment.
