# Frozen study protocol

This package implements the draft preregistration in
[`PREREGISTRATION.md`](PREREGISTRATION.md). It is not a participant launch
authorization and contains no generated stimuli or results.

## Freeze gate

Before the first valid submission, tag the repository commit and freeze:

- `protocol_version` `cocoemo-listening-v1`;
- the exact JSON manifest and its canonical SHA-256;
- every WAV byte hash and decoded duration;
- source model/vector/reference-voice/release/commit/license provenance;
- invitation-code HMAC lookup, group, and priority ordering;
- private Bucket mount, withdrawal procedure, and consent text.

The app rejects a different protocol version, missing or changed media,
non-private storage, weak/missing secrets, missing invitation hashes, and an
incomplete 72 experimental plus 6 quality manifest.

## Assignment and presentation

Each code is assigned to a fixed group 0, 1, or 2. A private, at-least-32
character priority seed ranks the HMAC code IDs within each group by HMAC; the
frozen lookup stores only ranks `0..n-1` and a SHA-256 seed commitment. This is
the locked random tie-break order for timestamp ties, not sequential generation
order. Experimental condition is `(group + item_index mod 3) mod 3`; per-code
HMAC sorting gives a deterministic schedule. Duplicate quality items resolve to
the participant's already assigned condition and are placed immediately after
their source. The four non-duplicate quality items are distributed through the
experimental schedule.

The rating form remains unavailable until the audio element reports completion.
Browser playback duration is stored as an attention signal, not proof that a
person listened. The 90% frozen-duration threshold is applied only to the
predeclared aggregate exclusion: fewer than 22 of 24 experimental trials meeting
that threshold excludes the completed record. Individual short-playback trials
remain stored and are never silently rejected at response time.

## Response contract

Every trial requires:

- naturalness as a whole number from 1 through 5;
- one dominant emotion: angry, happy, sad, surprised, or neutral;
- five whole-number allocations summing exactly to 100.

The server rejects fields for identity, IP/user-agent, microphone, free text,
or payment. The browser-carried session state has an HMAC over its assignment,
nonce, current index, and canonical response history, so a forged pointer or
edited history fails closed. The server creates one `0600` immutable JSON file
per HMAC ID using an atomic no-follow create; a later submission from the same
code is rejected.

## Attention and exclusion

The stored record reports every failure. The only exclusion reasons are those
in the preregistration: absent consent/adult confirmation, incomplete or invalid
answers, an incorrect explicit instruction, fewer than 22 locked experimental
playbacks, duplicate dominant/allocation inconsistency (L1 allocation distance
greater than 40), and completion under eight minutes. Calibration checks are
reported but are not additional exclusion criteria.

## Final analysis gate

`export_study.py` retains the earliest valid ten submissions per group, using
the frozen random priority only when timestamps tie, and includes all exclusions in its
export. `analyse_study.py` will not produce a final analysis unless there are
exactly 30 retained participants, ten per group, and ten ratings for every one
of the 72 frozen stimuli. It reports crossed participant/content-family
bootstrap intervals only; it does not manufacture an interpretive verdict.

The planned naturalness noninferiority contrast is CoCoEmo minus instruction
baseline on the mixed track, with margin `-0.35`. Mixed-emotion JSD and mismatch
target allocation/dominant hit contrasts use CoCoEmo minus alpha-zero. These are
pre-launch analysis definitions, not results or paper-reproduction claims.
