# CoCoEmo independent listening study preregistration

Status: draft, not yet opened to participants. The protocol, manifest hash,
stimulus hashes, exclusions, and analysis code must be frozen in a tagged commit
before the first valid response.

## Scope

This is an independent blinded evaluation of CoCoEmo outputs. It is not described
as a reproduction of the paper N-MOS unless the original stimuli, sampling, and
rating protocol become available.

The study tests two tracks:

- 12 mixed-emotion content families at CoCoEmo alpha 5;
- 12 high text-emotion-mismatch content families at CoCoEmo alpha 6.

Each family has three conditions: alpha-zero control, instruction baseline, and
CoCoEmo. This yields 72 experimental stimuli.

## Participants and assignment

Recruit 36 adults and retain the first 10 valid responses in each of three
preassigned groups, for 30 retained raters. Twelve one-time invitation codes are
generated per group; an HMAC-ranked priority generated from a frozen private
seed resolves simultaneous valid submissions without post-hoc outcome selection.

Each rater receives 24 experimental trials and 6 quality trials. They hear each
content family once and exactly 8 experimental trials from each condition. For
zero-indexed group `g` and content family `j`, the presented condition is:

```text
(g + (j mod 3)) mod 3
```

Across 30 retained raters, every one of the 72 stimuli receives 10 ratings.

## Outcomes

Primary outcomes:

1. Mixed emotion: Jensen-Shannon distance between the 100-point listener
   allocation and the target allocation; lower is better.
2. Mismatch: target-emotion allocation and dominant-emotion hit rate; higher is
   better.
3. Naturalness: 1-5 ACR difference. CoCoEmo noninferiority margin is -0.35.

Secondary outcomes use Holm correction. The analysis uses crossed participant
and content-family bootstrap intervals, with a mixed-effects sensitivity model.

## Trial response

Responses remain locked until the stimulus has been played. Every experimental
trial records:

- naturalness on a 1-5 absolute category scale;
- one dominant emotion from angry, happy, sad, surprised, or neutral;
- integer allocations for those five emotions that sum to exactly 100.

Playback telemetry is a quality signal, not proof of listening.

## Quality gates

The six quality trials comprise two explicit audio instructions, two hidden
duplicate stimuli, and two calibration checks. A participant is excluded only
under these predeclared rules:

- consent or adult confirmation is absent;
- the response is incomplete or an allocation does not sum to 100;
- either explicit instruction item is incorrect;
- fewer than 22 of 24 experimental stimuli satisfy the locked listening-time
  threshold;
- the duplicate dominant emotion disagrees or the duplicate allocation exceeds
  the locked distance threshold;
- total completion time is below the locked minimum.

Every exclusion and failed balance check is reported. No record is silently
dropped.

## Privacy and storage

No name, email, Hugging Face identity, IP address, raw user agent, microphone,
free text, demographic field, or payment identifier is collected. Compensation
records, if any, are separate from study responses.

The public study interface stores only an HMAC of a one-time invitation code in a
private Hugging Face Bucket. Each submission is an immutable JSON object. The
HMAC secret and withdrawal lookup are destroyed after the declared withdrawal
window, leaving the retained export effectively anonymous.

The research team must confirm applicable ethics, consent, and compensation
requirements before recruitment.

## Licensing gate

Study stimuli use original text and a reference voice from a consenting adult
with a written release for this public research use. No ESD, IEMOCAP, CREMA-D,
RAVDESS, or other evaluation-corpus recording is redistributed through the study
interface. Every generated stimulus records model, vector, reference-voice,
generator-commit, and license provenance.

## Interpretation

Results are reported as support, contradiction, or inconclusive evidence for the
relevant CoCoEmo claim. Automated metrics and agent judgments are never presented
as naturalness or listening evidence.
