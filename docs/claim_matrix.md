# CoCoEmo claim matrix

Source lock:

- Paper: [arXiv:2602.03420v2](https://arxiv.org/pdf/2602.03420v2), 16 June 2026.
- Official code: [CoCoEmo at dcc3191](https://github.com/wsssy/CoCoEmo/tree/dcc319148dd2e0e0f3039e06e81e0f770d108f08).
- Primary backbone: [CosyVoice2-0.5B at eec1ae6](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/tree/eec1ae6c79877dbd9379285cf8789c9e0879293d).

## Claim 1: SLM drives emotional prosody

Paper anchor: Section 2.1, Figure 2, and Table 1. On 300 CosyVoice2 cross-conditioned samples, the paper reports lower F0 CCC for SLM-driven speech than flow-driven speech (`0.109` versus `0.305`), lower energy CCC (`0.308` versus `0.737`), and higher speaking-rate standard deviation (`0.691` versus `0.518`).

Required evidence: paired cross-conditioning with all-neutral and permuted-emotion controls, complete WAV accounting, per-sample F0/energy/rate, and actor-cluster bootstrap intervals. The paper does not identify the Table 1 source set, its feature extractor, alignment procedure, or speaking-rate definition, so a public RAVDESS test is directional rather than exact.

Executable protocol: `claim1-ravdess-gpu-smoke.yaml` is deliberately non-evidentiary (one group x one target x three causal conditions = 3 renders). `claim1-ravdess-scaled-directional.yaml` fixes a substantive public proxy: 20 explicitly listed groups from exactly 20 distinct RAVDESS actors x four non-neutral targets x five conditions (neutral-both, SLM-driven, Flow-driven, emotional-both, and a deterministic no-fixed-point permuted-emotion-both control) = 400 rendered WAVs and 80 group-emotion rows per condition. The older 1,440-WAV description was arithmetically incompatible with the non-neutral RAVDESS target set and is not used by the executable configuration.

Every render passes an explicit empty `source_speech_token` and is rejected unless the pinned CosyVoice `model.llm.inference` method is observed exactly once. The deterministic seed is derived from the base seed plus `(group_id, target_emotion)` and reset before every matched condition, so arm ordering cannot change decoding draws. A pinned maximum clipping fraction of `0.001` and finite-audio checks are enforced. Equal generated-WAV hashes remain a reported result rather than a failure.

The only pass/fail outcome for this public proxy is machine-readable signed directional support: all three actor-cluster bootstrap intervals must exclude zero in the paper's reported direction to yield `directional_supported`; any confidently reversed direction yields `directional_failed`; otherwise it is `directional_inconclusive`. Whether a paper contrast lies inside a proxy interval is retained as a diagnostic only and never changes that outcome.

## Claim 2: mean-difference steering vectors

Paper anchor: Section 3.1, Equation 6, Section 4.1, and Appendix D.1/Table 6. The paper constructs each direction as `mean(h_emotion) - mean(h_neutral)` using speaker- and transcript-matched pairs from ESD, RAVDESS, and CREMA-D, totaling 20,691 utterances.

Required evidence: pairing audit, speaker-disjoint splits, vector shapes and hashes, held-out control, and shuffled-label, speaker-mismatched, and norm-matched-random falsification vectors. Exact reconstruction is blocked until licensed ESD access is available.

## Claim 3: mid-to-late attention sites

Paper anchor: Section 2.2/Figure 3 and Section 4.1. The paper identifies CosyVoice2 layers 10-17 and IndexTTS2 layers 5-10 as highly separable, then selects CosyVoice2 layers 17 and 14 and IndexTTS2 layers 6, 8, and 1.

Required evidence: speaker-disjoint multiclass probes over all hook locations, permuted-label nulls, validation-only selection, non-selected-layer interventions, and the paper's WER quality constraint. IndexTTS2 remains blocked until its released adapter and steering-vector dimensions pass a GPU canary.

## Claim 4: mixed-emotion controllability and quality

Paper anchor: Sections 3.3 and 4.2, Table 2, and Appendix D.2. CREMA-D contributes 772 held-out multi-rater items and IEMOCAP 1,055 out-of-distribution items.

The challenge prompt's composite tuple is not a single Table 2 result. On CREMA-D, TEP `0.335` and Spearman rho `0.319` belong to `CoCoEmo (Ins1, alpha=5)`, whose N-MOS is `3.00`; N-MOS `3.96` belongs to bare `CoCoEmo (alpha=5)`, while `4.25` belongs to bare `CoCoEmo (alpha=3)`. This documentary inconsistency is a falsification finding; empirical results must be compared row by row.

Release-conformance finding: the released mixed-synthesis script defines only `angry`, `happy`, `sad`, and `surprise`; it never reads `p_neutral` and ships no neutral steering vector. Therefore, when a five-way target contains neutral mass, the released helper composes the unrenormalized non-neutral sum. This is exactly equivalent to treating the missing neutral direction as a zero vector, and attenuates the mixed direction by `1 - p_neutral` relative to a non-neutral-renormalized counterfactual. The deterministic source-and-vector audit is [`repro/claim4_mixing_audit.py`](../repro/claim4_mixing_audit.py); its generated evidence is [`claim4_mixing_audit.json`](evidence/claim4_mixing_audit.json).

Required evidence: every named Table 2 condition, mixed-label controls, paper-faithful and independent frozen evaluators, paired intervals, and actual blinded human ratings for naturalness. IEMOCAP results are blocked without licensed access.

## Claim 5: high text-emotion mismatch

Paper anchor: Section 4.3, Table 3, and Appendices D.2 and E.4. For CosyVoice2 at alpha 6, the paper reports E-SIM `0.862`, TEP `0.504`, S-SIM `0.892`, WER `8.74`, and N-MOS `4.40`; the instruction baseline reports `0.843`, `0.436`, `0.887`, `4.01`, and `4.23` respectively.

Required evidence: deterministic valence-arousal severity bins, complete paired output accounting, instruction, random, negative, and flow-only controls, plus blinded human ratings. Exact Table 3 verification is blocked without licensed IEMOCAP; a public-data mismatch test must be labeled a proxy.
