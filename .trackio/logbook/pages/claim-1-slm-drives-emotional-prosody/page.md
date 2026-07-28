# Claim 1: SLM drives emotional prosody


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_abaaa007e862", "created_at": "2026-07-28T04:10:39+00:00", "title": "Claim and preregistered test"}
-->
**Claim.** Section 2.1, Figure 2, and Table 1 of [arXiv:2602.03420v2](https://arxiv.org/pdf/2602.03420v2) report that the speech-language model is the primary driver of emotional prosody in [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/tree/eec1ae6c79877dbd9379285cf8789c9e0879293d). Across 300 cross-conditioned samples, SLM-driven versus flow-driven speech has F0 CCC 0.109 versus 0.305, energy CCC 0.308 versus 0.737, and speaking-rate standard deviation 0.691 versus 0.518.

**Preregistered public-data test.** The executable proxy uses 20 distinct [RAVDESS](https://zenodo.org/records/1188976) actors, four target emotions, and five matched conditions: neutral-both, SLM-driven, Flow-driven, emotional-both, and a no-fixed-point permuted-emotion control. It produces 400 WAVs. Actor-cluster bootstrap intervals test three signed SLM-minus-Flow differences: F0 CCC below zero, energy CCC below zero, and speaking-rate-proxy dispersion above zero. All three intervals wholly in the predicted direction support the claim; any interval wholly reversed fails it; every other result is inconclusive. The paper numbers are descriptive context only because its source set and acoustic extraction details are unspecified.

**Status.** Protocol frozen at [reproduction commit eb1a603](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/eb1a603). Empirical evidence awaits the substantive Hugging Face GPU Job through OpenResearch. Source code is pinned to [official CoCoEmo commit dcc3191](https://github.com/wsssy/CoCoEmo/tree/dcc319148dd2e0e0f3039e06e81e0f770d108f08).


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_85a41b56c996", "created_at": "2026-07-28T04:10:42+00:00", "title": "Release audit and causal replacement"}
-->
The released cross-modal helper cannot implement the paper protocol: it extracts speech tokens from audio and feeds them directly to Flow, bypassing the speech-language model, and advertises an unimplemented token-only method. A direct rerun therefore cannot test the causal SLM claim.

At [reproduction commit eb1a603](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/eb1a603), the replacement uses official CosyVoice2 frontend fields and model.tts directly. Strict contracts isolate the SLM and Flow conditioning bundles, require an explicit empty source_speech_token, reset every matched arm to the same SHA256-derived draw seed, and record tensor hashes. A thread-safe per-render trace must observe exactly one model.llm.inference call; the planned GPU smoke must still prove that contract against the real pinned runtime.

The official RAVDESS archive is pinned by URL, size, and MD5. An extracted cache is accepted only with a matching deterministic SHA-256 manifest receipt covering every file. The 400-output scaled run is directional public-data evidence, not an exact reproduction of the unidentified 300-sample Table 1 set. Naturalness and perceived emotion remain reserved for the separately preregistered blinded listening study.
