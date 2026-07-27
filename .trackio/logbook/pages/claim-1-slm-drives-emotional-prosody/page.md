# Claim 1: SLM drives emotional prosody


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_aa20efbe4e0f", "created_at": "2026-07-27T19:35:41+00:00", "title": "Claim and preregistered test"}
-->
**Claim.** Section 2.1, Figure 2, and Table 1 of [arXiv:2602.03420v2](https://arxiv.org/pdf/2602.03420v2) report that the speech-language model is the primary driver of emotional prosody in [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/tree/eec1ae6c79877dbd9379285cf8789c9e0879293d). Across 300 cross-conditioned samples, SLM-driven versus flow-driven speech has F0 CCC 0.109 versus 0.305, energy CCC 0.308 versus 0.737, and speaking-rate standard deviation 0.691 versus 0.518.

**Preregistered test.** Run paired SLM-only, flow-only, both-emotional, and all-neutral conditions with a permuted-label control. Pass requires all three reported directions to agree and the paper estimates to fall inside paired 95% bootstrap intervals; directional agreement outside those intervals is partial; a significant reversal fails. The paper does not identify the Table 1 source set, so a public [RAVDESS](https://zenodo.org/records/1188976) run is directional rather than exact.

**Status.** Empirical evidence pending a substantive Hugging Face GPU Job. Source code is pinned to [official CoCoEmo commit dcc3191](https://github.com/wsssy/CoCoEmo/tree/dcc319148dd2e0e0f3039e06e81e0f770d108f08).


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1b6618d24ee7", "created_at": "2026-07-27T20:01:26+00:00", "title": "Released-protocol audit and causal replacement"}
-->
The released cross-modal helper cannot implement the paper protocol: it extracts speech tokens from audio and feeds them directly to Flow, bypassing the speech-language model, and it advertises an extract_tokens_only method that is not implemented. This makes a simple rerun incapable of testing whether the SLM is the primary prosody driver.

At [reproduction commit 47f6978](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/47f69785e60b6b1af254dbeaf5d8ca062f515604), the replacement uses official CosyVoice2 frontend_zero_shot fields and model.tts directly. Unit contracts prove the SLM-driven condition changes only llm_prompt_speech_token and its length in the strict primary analysis, while the Flow-driven condition changes only flow_prompt_speech_token, prompt_speech_feat, their lengths, and flow_embedding. source_speech_token remains empty so the SLM path must execute. Tensor-field hashes are recorded for every render.

The legal public RAVDESS design contains exactly 96 independent complete actor x statement x repetition groups with neutral, happy, sad, angry, and surprise references. The harness rejects incomplete groups and rejects reused or padded groups presented as N=300. A full directional run requires 1,440 WAVs across SLM-driven, Flow-driven, and neutral-control conditions; GPU evidence is pending OpenResearch authorization.
