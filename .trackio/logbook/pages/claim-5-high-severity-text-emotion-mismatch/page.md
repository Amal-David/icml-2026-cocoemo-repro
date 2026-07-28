# Claim 5: High-severity text-emotion mismatch


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_fb71ebb89468", "created_at": "2026-07-27T19:35:43+00:00", "title": "Claim and preregistered test"}
-->
**Claim.** Section 4.3, Table 3, and Appendices D.2/E.4 report that CosyVoice2 at alpha 6 achieves E-SIM 0.862, TEP 0.504, S-SIM 0.892, WER 8.74, and N-MOS 4.40 on high text-emotion mismatch; the instruction baseline reports 0.843, 0.436, 0.887, 4.01, and 4.23.

**Documentary finding.** The exact Table 3 experiment is not publicly reconstructible. The [paper](https://arxiv.org/abs/2602.03420) does not report the high-severity subset IDs, count, split seed, complete text-scoring implementation, or frozen evaluator revisions. Its method text also points to inconsistent text-emotion model provenance. We therefore do not relabel a public-data experiment as an exact Table 3 reproduction.

**Frozen public mechanism proxy.** [Commit ece947f](https://github.com/Amal-David/icml-2026-cocoemo-repro/commit/ece947f) preregisters an author-owned **categorical semantic-conflict** test on [RAVDESS](https://zenodo.org/records/1188976); it does not estimate valence-arousal severity and is never called “high mismatch.” The full design has 24 actors x 4 target emotions x 8 arms = 768 generated WAVs. Each actor/target cell uses the same neutral speaker reference and target-emotion source, paired with one of 16 frozen text families balanced across six actors.

The eight arms are no steering, native instruction, CoCoEmo alpha 3, CoCoEmo alpha 6, wrong-target alpha 6, negative-target alpha 6, norm-matched random alpha 6, and a flow-side emotional-reference control. The 32-WAV canary renders one actor while building the full 96-source leave-one-actor-out emotion-prototype bank, so it exercises the real evaluation path without producing claim evidence. Both configurations pin native 24 kHz [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B), the released top-two attention sites (layers 17 and 14), exact vector hashes, the text-catalog hash, output counts, and evaluator revisions.

**Decision rule.** Primary evidence is paired target-emotion probability and cosine similarity to leave-one-actor-out target-emotion prototypes. [WavLM speaker verification](https://huggingface.co/microsoft/wavlm-base-plus-sv) and [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) word error rate are frozen quality gates. Inference resamples the 24 actors as the sole independence units for 10,000 replicates; global effects first average each actor's four balanced observed cells. No missing output, evaluator failure, repeated waveform, partial retry directory, or sample-rate drift is accepted.

No GPU result or directional verdict is reported yet. The frozen protocol and its controls are implementation evidence only until a terminal [Hugging Face Job](https://huggingface.co/docs/huggingface_hub/guides/jobs) produces the complete artifact ledger through OpenResearch.

The reviewed but unpublished listening-study implementation is frozen at [GitHub commit 5a3af9a](https://github.com/Amal-David/icml-2026-cocoemo-repro/commit/5a3af9a). It validates media hashes and licenses, counterbalances conditions, stores no direct identifiers, and applies preregistered attention and playback-telemetry exclusions before crossed participant/content bootstrap analysis. It deliberately includes no audio, recruitment, ethics approval, participant data, or naturalness result.

**Access boundary.** Exact Table 3 verification requires licensed [IEMOCAP](https://sail.usc.edu/iemocap/) plus the authors' unreleased mismatch manifest and scorer provenance. This RAVDESS experiment is explicitly a public mechanism proxy. Perceived target emotion and naturalness remain unverified until the blinded study is lawfully deployed and analyzed.
