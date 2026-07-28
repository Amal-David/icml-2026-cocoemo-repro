# Claim 4: Mixed-emotion controllability and quality


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2299c5324cb7", "created_at": "2026-07-27T19:35:42+00:00", "title": "Documentary falsification and preregistered test"}
-->
**Claim audit.** The challenge prompt combines numbers that do not occur in one Table 2 row of [arXiv:2602.03420v2](https://arxiv.org/pdf/2602.03420v2). On CREMA-D, TEP 0.335 and Spearman rho 0.319 belong to CoCoEmo with instruction 1 at alpha 5, whose N-MOS is 3.00. N-MOS 3.96 belongs to bare CoCoEmo at alpha 5, while 4.25 belongs to bare CoCoEmo at alpha 3. The composite numerical claim therefore **fails documentary audit** and will be tested row by row.

**Preregistered test.** Reproduce each named CREMA-D condition separately, then compare genuine mixed vectors with instruction-only, dominant-only, label-shuffled, and norm-matched-random controls using paper-faithful and frozen independent evaluators. Report paired bootstrap intervals and full denominators. Naturalness requires a real blinded listening study; no automatic score or agent judgment will substitute for N-MOS.

**Scope.** The public [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) subset is feasible. Exact out-of-distribution confirmation on [IEMOCAP](https://sail.usc.edu/iemocap/) remains blocked without licensed access.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_dcaafaa6f936", "created_at": "2026-07-27T19:58:39+00:00", "title": "CREMA-D selection-count falsification"}
-->
At [clean reproduction commit 19720d4](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/19720d49a4cb1ff682e33f9fb320034ac9e52384), the fixed runner downloaded the official [CREMA-D vote table at 1658cd3](https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc), verified SHA-256 774d9d759cb5caf2542758a39eca8c0f1d2777f9987ff5d5adf8066d192f377b, and independently recomputed the published selection rule.

Results: 7,442 audio-only rows; 791 rows under the closest paper-consistent supported-disagreement filter; 14 rows under the literal more-than-two unique non-neutral categories reading; and 323 rows under the at-least-three non-neutral votes reading. None yields the paper-reported 772. No records were discarded to force agreement.

The transparent 791-row local-audit proxy, selection variants, and provenance are published at [amal-david/cocoemo-repro-artifacts](https://huggingface.co/datasets/amal-david/cocoemo-repro-artifacts/tree/main/local/claim4-metadata/19720d49a4cb). This legacy local-audit path is documentary support only: new Job evidence is accepted only from a deterministic `runs/` path with a remote `COMPLETE.json` written after its public-safe manifest. It is not a substitute for the required Hugging Face GPU mixed-emotion experiment.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bc561940f347", "created_at": "2026-07-28T03:23:52+00:00", "title": "Neutral-mass release audit"}
-->
**Release-conformance finding.** At [reproduction commit 42d6554](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/42d655441096f845c42676fcdf0887fdc127fed6), an executable AST-and-vector audit shows that the released mixed-synthesis path loads only angry, happy, sad, and surprise vectors and never reads `p_neutral`. A valid fixed probe with 40% neutral mass therefore produces the unrenormalized non-neutral sum: it is exactly identical to an explicit zero-neutral-vector interpretation and has norm ratio 0.600000025 versus renormalizing the four available directions (L2 difference 0.376424 at CosyVoice2 attention-output layer 17). The [hashed JSON evidence](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/42d655441096f845c42676fcdf0887fdc127fed6/docs/evidence/claim4_mixing_audit.json) records the pinned source hash and all four released vector hashes. This falsifies faithful five-way composition in the release when neutral probability is positive; it does not by itself measure generated-speech quality.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_471682e89f61", "created_at": "2026-07-28T04:58:05+00:00", "title": "Frozen public proxy and GPU canary"}
-->
At [reproduction commit ce76d30](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/ce76d301ac986a47fd06eeba88533f6f7a99a383), the public Claim 4 proxy is frozen before synthesis. The pinned CREMA-D vote table yields 118 proxy-eligible rows across 60 actors under an explicit, non-paper-authored rule: supported disagreement, positive neutral mass, at least two active angry/happy/sad labels, and a unique non-neutral maximum. Exactly one item per actor is selected by SHA-256 rank, with a distinct same-actor reference whose filename label and unique majority vote are neutral. Both target and reference WAVs must match their pinned Git-LFS object hash and size.

The full protocol is 60 actors x seven matched arms = 420 WAVs. The [non-evidence canary configuration](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/ce76d301ac986a47fd06eeba88533f6f7a99a383/configs/claim4-cremad-mixed-canary.yaml) hash-locks the [full configuration](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/ce76d301ac986a47fd06eeba88533f6f7a99a383/configs/claim4-cremad-mixed-directional.yaml) but generates only one actor x seven arms. It requires CUDA, pinned CosyVoice source/model preparation, per-arm RNG resets, neutral-reference transcript fidelity, complete finite WAV accounting, and the one expected released-mix/explicit-zero-neutral hash equivalence. A full verdict requires exactly 60 actors and 420 metric rows.

The preregistered automated outcome compares proportion Spearman rho and H-rate against both norm-matched shuffled and random controls, checks non-inferiority to a dominant-only control, and gates [WavLM speaker similarity at 0a23162](https://huggingface.co/microsoft/wavlm-base-sv/tree/0a23162ffc49adcf42bdf836a00cb2eb45af3601) and [Whisper large-v3 WER at 06f233f](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1). Emotion2Vec is pinned to [emotion2vec/emotion2vec_plus_large at 6c303ba](https://huggingface.co/emotion2vec/emotion2vec_plus_large/tree/6c303ba987b86b93193de93e34bb2b077a6bedc4). GPU evidence remains pending OpenResearch and Hugging Face Jobs authorization; none of these metrics substitutes for blinded N-MOS ratings.

The blinded 30-trial listening protocol, consent text, frozen-manifest contract, private HMAC invitation flow, immutable response storage, exclusion rules, and crossed participant/content bootstrap analysis are preregistered in [GitHub commit 5a3af9a](https://github.com/Amal-David/icml-2026-cocoemo-repro/commit/5a3af9a). The package passed 18 focused tests and an independent security/statistical review, but it contains no stimuli, participants, ethics approval, or ratings. It is methodology, not Claim 4 evidence, until those prerequisites are satisfied.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_47af9010b3d2", "created_at": "2026-07-28T00:00:00+00:00", "title": "Licensing and terminal-canary gate"}
-->
The frozen Claim 4 configuration records the authoritative pinned identifiers and redistribution boundary for [CREMA-D at 1658cd3](https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc) ([ODbL/DbCL terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)), [CosyVoice2-0.5B at eec1ae6](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/tree/eec1ae6c79877dbd9379285cf8789c9e0879293d), [Emotion2Vec at 6c303ba](https://huggingface.co/emotion2vec/emotion2vec_plus_large/tree/6c303ba987b86b93193de93e34bb2b077a6bedc4), [WavLM at 0a23162](https://huggingface.co/microsoft/wavlm-base-sv/tree/0a23162ffc49adcf42bdf836a00cb2eb45af3601), and [Whisper at 06f233f](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1). The public uploader excludes all source audio, generated audio, model weights, and caches; only derived metrics and hashes are eligible. Thus the reproduction does not rely on redistributing generated speech.

The committed [pending canary receipt](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/ce76d301ac986a47fd06eeba88533f6f7a99a383/receipts/claim4-runtime-canary.json) blocks full mode before CUDA, data, or model work. It can be replaced only in an ORX child after a terminal GPU canary with the unchanged full/canary file hashes, canonical canary digest, canary source commit and run name, terminal Job URL, and immutable Hub commit plus remote `COMPLETE.json` URL and hash. The procedure is documented in [Claim 4 Canary Receipt](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/ce76d301ac986a47fd06eeba88533f6f7a99a383/docs/claim4_canary_receipt.md). It remains a prerequisite, not empirical evidence.
