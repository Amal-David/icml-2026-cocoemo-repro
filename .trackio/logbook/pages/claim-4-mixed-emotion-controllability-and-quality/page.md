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

The complete local-audit artifact set, including the transparent 791-row proxy, selection variants, provenance, hashes, and COMPLETE marker, is published at [amal-david/cocoemo-repro-artifacts](https://huggingface.co/datasets/amal-david/cocoemo-repro-artifacts/tree/main/local/claim4-metadata/19720d49a4cb). This is released-data protocol evidence, not a substitute for the required Hugging Face GPU mixed-emotion experiment.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bc561940f347", "created_at": "2026-07-28T03:23:52+00:00", "title": "Neutral-mass release audit"}
-->
**Release-conformance finding.** At [reproduction commit 42d6554](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/42d655441096f845c42676fcdf0887fdc127fed6), an executable AST-and-vector audit shows that the released mixed-synthesis path loads only angry, happy, sad, and surprise vectors and never reads `p_neutral`. A valid fixed probe with 40% neutral mass therefore produces the unrenormalized non-neutral sum: it is exactly identical to an explicit zero-neutral-vector interpretation and has norm ratio 0.600000025 versus renormalizing the four available directions (L2 difference 0.376424 at CosyVoice2 attention-output layer 17). The [hashed JSON evidence](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/42d655441096f845c42676fcdf0887fdc127fed6/docs/evidence/claim4_mixing_audit.json) records the pinned source hash and all four released vector hashes. This falsifies faithful five-way composition in the release when neutral probability is positive; it does not by itself measure generated-speech quality.
