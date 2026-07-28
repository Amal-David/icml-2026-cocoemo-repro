# Claim 2: Mean-difference steering vectors


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ae4293400b4f", "created_at": "2026-07-27T19:35:42+00:00", "title": "Claim and preregistered test"}
-->
**Claim.** Section 3.1, Equation 6, Section 4.1, and Appendix D.1/Table 6 define each steering direction as mean emotional activation minus mean neutral activation using speaker- and transcript-matched pairs. The reported corpus combines [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data), [RAVDESS](https://zenodo.org/records/1188976), and [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) for 20,691 utterances, roughly 4,000 per emotion.

**Preregistered test.** Audit every pair and speaker split, reconstruct vectors from legally available data, and compare genuine directions with shuffled-label, speaker-mismatched, and norm-matched Gaussian controls on held-out speakers. Pass requires the exact pairing/count audit and a held-out advantage over controls; a public subset with the same directional effect is partial; invalid pairing, incompatible vectors, or no advantage fails.

**Access boundary.** Exact reconstruction is blocked until licensed ESD access exists. The released vectors can be validated independently, but they cannot prove the unreleased 20,691-item construction manifest.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_87456eb664f2", "created_at": "2026-07-27T19:40:30+00:00", "title": "Released-vector provenance audit"}
-->
At [reproduction commit e9eff32](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/e9eff32ff0ee7bd6b6d8cba0cde37cf5a69e602e), the executable audit loaded and hashed all eight released checkpoints. All four CosyVoice2 direction sets contain 24 attention-layer tensors of shape 1 x 896 and match the declared backbone dimension. Their embedded metadata names ESD and RAVDESS for surprise, and ESD, RAVDESS, and CREMA-D for angry, sad, and happy; it does not include the original utterance manifest or pairing records, so the 20,691-item construction remains unverified.

This is local artifact-conformance evidence only. It does not replace a substantive Hugging Face GPU extraction run.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_claim23_vector_proxy", "created_at": "2026-07-28T05:35:00+00:00", "title": "Frozen public paired-vector proxy"}
-->
At [reproduction commit 34cac6d](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/34cac6d), Claim 2 has a frozen [full public-RAVDESS configuration](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/34cac6d/configs/claim23-ravdess-paired-sites.yaml) and hash-locked [12-clip structural canary](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/34cac6d/configs/claim23-ravdess-canary.yaml). The full proxy uses 480 unique intensity-01 speech clips: 24 actors x two statements x two repetitions x five emotions. Actors 01-12, 13-17, and 18-24 are fixed as train, validation, and test, giving 48 exact train pairs and 28 untouched test pairs per target emotion.

The corrected extractor uses a deterministic teacher-forced CosyVoice2 unistream sequence and reads each hook at its true final speech-token position. It refuses the released path's silent skips and padded `[:, -1, :]` activations. Equation 6 is reconstructed at all 240 operation-layer sites; the held-out directional test at attention-output layer 17 compares exact matched deltas with deterministic different-speaker neutral mismatches, 100 within-group label permutations, and 100 norm-matched random vectors. Full pair and per-pair control ledgers, vector hashes, actor-cluster intervals, and a fail/inconclusive/support verdict are mandatory artifacts. GPU evidence is still pending; this remains a preregistered public proxy, not proof of the unreleased 20,691-item construction.
