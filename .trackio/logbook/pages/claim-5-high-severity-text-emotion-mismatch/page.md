# Claim 5: High-severity text-emotion mismatch


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_fb71ebb89468", "created_at": "2026-07-27T19:35:43+00:00", "title": "Claim and preregistered test"}
-->
**Claim.** Section 4.3, Table 3, and Appendices D.2/E.4 report that CosyVoice2 at alpha 6 achieves E-SIM 0.862, TEP 0.504, S-SIM 0.892, WER 8.74, and N-MOS 4.40 on high text-emotion mismatch; the instruction baseline reports 0.843, 0.436, 0.887, 4.01, and 4.23.

**Preregistered test.** Reconstruct deterministic valence-arousal severity bins and compare alpha 0, 3, and 6 with instruction, wrong-vector, negative-vector, shuffled-vector, and flow-only controls. Report paired emotion, identity, content, and prosody metrics with complete sample accounting and clustered bootstrap intervals. Perceived target emotion and naturalness require blinded human ratings.

**Access boundary.** Exact Table 3 verification requires licensed [IEMOCAP](https://sail.usc.edu/iemocap/) and an unreleased mismatch manifest. A public CREMA-D or RAVDESS mismatch experiment is explicitly a mechanism proxy, never an exact reproduction.
