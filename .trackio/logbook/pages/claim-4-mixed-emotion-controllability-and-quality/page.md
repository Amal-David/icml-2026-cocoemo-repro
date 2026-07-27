# Claim 4: Mixed-emotion controllability and quality


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2299c5324cb7", "created_at": "2026-07-27T19:35:42+00:00", "title": "Documentary falsification and preregistered test"}
-->
**Claim audit.** The challenge prompt combines numbers that do not occur in one Table 2 row of [arXiv:2602.03420v2](https://arxiv.org/pdf/2602.03420v2). On CREMA-D, TEP 0.335 and Spearman rho 0.319 belong to CoCoEmo with instruction 1 at alpha 5, whose N-MOS is 3.00. N-MOS 3.96 belongs to bare CoCoEmo at alpha 5, while 4.25 belongs to bare CoCoEmo at alpha 3. The composite numerical claim therefore **fails documentary audit** and will be tested row by row.

**Preregistered test.** Reproduce each named CREMA-D condition separately, then compare genuine mixed vectors with instruction-only, dominant-only, label-shuffled, and norm-matched-random controls using paper-faithful and frozen independent evaluators. Report paired bootstrap intervals and full denominators. Naturalness requires a real blinded listening study; no automatic score or agent judgment will substitute for N-MOS.

**Scope.** The public [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) subset is feasible. Exact out-of-distribution confirmation on [IEMOCAP](https://sail.usc.edu/iemocap/) remains blocked without licensed access.
