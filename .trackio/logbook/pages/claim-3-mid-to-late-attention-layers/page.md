# Claim 3: Mid-to-late attention layers


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f3d17a9feb36", "created_at": "2026-07-27T19:35:42+00:00", "title": "Claim and preregistered test"}
-->
**Claim.** Section 2.2/Figure 3 identifies CosyVoice2 layers 10-17 and IndexTTS2 layers 5-10 as highly emotion-separable; Section 4.1 selects CosyVoice2 layers 17 and 14 and IndexTTS2 layers 6, 8, and 1.

**Preregistered test.** Fit speaker-disjoint multiclass probes across every layer and hook site, select only on validation data, compare selected and matched non-selected sites, and enforce the paper WER quality bound. Include a permuted-label null and corrected multiple comparisons. Exact reproduction passes only if the reported bands and final sets survive held-out evaluation; the same band with different top-k is partial; leakage, incompatible hooks, or no control advantage fails.

**Release audit.** The public IndexTTS2 extraction path imports the CosyVoice-shaped helper, and shipped vector dimensions require a GPU canary before any Index result is accepted. The first substantive run therefore uses the pinned [CosyVoice2 model](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/tree/eec1ae6c79877dbd9379285cf8789c9e0879293d).


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c19381a54317", "created_at": "2026-07-27T19:40:30+00:00", "title": "Executable release conformance audit"}
-->
The clean-tree preflight at [commit e9eff32](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/e9eff32ff0ee7bd6b6d8cba0cde37cf5a69e602e) produced hashed provenance and passed 8 contract tests. It found three reproducible release deviations:

1. All four shipped IndexTTS2 checkpoint families contain 24 attention-layer vectors of shape 1 x 1280, while the public IndexTTS2 adapter declares hidden dimension 1024.
2. Both scripts/extract.py and scripts/discriminability.py import extract_with_hooks_audio from the Index core instead of the adapter-specific extract_with_hooks_audio_indextts2 helper.
3. The paper specifies norm-preserving steering, but prepare_steering_injection_config actively selects translation_op_; the norm-preserving operator is commented out.

These findings block accepting IndexTTS2 results before a GPU canary and make raw translation a named implementation-deviation control for CosyVoice2. They are source/artifact conformance evidence, not a substitute for the speaker-disjoint GPU layer scan.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_claim23_site_proxy", "created_at": "2026-07-28T05:35:00+00:00", "title": "Frozen speaker-disjoint site scan"}
-->
At [reproduction commit 34cac6d](https://github.com/Amal-David/icml-2026-cocoemo-repro/tree/34cac6d), the same [public-RAVDESS full configuration](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/34cac6d/configs/claim23-ravdess-paired-sites.yaml) freezes a five-class, speaker-disjoint scan of ten paper-defined CosyVoice2 hook types across all 24 layers. Train, validation, and test contain 12, five, and seven disjoint actors respectively. A nearest-centroid linear rule is fit on train only; the top two sites are selected on validation only; and a 200-permutation maximum-over-240-sites null controls selection multiplicity.

The held-out report must include the validation-selected sites and the paper's attention-output layers 14 and 17 even when they are not selected, with validation ranks, seven-actor bootstrap intervals, full predictions, and five-by-five confusion matrices. Support requires the winning site to be attention output in layers 10-17, a max-T p-value at most 0.05, a held-out lower accuracy bound above 0.20, and a positive lower bound against the same-layer median of the other nine operations. The [12-clip canary](https://github.com/Amal-David/icml-2026-cocoemo-repro/blob/34cac6d/configs/claim23-ravdess-canary.yaml) is structural only. CosyVoice2 GPU evidence remains pending, and IndexTTS2 remains blocked by its released adapter/vector incompatibilities.
