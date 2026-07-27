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
