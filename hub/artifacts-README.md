---
pretty_name: CoCoEmo ICML 2026 reproduction artifacts
license: other
task_categories:
  - text-to-speech
  - audio-classification
tags:
  - reproducibility
  - emotional-tts
  - icml-2026-agent-repro
---

# CoCoEmo ICML 2026 reproduction artifacts

This dataset stores immutable outputs for the independent reproduction of
[CoCoEmo](https://arxiv.org/abs/2602.03420) in the
[ICML 2026 Agent Reproducibility Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).

Each completed run is stored under a commit- and configuration-derived path and
contains, where applicable:

- repository, model, data, evaluator, hardware, and Job provenance;
- resolved configuration and SHA-256 manifests;
- requested, generated, and evaluated sample counts;
- per-sample metrics and aggregate bootstrap intervals;
- generated audio whose source licenses permit redistribution;
- a `COMPLETE.json` marker written only after every required artifact uploads.

The reproduction code is available at
[Amal-David/icml-2026-cocoemo-repro](https://github.com/Amal-David/icml-2026-cocoemo-repro).
The original implementation is pinned at
[wsssy/CoCoEmo@dcc3191](https://github.com/wsssy/CoCoEmo/tree/dcc319148dd2e0e0f3039e06e81e0f770d108f08).

## Licensing

The reproduction harness is MIT licensed. Source recordings and generated audio
retain the restrictions of their source datasets and model licenses. Each run
must record those licenses in provenance; files without clear redistribution
permission are represented only by hashes and aggregate results, not uploaded.

## Integrity

Absence of `COMPLETE.json`, a count mismatch, a missing required metric, or a
dirty Git worktree makes a run ineligible as claim evidence.
