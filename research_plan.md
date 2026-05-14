# Research Plan

This project is an on-demand architecture lab. The main score remains validation
bits per byte (`val_bpb`): lower is better. Secondary diagnostics can be added
when an experiment targets a weakness that `val_bpb` alone does not fully
measure.

## Buckets

### Lab-Ready Now

These can be tested by editing `train.py` and running the standard fixed-budget
experiment.

- **Sample efficiency:** reach lower `val_bpb` with the same time/token budget.
- **Raw modeling quality:** improve `val_bpb`, loss curve, or throughput without
  adding unjustified complexity.
- **Sparse computation:** use fewer active channels/modules/tokens while matching
  or beating the baseline.
- **Modularity:** add reusable submodules, grouped MLPs, or light routing.
- **Memory:** add recurrence, memory tokens, or compact state paths.
- **Optimization:** improve learning rate schedules, optimizer settings, update
  rules, or initialization.

### Needs Small Harness Additions

These need extra measurement scripts or short alternate protocols, but do not
require a new project.

- **Continual learning:** train on phase A, adapt on phase B, then measure both
  new learning and forgetting.
- **Robustness:** evaluate the checkpoint under controlled corruptions, shifted
  data, or transformed inputs.
- **Calibration/metacognition:** measure confidence, entropy, token accuracy, and
  expected calibration error.
- **Synthetic reasoning:** train/evaluate on tiny algorithmic or pattern tasks
  such as copy, reverse, bracket matching, or simple rule transforms.
- **Out-of-distribution transfer:** evaluate on a held-out domain or style after
  training on the usual data.

### Bigger Research Track

These are worth exploring later, but they need new datasets, environments, or
training loops.

- **Grounding/world models:** action-conditioned prediction, gridworlds, or
  state-transition learning.
- **Planning/test-time thinking:** internal refinement loops, verifier heads, or
  compute allocated based on difficulty.
- **Interactive skill acquisition:** ARC-like tasks where the model must infer
  rules from a few examples.

## Experiment Protocol

Each experiment should name:

- **Primary bucket:** the main weakness it tries to improve.
- **Secondary bucket:** optional, when the idea crosses categories.
- **Hypothesis:** why the change might help.
- **Patch:** the smallest code change that tests the hypothesis.
- **Primary metric:** usually `val_bpb`.
- **Secondary metrics:** optional diagnostics from `evals/`.
- **Decision:** `keep`, `discard`, or `follow-up`.

## Combination Policy

Do not combine unproven ideas. First test each component alone. A cross-cutting
experiment is allowed when at least two changes have individually shown promise
or have complementary failure modes worth testing.

Combination experiments should record:

- The component commits or descriptions.
- The expected interaction.
- Whether the combination is additive, redundant, or harmful.
- Whether the added complexity is still justified.

Examples:

- **Memory + sample efficiency:** add a tiny recurrent state and check whether it
  reaches baseline `val_bpb` in fewer steps.
- **Modularity + continual learning:** route new-domain learning through a small
  added module to reduce forgetting.
- **Sparsity + robustness:** test whether learned gates make the model less
  brittle under token corruption.
- **Planning + calibration:** add internal refinement only when confidence is low.

## Current First-Pass Priority

The first hard-sparsity MLP test was worse than baseline. Future sparsity tests
should be gentler: learned gates, late-layer-only sparsity, or regularization
instead of thresholding every MLP activation.
