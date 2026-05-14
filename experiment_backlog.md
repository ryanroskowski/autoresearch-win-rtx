# Experiment Backlog

This backlog turns the research buckets into concrete, on-demand experiments.
Each item should be run alone before being combined with other ideas.

## Current Baseline

- Branch: `autoresearch/may14-neuro`
- Baseline commit: `f9516d5`
- Baseline `val_bpb`: `1.067957`
- First failed idea: hard competitive sparse MLP, `1.110201`

## Lab-Ready Now

### Sample Efficiency

**SE-1: Faster early learning schedule**
- Hypothesis: the baseline schedule may underuse the first few minutes on a
  consumer GPU.
- Change: adjust warmdown/warmup or matrix LR conservatively.
- Metric: lower `val_bpb` at the same fixed budget; optionally compare loss at
  early steps.

**SE-2: Smaller model, more steps**
- Hypothesis: a slightly smaller model may train farther in five minutes and beat
  the larger baseline on TinyStories.
- Change: reduce `DEPTH` or `ASPECT_RATIO`.
- Metric: `val_bpb`, tokens processed, steps.

### Raw Modeling Quality

**MQ-1: Window pattern sweep**
- Hypothesis: the `SSSL` attention pattern may not be optimal on this GPU/data.
- Change: test `LLLL`, `SLLL`, or `SSLL`.
- Metric: `val_bpb`, throughput, memory.

**MQ-2: Residual initialization sweep**
- Hypothesis: `x0_lambdas` and residual scales influence early training stability.
- Change: test smaller or layer-dependent initial residual mixing.
- Metric: `val_bpb`, loss curve stability.

### Sparse Computation

**SP-1: Late-layer learned MLP gate**
- Hypothesis: learned gates can suppress unhelpful MLP transforms without the
  blunt damage of hard thresholding.
- Change: add a per-channel or per-layer gate initialized near the baseline.
- Metric: `val_bpb`; optional calibration/robustness diagnostics.

**SP-2: Late-layer-only sparsity**
- Hypothesis: sparse competition may be less harmful after early dense features
  have formed.
- Change: apply gentle sparsity only in the final half of layers.
- Metric: `val_bpb`, throughput.

### Modularity

**MO-1: Grouped MLP**
- Hypothesis: splitting the MLP into independent channel groups encourages
  specialization with little overhead.
- Change: grouped feedforward projection or grouped activation mixing.
- Metric: `val_bpb`, parameter count, speed.

**MO-2: Tiny routed MLP branches**
- Hypothesis: two small MLP branches with a learned token gate may outperform one
  dense MLP at similar parameter count.
- Change: add simple two-way routing in MLP.
- Metric: `val_bpb`, routing entropy, speed.

### Memory

**ME-1: Recurrent residual state**
- Hypothesis: a tiny recurrent summary can improve sample efficiency on stories.
- Change: add a per-layer or global state path inside a block.
- Metric: `val_bpb`; watch for speed regressions.

**ME-2: Memory tokens**
- Hypothesis: reserved summary tokens can carry useful context through layers.
- Change: prepend a small number of learned memory tokens internally.
- Metric: `val_bpb`, memory use.

### Optimization

**OP-1: AdamW/Muon LR sweep**
- Hypothesis: consumer-GPU TinyStories defaults may not be tuned for this setup.
- Change: small changes to `MATRIX_LR`, `EMBEDDING_LR`, or warmdown.
- Metric: `val_bpb`, loss curve.

**OP-2: Weight decay sweep**
- Hypothesis: current weight decay may be too high or low for the five-minute
  regime.
- Change: test nearby `WEIGHT_DECAY` values.
- Metric: `val_bpb`.

## Needs Small Harness Additions

### Continual Learning

**CL-1: Two-phase TinyStories split**
- Harness needed: phase A/phase B train script or train options.
- Hypothesis: modular/gated updates forget less after switching data slices.
- Metric: phase A BPB after phase B, phase B BPB, forgetting delta.

### Robustness

**RB-1: Token corruption diagnostic**
- Harness status: available in `evals/diagnostic_eval.py`.
- Hypothesis: learned gates or memory paths reduce degradation under random token
  corruption.
- Metric: clean BPB, corrupted BPB, degradation delta.

### Calibration / Metacognition

**CA-1: Confidence calibration diagnostic**
- Harness status: available in `evals/diagnostic_eval.py`.
- Hypothesis: some architectures are less overconfident for the same BPB.
- Metric: token accuracy, mean confidence, entropy, ECE.

### Synthetic Reasoning

**SR-1: Copy/reverse/bracket microtasks**
- Harness needed: synthetic dataset generator plus tiny train/eval protocol.
- Hypothesis: recurrence or memory should help on algorithmic sequence tasks.
- Metric: task accuracy and generalization to longer lengths.

### Out-of-Distribution Transfer

**OD-1: TinyStories test split**
- Harness status: mostly available through the existing dataloader and
  diagnostic eval split support.
- Hypothesis: simpler/modular models overfit less to validation distribution.
- Metric: val BPB vs test BPB gap.

## Cross-Cutting Combinations

Only run these after components have been tested alone.

- **SE + OP:** smaller model plus tuned schedule.
- **SP + RB:** learned gates plus corruption robustness diagnostic.
- **MO + CL:** routed modules plus two-phase forgetting test.
- **ME + SR:** recurrent memory plus synthetic copy/reverse tasks.
- **CA + Planning:** confidence-triggered extra compute, once planning support
  exists.

## Deferred Harnesses

- Two-phase continual-learning trainer.
- Synthetic sequence task generator and evaluator.
- Test-time refinement/planning loop.
- Interactive environment/world-model harness.
