# Experiment Backlog

This backlog turns the research buckets into concrete, on-demand experiments.
Each item should be run alone before being combined with other ideas.

## Current Baseline

- Branch: `autoresearch/may14-neuro`
- Baseline commit: `f9516d5`
- Baseline `val_bpb`: `1.067957`
- Fresh baseline-control commit: `7b24894`
- Fresh baseline-control `val_bpb`: `1.002585`
- First failed idea: hard competitive sparse MLP, `1.110201`

## Neuroscience / Biology Priority Queue

These are ordered by expected signal-to-complexity on this repo. Each item should
translate a biological idea into the smallest ML mechanism that can be tested.

**NB-1: Neuromodulated residual gate**
- Biology inspiration: neuromodulators adjust gain, attention, uncertainty, and
  learning state across broad circuits.
- Hypothesis: a token-conditioned gate initialized to the baseline can learn when
  to amplify or suppress the MLP residual, improving sample efficiency or
  robustness without blunt sparsity.
- Bucket: sample efficiency + calibration/metacognition + sparse computation.
- Harness: standard `val_bpb`; optional calibration/corruption diagnostics.
- Status: discarded after fresh baseline-control. Initial run `f6be652`
  (`1.046428`) and repeat `5c96b86` (`1.033295`) both lost to the fresh control
  `7b24894` (`1.002585`).

**NB-1A: Repeat neuromodulated residual gate**
- Hypothesis: the NB-1 improvement is a real signal, not a one-run fluctuation.
- Change: rerun the same patch, ideally with another seed if seed support is
  added, or with the same seed as a determinism check.
- Metric: `val_bpb`, steps, throughput.
- Status: done once; repeat was worse than fresh baseline.

**NB-1B: Gate ablation and mechanism check**
- Hypothesis: the learned input-conditioned gate matters, not just the added
  parameter path or small optimizer perturbation.
- Change: compare learned gate, frozen gate at 1, layer-only scalar gate, and
  attention-residual gate.
- Metric: `val_bpb`; optionally log gate mean/std by layer.
- Status: frozen identity gate scored `1.015049`, which triggered a fresh
  baseline-control rerun. Do not continue this line until same-session controls
  are in place.

**NB-2: Dendritic MLP branches**
- Biology inspiration: dendrites perform local nonlinear computation before a
  neuron emits its output.
- Hypothesis: several small MLP branches plus a learned combiner may outperform a
  single flat MLP at similar parameter count.
- Bucket: modularity + raw modeling quality.
- Status: first branch-gate implementation discarded at `7b79a17` (`1.051728`).
  Future versions need a stronger reason than simple branch gain.

**NB-3: Nested cortical-column MLP**
- Biology inspiration: cortex has nested circuits such as minicolumns, columns,
  areas, and long-range connections.
- Hypothesis: grouped feedforward channels with limited cross-group mixing can
  encourage specialization while preserving enough communication.
- Bucket: modularity + sparse computation.

**NB-4: Laminar / top-down feedback block**
- Biology inspiration: neocortical layers have different feedforward,
  integration, and feedback roles.
- Hypothesis: a small top-down or skip-feedback signal can improve refinement and
  sample efficiency compared with a purely feedforward stack.
- Bucket: memory + planning/test-time thinking.

**NB-5: Predictive-coding auxiliary loss**
- Biology inspiration: predictive processing theories emphasize predicting
  hidden causes and propagating error signals.
- Hypothesis: a tiny hidden-state prediction objective can regularize internal
  representations and improve early learning.
- Bucket: sample efficiency + raw modeling quality.

**NB-6: Hippocampal replay for continual learning**
- Biology inspiration: hippocampus supports fast episodic learning and replay for
  cortical consolidation.
- Hypothesis: replaying a small buffer from phase A while adapting to phase B
  reduces forgetting.
- Bucket: continual learning + memory.
- Harness needed: two-phase continual-learning trainer.

**NB-7: Homeostatic excitation/inhibition balance**
- Biology inspiration: cortical circuits regulate excitatory and inhibitory
  activity to avoid runaway dynamics.
- Hypothesis: a light activation-balance penalty or inhibitory channel can
  stabilize training and improve robustness.
- Bucket: robustness + optimization + sparse computation.

**NB-8: Slow/fast pathways and rhythms**
- Biology inspiration: neural systems operate across multiple timescales and
  oscillatory rhythms.
- Hypothesis: layers or channels updated at different frequencies can improve
  efficiency or memory.
- Bucket: memory + sample efficiency.

**NB-9: Neurogenesis / adaptive growth**
- Biology inspiration: biological systems can add or repurpose capacity as they
  encounter novelty.
- Hypothesis: small expandable modules can absorb new tasks while preserving old
  behavior.
- Bucket: continual learning + modularity.
- Harness needed: two-phase continual-learning trainer.

**NB-10: Chemical broadcast tokens**
- Biology inspiration: chemical signals broadcast low-dimensional global state
  that modulates many local computations.
- Hypothesis: a few learned global modulatory vectors can improve coordination
  across layers.
- Bucket: memory + calibration/metacognition + modulation.

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

**SE-3 / NB-1: Neuromodulated residual gate**
- Hypothesis: token-conditioned gain control can improve early learning while
  starting from baseline-equivalent behavior.
- Change: multiply the MLP residual by a learned scalar gate initialized to 1.
- Metric: `val_bpb`; optional calibration and corruption diagnostics.

### Raw Modeling Quality

**MQ-1: Window pattern sweep**
- Hypothesis: the `SSSL` attention pattern may not be optimal on this GPU/data.
- Change: test `LLLL`, `SLLL`, or `SSLL`.
- Metric: `val_bpb`, throughput, memory.
- Status: `LLLL` discarded at `9b53e4a` (`1.030899`), worse than fresh `SSSL`
  baseline. `SLLL` or `SSLL` remain untested.

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

**MO-3 / NB-2: Dendritic MLP branches**
- Hypothesis: local branch computation plus learned combining can provide a
  dendrite-like inductive bias.
- Change: split MLP hidden channels into branches with branch-level gates.
- Metric: `val_bpb`, parameter count, speed.

**MO-4 / NB-3: Nested cortical-column MLP**
- Hypothesis: nested channel groups improve specialization without full MoE
  overhead.
- Change: grouped feedforward channels plus occasional cross-group mixing.
- Metric: `val_bpb`, speed, robustness diagnostics.

### Memory

**ME-1: Recurrent residual state**
- Hypothesis: a tiny recurrent summary can improve sample efficiency on stories.
- Change: add a per-layer or global state path inside a block.
- Metric: `val_bpb`; watch for speed regressions.

**ME-2: Memory tokens**
- Hypothesis: reserved summary tokens can carry useful context through layers.
- Change: prepend a small number of learned memory tokens internally.
- Metric: `val_bpb`, memory use.

**ME-3 / NB-4: Laminar feedback signal**
- Hypothesis: top-down or late-to-early-style modulation improves refinement.
- Change: add a small feedback/modulation path across layer groups.
- Metric: `val_bpb`; later, calibration and synthetic reasoning diagnostics.

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

**CL-2 / NB-6: Hippocampal replay**
- Harness needed: phase A/phase B trainer with a replay buffer.
- Hypothesis: small replay buffers reduce forgetting more cheaply than full
  mixed-data retraining.
- Metric: forgetting delta and phase B learning speed.

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
