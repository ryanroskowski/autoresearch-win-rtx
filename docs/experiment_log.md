# Autoresearch Experiment Log

Run tag: `may14-neuro`
Machine: Windows, NVIDIA GeForce RTX 3080 10 GB
Dataset: TinyStories GPT-4 clean

## Baseline

- Commit: `f9516d5`
- Status: `keep`
- `val_bpb`: `1.067957`
- Peak VRAM: `5.3 GB`
- Steps: `38`
- Total tokens: `19.9M`
- Description: baseline unmodified Windows RTX fork.

## Experiment 1: Competitive Sparse MLP

- Commit: `c4ef467`
- Status: `discard`
- `val_bpb`: `1.110201`
- Baseline delta: `+0.042244` worse
- Peak VRAM: `5.3 GB`
- Steps: `37`
- Total tokens: `19.4M`
- Description: mean-threshold competitive sparse MLP with active-fraction rescale.

Interpretation:
Hard thresholding every MLP layer hurt validation loss and slightly reduced
throughput. Future sparsity tests should use gentler learned gates, late-layer
sparsity, or regularization.

## Experiment 2: Neuromodulated MLP Residual Gate

- Commit: `f6be652`
- Status: `discard`, superseded by fresh baseline-control rerun
- `val_bpb`: `1.046428`
- Baseline delta: `-0.021529` better
- Peak VRAM: `5.3 GB`
- Steps: `40`
- Total tokens: `21.0M`
- Description: token-conditioned scalar gate on the MLP residual, initialized so
  the model starts baseline-equivalent.

Secondary diagnostics over 32,768 eval tokens:
- Clean BPB: `1.151669`
- Clean token accuracy: `0.321113`
- Clean ECE-10: `0.029137`
- 5% token-corruption BPB: `1.251907`
- 5% token-corruption token accuracy: `0.302737`
- 5% token-corruption ECE-10: `0.013953`

Interpretation:
This is the first positive signal. A gentle neuromodulation/gain-control analogue
improved the main fixed-budget score with essentially unchanged memory use. It
needs repeat runs and ablations before making a strong claim.

Later note:
The follow-up batch found that a fresh baseline-control run under current GPU
conditions scored `1.002585`, substantially better than this gate result. Treat
this experiment as a useful lesson in controls, not as a kept architecture win.

## Experiment 3: Neuromodulated MLP Gate Repeat

- Commit: `5c96b86`
- Status: `discard`
- `val_bpb`: `1.033295`
- Peak VRAM: `5.3 GB`
- Steps: `41`
- Total tokens: `21.5M`
- Description: repeat of the token-conditioned MLP residual gate.

Interpretation:
The repeat beat the original stale baseline but lost to the fresh baseline
control. The architecture is not currently supported as an improvement.

## Experiment 4: Frozen Identity MLP Gate

- Commit: `df6786c`
- Status: `discard`
- `val_bpb`: `1.015049`
- Peak VRAM: `5.3 GB`
- Steps: `41`
- Total tokens: `21.5M`
- Description: kept the gate-shaped module present but forced `gate = 1`.

Interpretation:
This ablation was surprisingly strong, which suggested the original baseline was
not a fair enough control. That led to the fresh baseline rerun below.

## Experiment 5: Fresh Baseline Control

- Commit: `7b24894`
- Status: `keep`
- `val_bpb`: `1.002585`
- Peak VRAM: `5.3 GB`
- Steps: `42`
- Total tokens: `22.0M`
- Description: reran the clean baseline architecture under current conditions.

Interpretation:
This is the new score to beat. The earlier apparent gate improvement was
confounded by run conditions and throughput differences. Future claims should be
compared against this control or another fresh same-session baseline.

## Experiment 6: Dendritic Branch-Gated MLP

- Commit: `7b79a17`
- Status: `discard`
- `val_bpb`: `1.051728`
- Peak VRAM: `5.3 GB`
- Steps: `40`
- Total tokens: `21.0M`
- Description: split the MLP hidden state into four branch groups with
  token-conditioned gain controls initialized to 1.

Interpretation:
The branch-gated MLP was slower and worse than the fresh baseline. Dendritic
branching remains interesting, but this implementation is not the right form.

## Experiment 7: All-Full Attention Window Pattern

- Commit: `9b53e4a`
- Status: `discard`
- `val_bpb`: `1.030899`
- Peak VRAM: `5.3 GB`
- Steps: `42`
- Total tokens: `22.0M`
- Description: changed the attention window pattern from `SSSL` to `LLLL`.

Interpretation:
Full attention in every layer lost to the default window pattern on this setup.
The default `SSSL` remains the better control.
