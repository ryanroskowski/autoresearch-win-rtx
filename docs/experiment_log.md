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
- Status: `keep`
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
