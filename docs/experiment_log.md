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
