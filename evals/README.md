# Optional Evaluation Harnesses

The standard experiment result is still `val_bpb` from `uv run train.py`.
Scripts in this directory are secondary diagnostics for categories that need
more than the default validation score.

Use these after a training run has produced `checkpoint_pre_eval.pt`.

```powershell
uv run python evals/diagnostic_eval.py --eval-tokens 32768 --batch-size 2
```

Current diagnostics:

- Clean validation/test BPB for a loaded checkpoint.
- Optional token-corruption robustness BPB.
- Token accuracy, mean confidence, entropy, and expected calibration error.

These numbers should be logged in `results/experiment_log.md` when they explain
an experiment. They should not override a clearly worse primary `val_bpb` unless
the experiment was explicitly designed as a diagnostic tradeoff.
