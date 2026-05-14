"""
Optional checkpoint diagnostics for autoresearch experiments.

This script does not replace the primary train.py val_bpb result. It loads the
current code's model plus a saved checkpoint and measures secondary properties
such as robustness and calibration.
"""

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare import EVAL_TOKENS, MAX_SEQ_LEN, Tokenizer, get_token_bytes, make_dataloader
from train import DEPTH, GPT, build_model_config, detect_runtime


def _load_model(checkpoint_path, runtime, dataset=None):
    tokenizer = Tokenizer.from_directory(dataset=dataset)
    config = build_model_config(
        DEPTH,
        tokenizer.get_vocab_size(),
        runtime,
        use_activation_checkpointing=False,
    )
    model = GPT(config).to(runtime.device)
    state_dict = torch.load(checkpoint_path, map_location=runtime.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def _corrupt_inputs(x, vocab_size, probability):
    if probability <= 0:
        return x
    mask = torch.rand(x.shape, device=x.device) < probability
    random_tokens = torch.randint(0, vocab_size, x.shape, device=x.device, dtype=x.dtype)
    return torch.where(mask, random_tokens, x)


@torch.no_grad()
def evaluate_diagnostics(
    model,
    tokenizer,
    batch_size,
    split,
    device,
    amp_dtype,
    eval_tokens,
    corrupt_prob=0.0,
    include_calibration=True,
):
    token_bytes = get_token_bytes(device=device, dataset=tokenizer.dataset)
    loader = make_dataloader(
        tokenizer,
        batch_size,
        MAX_SEQ_LEN,
        split,
        device=device,
        dataset=tokenizer.dataset,
    )
    steps = max(1, eval_tokens // (batch_size * MAX_SEQ_LEN))
    vocab_size = tokenizer.get_vocab_size()

    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    correct_tokens = 0
    confidence_sum = 0.0
    entropy_sum = 0.0
    ece_bins = 10
    bin_correct = torch.zeros(ece_bins, dtype=torch.float64)
    bin_confidence = torch.zeros(ece_bins, dtype=torch.float64)
    bin_count = torch.zeros(ece_bins, dtype=torch.float64)

    for _ in range(steps):
        x, y, _ = next(loader)
        x_eval = _corrupt_inputs(x, vocab_size, corrupt_prob)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(x_eval)

        logits_f = logits.float()
        y_flat = y.view(-1)
        loss_flat = F.cross_entropy(
            logits_f.view(-1, logits_f.size(-1)),
            y_flat,
            ignore_index=-1,
            reduction="none",
        )
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()

        if include_calibration:
            probs = F.softmax(logits_f, dim=-1)
            confidence, pred = probs.max(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-30).log()).sum(dim=-1)

            pred_flat = pred.view(-1)[mask]
            confidence_flat = confidence.view(-1)[mask]
            entropy_flat = entropy.view(-1)[mask]
            target_flat = y_flat[mask]
            correct_flat = pred_flat.eq(target_flat)

            total_tokens += int(mask.sum().item())
            correct_tokens += int(correct_flat.sum().item())
            confidence_sum += confidence_flat.sum().item()
            entropy_sum += entropy_flat.sum().item()

            bins = torch.clamp((confidence_flat * ece_bins).long(), max=ece_bins - 1).cpu()
            correct_cpu = correct_flat.double().cpu()
            confidence_cpu = confidence_flat.double().cpu()
            for bin_idx in range(ece_bins):
                bin_mask = bins == bin_idx
                if bin_mask.any():
                    bin_count[bin_idx] += bin_mask.sum().item()
                    bin_correct[bin_idx] += correct_cpu[bin_mask].sum().item()
                    bin_confidence[bin_idx] += confidence_cpu[bin_mask].sum().item()

    if total_bytes == 0:
        raise RuntimeError("Evaluation produced zero target bytes; cannot compute BPB.")

    metrics = {
        "split": split,
        "eval_tokens": steps * batch_size * MAX_SEQ_LEN,
        "corrupt_prob": corrupt_prob,
        "bpb": total_nats / (math.log(2) * total_bytes),
    }

    if include_calibration and total_tokens > 0:
        ece = 0.0
        for bin_idx in range(ece_bins):
            count = bin_count[bin_idx].item()
            if count == 0:
                continue
            acc = bin_correct[bin_idx].item() / count
            conf = bin_confidence[bin_idx].item() / count
            ece += (count / total_tokens) * abs(acc - conf)
        metrics.update(
            {
                "token_accuracy": correct_tokens / total_tokens,
                "mean_confidence": confidence_sum / total_tokens,
                "mean_entropy": entropy_sum / total_tokens,
                "ece_10": ece,
            }
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run optional diagnostics on checkpoint_pre_eval.pt")
    parser.add_argument("--checkpoint", default="checkpoint_pre_eval.pt")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-tokens", type=int, default=min(EVAL_TOKENS, MAX_SEQ_LEN * 2 * 8))
    parser.add_argument("--corrupt-prob", type=float, default=0.0)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print only machine-readable JSON.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    if args.json:
        with contextlib.redirect_stdout(io.StringIO()):
            runtime = detect_runtime()
    else:
        runtime = detect_runtime()
    model, tokenizer = _load_model(checkpoint_path, runtime, dataset=args.dataset)
    metrics = evaluate_diagnostics(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        split=args.split,
        device=runtime.device,
        amp_dtype=runtime.amp_dtype,
        eval_tokens=args.eval_tokens,
        corrupt_prob=args.corrupt_prob,
        include_calibration=not args.no_calibration,
    )

    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print("---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
