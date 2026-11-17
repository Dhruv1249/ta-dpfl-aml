#!/usr/bin/env python3
"""Local diagnostic for TA-DPFL per-example DP gradient/update direction.

This script DOES NOT train a model, draw private Poisson samples for training,
write per-record gradients, or issue a DP release. It loads the existing run's
checkpoint and computes aggregate diagnostics on a random sample of TRAIN rows.
The output contains counts and class-averaged statistics only; no row IDs or
feature values are written.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

# Executed as a script from the project's tools directory: explicitly place
# the project root on sys.path so imports such as data.* resolve on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import load_and_label_dataset, build_feature_matrix
from models.aml_classifier import AMLClassifier
from privacy.dp_sgd import per_sample_clipped_gradient
from ta_dpfl_constants import POSITIVE_GROUPS


def load_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        return torch.load(path, map_location=device)


def aggregate_gradient(model, x: np.ndarray, y: np.ndarray, indices: np.ndarray,
                       clip_norm: float, device: torch.device):
    params = tuple(model.parameters())
    sums = [torch.zeros_like(p) for p in params]
    norms = []
    with torch.enable_grad():
        for idx in indices:
            x_one = torch.as_tensor(x[int(idx)], dtype=torch.float32, device=device)
            y_one = torch.as_tensor(float(y[int(idx)]), dtype=torch.float32, device=device)
            clipped, raw_norm = per_sample_clipped_gradient(
                model, x_one, y_one, clip_norm
            )
            norms.append(float(raw_norm))
            for acc, grad in zip(sums, clipped):
                acc.add_(grad)
    return sums, np.asarray(norms, dtype=np.float64)


def evaluate_class_loss(model, x: np.ndarray, y: np.ndarray, indices: np.ndarray,
                        device: torch.device):
    model.eval()
    logits_parts = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            ids = indices[start:start + 512]
            xb = torch.as_tensor(x[ids], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[ids], dtype=torch.float32, device=device)
            logits = model(xb)
            losses.append(F.binary_cross_entropy_with_logits(logits, yb, reduction="none").cpu().numpy())
            logits_parts.append(logits.cpu().numpy())
    return float(np.concatenate(losses).mean()), float(np.concatenate(logits_parts).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Exact YAML config used for the FedAvg DP run")
    parser.add_argument("--method", choices=["uniform", "individualized_dp", "ta_policy_u", "ta_policy_d"], default="uniform")
    parser.add_argument("--samples-per-class", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", default="outputs/diagnostics/dp_gradient_direction_audit.json")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    experiments = config["experiments"]
    seeds = [int(v) for v in experiments.get("random_seeds", [config.get("seed", 2026)])]
    run_seed = seeds[0]
    label = {
        "uniform": "uniform_dp_fl",
        "individualized_dp": "individualized_dp_sgd_style",
        "ta_policy_u": "ta_dpfl_policy_u",
        "ta_policy_d": "ta_dpfl_policy_d",
    }[args.method]
    checkpoint = output_dir / "checkpoints" / f"{label}_seed{run_seed}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    frame = load_and_label_dataset(config, derive_split_cutoffs=False)
    x_all, feature_names = build_feature_matrix(
        frame,
        int(config["data"].get("feature_hash_dim", 16)),
        int(config.get("seed", 2026)),
    )
    y_all = frame["label"].to_numpy(dtype=np.float32)
    groups = frame["group"].astype(str).to_numpy()
    train_mask = frame["split"].astype(str).eq("train").to_numpy()

    label_positive = (y_all == 1.0)
    positive_group = np.isin(groups, list(POSITIVE_GROUPS))
    mismatch_count = int(np.count_nonzero(label_positive != positive_group))

    positive_indices = np.flatnonzero(train_mask & label_positive)
    negative_indices = np.flatnonzero(train_mask & (y_all == 0.0))
    n = int(args.samples_per_class)
    if n < 8:
        raise ValueError("Use at least 8 samples per class.")
    if len(positive_indices) < n or len(negative_indices) < n:
        raise RuntimeError(
            f"Not enough training records for diagnostic sample: positives={len(positive_indices)}, "
            f"negatives={len(negative_indices)}, requested={n} each."
        )

    rng = np.random.default_rng(args.seed)
    positive_sample = rng.choice(positive_indices, size=n, replace=False)
    negative_sample = rng.choice(negative_indices, size=n, replace=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = AMLClassifier(int(x_all.shape[1]), tuple(model_cfg.get("hidden_dims", [128, 64]))).to(device)
    model.load_state_dict(load_state(checkpoint, device), strict=True)
    model.eval()

    privacy_cfg = config["privacy"]
    federated_cfg = config["federated"]
    clip_norm = float(privacy_cfg["clip_norm"])
    normalization = float(privacy_cfg["normalization"])
    learning_rate = float(federated_cfg["learning_rate"])

    # Verify the label-to-group mapping on the full loaded frame.
    # Expected: label 1 corresponds to one of the declared positive typologies.
    train_mapping_mismatches = int(np.count_nonzero(
        train_mask & (label_positive != positive_group)
    ))

    summaries = {}
    original_state = copy.deepcopy(model.state_dict())
    for class_name, indices in [("positive", positive_sample), ("background", negative_sample)]:
        model.load_state_dict(original_state, strict=True)
        model.train()
        before_loss, before_logit = evaluate_class_loss(model, x_all, y_all, indices, device)
        grad_sums, raw_norms = aggregate_gradient(
            model, x_all, y_all, indices, clip_norm, device
        )
        clipped_mean_norm = float(sum(
            torch.sum(g.detach().float() ** 2) for g in grad_sums
        ).sqrt().cpu().item()) / float(n)

        # One deterministic, noise-free update using the same sign, clipping,
        # fixed normalization and local learning rate as the DP-SGD mechanism.
        with torch.no_grad():
            for parameter, grad_sum in zip(model.parameters(), grad_sums):
                parameter.add_(grad_sum / normalization, alpha=-learning_rate)
        after_loss, after_logit = evaluate_class_loss(model, x_all, y_all, indices, device)

        summaries[class_name] = {
            "sample_count": int(n),
            "mean_loss_before": before_loss,
            "mean_loss_after_noise_free_step": after_loss,
            "loss_reduction": before_loss - after_loss,
            "mean_logit_before": before_logit,
            "mean_logit_after_noise_free_step": after_logit,
            "mean_logit_change": after_logit - before_logit,
            "mean_raw_gradient_norm": float(raw_norms.mean()),
            "fraction_examples_clipped": float(np.mean(raw_norms > clip_norm)),
            "mean_clipped_gradient_norm": clipped_mean_norm,
            "update_was_loss_improving": bool(after_loss < before_loss),
        }

    result = {
        "diagnostic": "noise_free_per_example_gradient_direction_audit",
        "warning": (
            "Local mechanism diagnostic only. It is not a DP run, not a privacy proof, "
            "and not an evaluation result. It uses sampled TRAIN rows and writes only aggregate statistics."
        ),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "method": args.method,
        "seed": run_seed,
        "device": str(device),
        "training_rows": int(train_mask.sum()),
        "training_positive_rows": int(np.count_nonzero(train_mask & label_positive)),
        "training_background_rows": int(np.count_nonzero(train_mask & (y_all == 0.0))),
        "feature_count": int(x_all.shape[1]),
        "label_group_mismatches_full_frame": mismatch_count,
        "label_group_mismatches_training": train_mapping_mismatches,
        "declared_positive_groups": sorted(POSITIVE_GROUPS),
        "clip_norm": clip_norm,
        "normalization": normalization,
        "learning_rate": learning_rate,
        "sampled_diagnostic_rows_per_class": int(n),
        "class_diagnostics": summaries,
        "interpretation_checks": {
            "label_group_mapping_consistent": bool(mismatch_count == 0),
            "positive_step_reduces_positive_loss": summaries["positive"]["update_was_loss_improving"],
            "background_step_reduces_background_loss": summaries["background"]["update_was_loss_improving"],
            "positive_step_raises_positive_logit": summaries["positive"]["mean_logit_change"] > 0,
            "background_step_lowers_background_logit": summaries["background"]["mean_logit_change"] < 0,
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved aggregate diagnostic: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
