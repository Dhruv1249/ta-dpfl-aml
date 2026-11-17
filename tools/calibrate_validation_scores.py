#!/usr/bin/env python3
"""Validation-based Monotonic Probability Calibration for TA-DPFL Models.

This script demonstrates Option A:
Under extreme class imbalance, unweighted SGD/FedAvg penalizes high-value features
to suppress false positives across massive background data, causing raw model scores
to anti-correlate with the true labels (yielding ROC-AUC ~ 0.275, meaning inverted
ROC-AUC ~ 0.725).

By fitting a standard monotonic calibrator (Platt Scaling / Logistic Regression)
strictly on the HELD-OUT VALIDATION SET:
    P(y = 1 | s) = sigma(w * s + b)
the calibrator learns the true sign of w (w < 0) from validation data alone without
peeking at the test set.

Applying this validation-learned calibration to test scores aligns the ranking,
restoring the underlying feature discrimination (ROC-AUC ~ 0.725) while strictly
preserving the DP guarantee of the underlying model weights.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import load_and_label_dataset, build_feature_matrix
from models.aml_classifier import AMLClassifier
from evaluation.metrics import select_threshold_at_fpr, evaluate_overall


def load_model(checkpoint_path: Path, input_dim: int, hidden_dims=(128, 64), device=torch.device("cpu")):
    model = AMLClassifier(input_dim, hidden_dims).to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def compute_scores(model, x, device, batch_size=32768):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            s = torch.sigmoid(model(xb)).cpu().numpy()
            scores.append(s)
    return np.concatenate(scores) if scores else np.empty(0, dtype=float)


def run_calibration(config_path: str, checkpoint_dir: str | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading dataset with config: {config_path} on {device}...")
    frame = load_and_label_dataset(cfg)
    x, feature_names = build_feature_matrix(
        frame,
        int(cfg["data"].get("feature_hash_dim", 16)),
        int(cfg.get("seed", 2026)),
    )
    y = frame["label"].to_numpy(dtype=np.int8)
    splits = frame["split"].to_numpy()

    val_idx = np.flatnonzero(splits == "validation")
    test_idx = np.flatnonzero(splits == "test")

    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    input_dim = x.shape[1]
    hidden_dims = tuple(cfg["model"].get("hidden_dims", [128, 64]))

    if checkpoint_dir:
        chk_dir = Path(checkpoint_dir)
    else:
        chk_dir = Path(cfg["output_dir"]) / "checkpoints"

    seed = int(cfg["experiments"].get("random_seeds", [cfg.get("seed", 2026)])[0])

    # Checkpoints to test
    target_checkpoints = [
        ("Uniform-DP-FL", chk_dir / f"uniform_dp_fl_seed{seed}.pt"),
        ("TA-DPFL-Policy-U", chk_dir / f"ta_dpfl_policy_u_seed{seed}.pt"),
        ("FedAvg (Non-private)", chk_dir / f"fedavg_seed{seed}.pt"),
        ("FedAdam (Non-private)", chk_dir / f"fedadam_seed{seed}.pt"),
    ]

    results = []

    for name, path in target_checkpoints:
        if not path.is_file():
            print(f"Skipping {name}: Checkpoint not found at {path}")
            continue

        print(f"\n{'='*70}\nEvaluating: {name} ({path.name})\n{'='*70}")
        model = load_model(path, input_dim, hidden_dims, device)

        # 1. Raw scores
        raw_val_scores = compute_scores(model, x_val, device)
        raw_test_scores = compute_scores(model, x_test, device)

        raw_val_auc = roc_auc_score(y_val, raw_val_scores)
        raw_val_ap = average_precision_score(y_val, raw_val_scores)
        raw_test_auc = roc_auc_score(y_test, raw_test_scores)
        raw_test_ap = average_precision_score(y_test, raw_test_scores)

        # Raw threshold selection at FPR <= 0.01
        raw_thresh_info = select_threshold_at_fpr(y_val, raw_val_scores, max_fpr=0.01)
        raw_test_metrics = evaluate_overall(y_test, raw_test_scores, raw_thresh_info["threshold"])

        print(f"--- RAW METRICS ---")
        print(f"Validation: ROC-AUC = {raw_val_auc:.6f}, AP = {raw_val_ap:.6f}")
        print(f"Test:       ROC-AUC = {raw_test_auc:.6f}, AP = {raw_test_ap:.6f}")
        print(f"Test Precision = {raw_test_metrics['precision']:.6f}, Recall = {raw_test_metrics['recall']:.6f}, F1 = {raw_test_metrics['F1']:.6f}")

        # 2. Validation-based Monotonic Calibration (Platt Scaling)
        # Fit logistic regression exclusively on validation split:
        # P(y=1 | raw_score)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        calibrator.fit(raw_val_scores.reshape(-1, 1), y_val)
        cal_slope = float(calibrator.coef_[0][0])
        cal_intercept = float(calibrator.intercept_[0])
        print(f"\n--- VALIDATION-FITTED CALIBRATION ---")
        print(f"Learned Platt Scaling parameters: w = {cal_slope:.6f}, b = {cal_intercept:.6f}")
        if cal_slope < 0:
            print(f"--> Confirmed: Validation data discovered negative slope (w < 0), naturally fixing the anti-correlation!")

        # Apply the validation-learned mapping to test scores
        cal_val_scores = calibrator.predict_proba(raw_val_scores.reshape(-1, 1))[:, 1]
        cal_test_scores = calibrator.predict_proba(raw_test_scores.reshape(-1, 1))[:, 1]

        cal_val_auc = roc_auc_score(y_val, cal_val_scores)
        cal_val_ap = average_precision_score(y_val, cal_val_scores)
        cal_test_auc = roc_auc_score(y_test, cal_test_scores)
        cal_test_ap = average_precision_score(y_test, cal_test_scores)

        # Calibrated threshold selection at FPR <= 0.01 on validation set
        cal_thresh_info = select_threshold_at_fpr(y_val, cal_val_scores, max_fpr=0.01)
        cal_test_metrics = evaluate_overall(y_test, cal_test_scores, cal_thresh_info["threshold"])

        print(f"--- CALIBRATED METRICS ---")
        print(f"Validation: ROC-AUC = {cal_val_auc:.6f}, AP = {cal_val_ap:.6f}")
        print(f"Test:       ROC-AUC = {cal_test_auc:.6f}, AP = {cal_test_ap:.6f}")
        print(f"Test Precision = {cal_test_metrics['precision']:.6f}, Recall = {cal_test_metrics['recall']:.6f}, F1 = {cal_test_metrics['F1']:.6f}")

        results.append({
            "method": name,
            "raw_test_auc": raw_test_auc,
            "calibrated_test_auc": cal_test_auc,
            "raw_test_ap": raw_test_ap,
            "calibrated_test_ap": cal_test_ap,
            "raw_test_recall": raw_test_metrics["recall"],
            "calibrated_test_recall": cal_test_metrics["recall"],
            "raw_test_precision": raw_test_metrics["precision"],
            "calibrated_test_precision": cal_test_metrics["precision"],
            "raw_test_f1": raw_test_metrics["F1"],
            "calibrated_test_f1": cal_test_metrics["F1"],
            "learned_slope_w": cal_slope,
            "learned_intercept_b": cal_intercept,
        })

    summary_df = pd.DataFrame(results)
    out_csv = Path(cfg["output_dir"]) / "calibrated_results_comparison.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nCalibration comparison saved to: {out_csv}")
    print(summary_df[["method", "raw_test_auc", "calibrated_test_auc", "raw_test_ap", "calibrated_test_ap"]])
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/partial_dp_server_fedavg_20261103_225611_73aa74.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    run_calibration(args.config, args.checkpoint_dir)
