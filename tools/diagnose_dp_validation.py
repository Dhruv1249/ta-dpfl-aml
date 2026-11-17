from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML config must contain a mapping.")
    return config


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size != 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose DP checkpoint score direction and discrimination using the "
            "validation split only. Does not train models or use the test split."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Exact YAML config used for the DP run")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run output directory; defaults to config output_dir")
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(config_path)
    run_dir = args.run_dir.resolve() if args.run_dir else (PROJECT_ROOT / config.get("output_dir", "outputs")).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Match the exact preprocessing and split path used by run_experiment.py.
    from data.preprocessing import load_and_label_dataset, build_feature_matrix
    from models.aml_classifier import AMLClassifier

    hardware_cfg = config.get("hardware", {})
    cpu_threads = int(hardware_cfg.get("cpu_threads", 4))
    torch.set_num_threads(max(1, cpu_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frame = load_and_label_dataset(config, derive_split_cutoffs=False)
    x, _feature_names = build_feature_matrix(
        frame,
        int(config["data"].get("feature_hash_dim", 16)),
        int(config.get("seed", 2026)),
    )
    val_mask = frame["split"].eq("validation").to_numpy()
    y_val = frame.loc[val_mask, "label"].to_numpy(dtype=np.int64)
    x_val = np.asarray(x[val_mask], dtype=np.float32)
    if len(y_val) == 0 or np.unique(y_val).size != 2:
        raise RuntimeError("Validation split is empty or does not contain both classes.")

    prevalence = float(y_val.mean())
    model_cfg = config.get("model", {})
    hidden_dims = tuple(int(v) for v in model_cfg.get("hidden_dims", [128, 64]))
    input_dim = int(x_val.shape[1])
    batch_size = int(hardware_cfg.get("evaluation_batch_size", 8192))
    if batch_size < 1:
        raise ValueError("evaluation_batch_size must be positive")

    method_checkpoints = {
        "Uniform-DP-FL": "uniform_dp_fl_seed2026.pt",
        "Individualized-DP-SGD-style": "individualized_dp_sgd_style_seed2026.pt",
        "TA-DPFL-Policy-U": "ta_dpfl_policy_u_seed2026.pt",
        "TA-DPFL-Policy-D": "ta_dpfl_policy_d_seed2026.pt",
    }

    rows: list[dict[str, Any]] = []
    print(f"Validation rows: {len(y_val):,}; positives: {int(y_val.sum()):,}; prevalence: {prevalence:.8f}")
    print("\nValidation-only checkpoint diagnostic (do not use inverted scores as a corrected model):")

    for method, filename in method_checkpoints.items():
        checkpoint = checkpoint_dir / filename
        if not checkpoint.is_file():
            print(f"MISSING: {checkpoint}")
            continue

        model = AMLClassifier(input_dim, hidden_dims).to(device)
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            # Compatibility fallback for older PyTorch versions; only load your own checkpoint.
            state = torch.load(checkpoint, map_location="cpu")
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint format in {checkpoint}")
        model.load_state_dict(state)
        model.eval()

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(x_val), batch_size):
                xb = torch.as_tensor(x_val[start:start + batch_size], dtype=torch.float32, device=device)
                logits = model(xb).reshape(-1)
                chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
        scores = np.concatenate(chunks).astype(np.float64, copy=False)
        positive_scores = scores[y_val == 1]
        negative_scores = scores[y_val == 0]
        auc = safe_auc(y_val, scores)
        inv_auc = safe_auc(y_val, 1.0 - scores)
        ap = float(average_precision_score(y_val, scores))
        inv_ap = float(average_precision_score(y_val, 1.0 - scores))

        row = {
            "method": method,
            "validation_n": int(len(y_val)),
            "validation_positives": int(y_val.sum()),
            "validation_prevalence": prevalence,
            "AP_raw_score": ap,
            "AP_inverted_score_diagnostic_only": inv_ap,
            "ROC_AUC_raw_score": auc,
            "ROC_AUC_inverted_score_diagnostic_only": inv_auc,
            "mean_score_positive": float(positive_scores.mean()),
            "mean_score_negative": float(negative_scores.mean()),
            "median_score_positive": float(np.median(positive_scores)),
            "median_score_negative": float(np.median(negative_scores)),
            "score_std_all": float(scores.std()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
        }
        rows.append(row)
        print(
            f"{method}: AP={ap:.8f} (prevalence={prevalence:.8f}), "
            f"AUC={auc:.6f}, inverted-AUC diagnostic={inv_auc:.6f}, "
            f"mean score positive={positive_scores.mean():.6g}, "
            f"negative={negative_scores.mean():.6g}, std={scores.std():.6g}"
        )
        del model, scores, chunks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise FileNotFoundError(
            f"No expected DP checkpoints found in {checkpoint_dir}. "
            "Check checkpoint names and the selected run directory."
        )

    out_path = run_dir / "dp_validation_score_diagnostic.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("This diagnostic used validation data only; it did not load or evaluate test data.")
    print("The inverted-score metrics are diagnostic indicators only, not a recommendation to reverse predictions.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
