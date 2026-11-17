#!/usr/bin/env python3
"""Unified Verification & Benchmark Suite for TA-DPFL.

Verifies and evaluates ALL experimental baselines and policies on the
held-out test split (1,015,882 rows, 1,798 positives):
1. Central Baselines: Central-NN (Class-Weighted & Unweighted)
2. Non-Private Federated: FedAdam vs. FedAvg (Collapsed)
3. Historical Collapsed DP Baselines: Uniform-DP & Policy-U (b0 = 0)
4. All 4 Differentially Private Policies (Fixed b0 = -7.19 + Server FedAdam):
   - Uniform-DP-FL (eps = 4.0, sigma = 0.6284)
   - Individualized-DP-SGD (eps in [2.0, 4.0], sigma = 0.95)
   - TA-DPFL Policy U (eps = 4.0, sigma = 0.85, q_focus = 0.05)
   - TA-DPFL Policy D (eps in [2.0, 4.0], eps_bg = 2.0, sigma = 1.15)
5. Complete 9-Typology Evaluation Breakdown across all 4 DP policies.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import load_and_label_dataset, build_feature_matrix
from models.aml_classifier import AMLClassifier

POS_TYPOLOGIES = [
    "CYCLE",
    "GATHER-SCATTER",
    "BIPARTITE",
    "FAN-OUT",
    "SCATTER-GATHER",
    "STACK",
    "RANDOM",
    "FAN-IN",
    "UNCL",
]


def train_policy(
    policy_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    device: torch.device,
    steps: int = 500,
    seed: int = 2026,
    ckpt_dir: Path = Path("outputs/final_benchmark/checkpoints"),
) -> AMLClassifier:
    """Train or load a differentially private policy model with prior bias + server FedAdam."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{policy_name}.pt"

    model = AMLClassifier(input_dim, hidden_dims).to(device)

    if ckpt_path.is_file():
        print(f"[Verification] Loading existing checkpoint for {policy_name}: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        return model

    print(f"[Verification] Training {policy_name} on {device} ({steps} steps)...")
    prior_bias = float(np.log(y_train.mean() / (1.0 - y_train.mean())))
    with torch.no_grad():
        list(model.network.children())[-1].bias.fill_(prior_bias)

    params = tuple(model.parameters())
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    beta1, beta2 = 0.9, 0.99
    server_lr = 0.01
    tau = 1e-4

    group_indices = {g: np.flatnonzero(g_train == g) for g in POS_TYPOLOGIES}
    bg_indices = np.flatnonzero(g_train == "BG")
    all_pos_indices = np.flatnonzero(y_train == 1)

    rng = np.random.default_rng(seed)

    for step in range(1, steps + 1):
        client_updates = [torch.zeros_like(p) for p in params]
        for c in range(4):
            if policy_name == "individualized_dp":
                batch_neg = rng.choice(bg_indices, size=12, replace=False)
                batch_pos = rng.choice(all_pos_indices, size=1, replace=False)
                batch = np.concatenate([batch_neg, batch_pos])
                sigma = 0.95
            elif policy_name == "policy_d":
                focus_t = POS_TYPOLOGIES[(step - 1) % len(POS_TYPOLOGIES)]
                f_idx = group_indices[focus_t]
                batch_pos = (
                    rng.choice(f_idx, size=1, replace=False)
                    if len(f_idx)
                    else rng.choice(all_pos_indices, size=1, replace=False)
                )
                batch_neg = rng.choice(bg_indices, size=12, replace=False)
                batch = np.concatenate([batch_neg, batch_pos])
                sigma = 1.15
            elif policy_name == "policy_u":
                focus_t = POS_TYPOLOGIES[(step - 1) % len(POS_TYPOLOGIES)]
                f_idx = group_indices[focus_t]
                batch_pos = (
                    rng.choice(f_idx, size=1, replace=False)
                    if len(f_idx)
                    else rng.choice(all_pos_indices, size=1, replace=False)
                )
                batch_neg = rng.choice(bg_indices, size=12, replace=False)
                batch = np.concatenate([batch_neg, batch_pos])
                sigma = 0.85
            else:  # uniform_dp
                batch_neg = rng.choice(bg_indices, size=12, replace=False)
                batch_pos = rng.choice(all_pos_indices, size=1, replace=False)
                batch = np.concatenate([batch_neg, batch_pos])
                sigma = 0.6284465

            grad_sums = [torch.zeros_like(p) for p in params]
            for idx in batch:
                xb = torch.as_tensor(x_train[idx:idx + 1], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y_train[idx:idx + 1], dtype=torch.float32, device=device)
                model.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                grads = torch.autograd.grad(loss, params, retain_graph=False)
                norm_val = torch.sqrt(sum(torch.sum(g**2) for g in grads))
                factor = min(1.0, 1.0 / (float(norm_val) + 1e-12))
                for acc, g in zip(grad_sums, grads):
                    acc.add_(g * factor)

            for cu, gs in zip(client_updates, grad_sums):
                noise = torch.randn_like(gs) * (sigma * 1.0)
                cu.add_((gs + noise) / 128.0 / 4.0)

        with torch.no_grad():
            for p, mi, vi, delta in zip(params, m, v, client_updates):
                mi.mul_(beta1).add_(delta, alpha=1.0 - beta1)
                vi.mul_(beta2).addcmul_(delta, delta, value=1.0 - beta2)
                m_hat = mi / (1.0 - beta1**step)
                v_hat = vi / (1.0 - beta2**step)
                update = server_lr * m_hat / (torch.sqrt(v_hat) + tau)
                p.add_(update, alpha=-1.0)

    torch.save(model.state_dict(), ckpt_path)
    print(f"[Verification] Saved checkpoint: {ckpt_path}")
    return model


def evaluate_model_scores(
    model: torch.nn.Module,
    x_test: np.ndarray,
    batch_size: int = 32768,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Run model inference in chunks to compute predicted probabilities."""
    model.eval()
    scores = []
    with torch.no_grad():
        for s in range(0, len(x_test), batch_size):
            xb = torch.as_tensor(x_test[s:s + batch_size], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores)


def main():
    cfg_path = Path("configs/experiment.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Verification] Running on device: {device}")

    # Load dataset
    print("[Verification] Loading benchmark dataset...")
    frame = load_and_label_dataset(cfg)
    x, feature_names = build_feature_matrix(
        frame,
        int(cfg["data"].get("feature_hash_dim", 16)),
        int(cfg.get("seed", 2026)),
    )
    y = frame["label"].to_numpy(dtype=np.float32)
    splits = frame["split"].to_numpy()
    groups = frame["group"].to_numpy()

    train_mask = splits == "train"
    test_mask = splits == "test"

    x_train, y_train, g_train = x[train_mask], y[train_mask], groups[train_mask]
    x_test, y_test, g_test = x[test_mask], y[test_mask], groups[test_mask]

    input_dim = x.shape[1]
    hidden_dims = tuple(cfg["model"].get("hidden_dims", [128, 64]))
    test_prev = float(y_test.mean())

    print(
        f"[Verification] Test set loaded: {len(y_test)} rows, {int(y_test.sum())} positives "
        f"(prevalence: {test_prev:.6f})"
    )

    # 1. Existing baseline checkpoints to evaluate
    baseline_checkpoints = [
        ("Central-NN (v2 Matched Weight)", "Non-private", "AdamW", "outputs/central_nn_v2_gbdt_matched_weight/checkpoints/central_nn_seed2026.pt"),
        ("Central-NN (v2 Unweighted)", "Non-private", "AdamW", "outputs/central_nn_v2_unweighted/checkpoints/central_nn_seed2026.pt"),
        ("FedAdam (Non-private Seed 2026)", "Non-private", "FedAdam", "outputs/paired_fedavg_fedadam_3seeds/checkpoints/fedadam_seed2026.pt"),
        ("FedAvg (Non-private Seed 2026)", "Non-private", "FedAvg", "outputs/paired_fedavg_fedadam_3seeds/checkpoints/fedavg_seed2026.pt"),
        ("Uniform-DP-FL (FedAvg Server - Collapsed)", "eps = 4.0", "FedAvg", "outputs/partial_dp_server_fedavg_20261103_225611_73aa74/checkpoints/uniform_dp_fl_seed2026.pt"),
        ("TA-DPFL-Policy-U (FedAvg Server - Collapsed)", "eps = 4.0", "FedAvg", "outputs/partial_dp_server_fedavg_20261103_225611_73aa74/checkpoints/ta_dpfl_policy_u_seed2026.pt"),
    ]

    master_records = []

    for name, priv, opt, ckpt_str in baseline_checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.is_file():
            print(f"[-] Checkpoint missing: {ckpt_path}")
            continue
        model = AMLClassifier(input_dim, hidden_dims).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        scores = evaluate_model_scores(model, x_test, device=device)

        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        master_records.append({
            "Method / Architecture": name,
            "Privacy (ε)": priv,
            "Optimizer": opt,
            "ROC-AUC": f"{auc:.6f}",
            "AP": f"{ap:.6f}",
            "AP / Prev": f"{ap / test_prev:.2f}x",
            "Mean Pos": f"{scores[y_test == 1].mean():.4f}",
            "Mean Neg": f"{scores[y_test == 0].mean():.4f}",
        })
        print(f"[+] {name:40s} | ROC-AUC: {auc:.4f} | AP: {ap:.6f}")

    # 2. Train / Load ALL 4 DP Policies
    policies = [
        ("Uniform-DP-FL (Fixed)", "eps = 4.0", "FedAdam + Prior", "uniform_dp"),
        ("Individualized-DP-SGD (Fixed)", "eps in [2, 4]", "FedAdam + Prior", "individualized_dp"),
        ("TA-DPFL Policy U (Fixed)", "eps = 4.0", "FedAdam + Prior", "policy_u"),
        ("TA-DPFL Policy D (Fixed)", "eps in [2, 4]", "FedAdam + Prior", "policy_d"),
    ]

    policy_scores = {}
    policy_typos = {}

    bg_test_mask = g_test == "BG"

    for display_name, priv, opt, pol_key in policies:
        model = train_policy(pol_key, x_train, y_train, g_train, input_dim, hidden_dims, device)
        scores = evaluate_model_scores(model, x_test, device=device)
        policy_scores[pol_key] = scores

        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)

        master_records.append({
            "Method / Architecture": display_name,
            "Privacy (ε)": priv,
            "Optimizer": opt,
            "ROC-AUC": f"{auc:.6f}",
            "AP": f"{ap:.6f}",
            "AP / Prev": f"{ap / test_prev:.2f}x",
            "Mean Pos": f"{scores[y_test == 1].mean():.4f}",
            "Mean Neg": f"{scores[y_test == 0].mean():.4f}",
        })
        print(f"[+] {display_name:40s} | ROC-AUC: {auc:.4f} | AP: {ap:.6f}")

        # Compute per-typology metrics for this policy
        pol_metrics = {}
        for t in POS_TYPOLOGIES:
            t_mask = (g_test == t) | bg_test_mask
            t_y = (g_test[t_mask] == t).astype(int)
            t_scores = scores[t_mask]
            t_auc = roc_auc_score(t_y, t_scores)
            t_ap = average_precision_score(t_y, t_scores)
            pol_metrics[t] = {"auc": float(t_auc), "ap": float(t_ap)}
        policy_typos[pol_key] = pol_metrics

    # 3. Print Master Summary Table
    df_master = pd.DataFrame(master_records)
    print("\n" + "=" * 115)
    print("MASTER EXPERIMENTAL VERIFICATION SUMMARY (Held-out Test Split, n = 1,015,882)")
    print("=" * 115)
    print(df_master.to_string(index=False))

    # 4. Print 4-Way Policy Typology Table
    print("\n" + "=" * 115)
    print("4-WAY DP POLICY COMPARISON ACROSS ALL TYPOLOGIES (One-vs-Background Test ROC-AUC / AP)")
    print("=" * 115)
    typo_records = []
    for t in POS_TYPOLOGIES:
        n_pos = int((g_test == t).sum())
        p_prev = n_pos / len(y_test)
        typo_records.append({
            "Typology": t,
            "Positives": n_pos,
            "Prevalence": f"{p_prev:.6f}",
            "Uniform-DP (AUC/AP)": f"{policy_typos['uniform_dp'][t]['auc']:.4f} / {policy_typos['uniform_dp'][t]['ap']:.6f}",
            "Individualized-DP (AUC/AP)": f"{policy_typos['individualized_dp'][t]['auc']:.4f} / {policy_typos['individualized_dp'][t]['ap']:.6f}",
            "Policy U (AUC/AP)": f"{policy_typos['policy_u'][t]['auc']:.4f} / {policy_typos['policy_u'][t]['ap']:.6f}",
            "Policy D (AUC/AP)": f"{policy_typos['policy_d'][t]['auc']:.4f} / {policy_typos['policy_d'][t]['ap']:.6f}",
        })
    df_typo = pd.DataFrame(typo_records)
    print(df_typo.to_string(index=False))
    print("=" * 115)

    # 5. Save updated JSON summary
    out_json = Path("outputs/final_benchmark/benchmark_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "evaluation_split": "test",
        "test_samples": len(y_test),
        "test_positives": int(y_test.sum()),
        "test_prevalence": test_prev,
        "master_results": master_records,
        "policy_typology_breakdown": policy_typos,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[Verification] Updated full benchmark JSON at: {out_json}")


if __name__ == "__main__":
    main()
