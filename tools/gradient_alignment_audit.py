from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable
import sys

# Ensure the project root is importable when this file is executed directly
# as: python .\tools\gradient_alignment_audit.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from data.preprocessing import load_and_label_dataset, build_feature_matrix
from federated.virtual_partition import build_virtual_client_partitions
from models.aml_classifier import AMLClassifier


def flatten_gradients(loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]) -> np.ndarray:
    grads = torch.autograd.grad(loss, tuple(parameters), allow_unused=True)
    chunks = []
    for grad, param in zip(grads, parameters):
        if grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float64))
        else:
            chunks.append(grad.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
    return torch.cat(chunks).numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None,
                "min": None, "max": None, "negative_count": 0, "negative_fraction": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "negative_count": int(np.sum(values < 0)),
        "negative_fraction": float(np.mean(values < 0)),
    }


def client_indices(groups: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(v, dtype=np.int64) for v in groups.values() if len(v)]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)


def gradient_for_batch(
    model: torch.nn.Module,
    x_array: np.ndarray,
    y_array: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    params = tuple(model.parameters())
    xb = torch.as_tensor(x_array[indices], dtype=torch.float32, device=device)
    yb = torch.as_tensor(y_array[indices], dtype=torch.float32, device=device)
    logits = model(xb)
    loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="mean")
    return flatten_gradients(loss, params)


def mean_class_gradient(
    model: torch.nn.Module,
    x_array: np.ndarray,
    y_array: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("Cannot calculate a class gradient for an empty index set")
    params = tuple(model.parameters())
    total = np.zeros(sum(p.numel() for p in params), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        xb = torch.as_tensor(x_array[batch_indices], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y_array[batch_indices], dtype=torch.float32, device=device)
        logits = model(xb)
        loss_sum = F.binary_cross_entropy_with_logits(logits, yb, reduction="sum")
        total += flatten_gradients(loss_sum, params)
    return total / float(len(indices))


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch versions
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)!r}")
    return state


def analyze_model_state(
    label: str,
    model: torch.nn.Module,
    client_batches: list[dict],
    x_array: np.ndarray,
    y_array: np.ndarray,
    train_indices: np.ndarray,
    negative_indices_for_gradient: np.ndarray,
    device: torch.device,
    class_gradient_batch_size: int,
) -> tuple[dict, list[dict]]:
    model.eval()
    client_gradients = []
    per_client_rows = []

    with torch.enable_grad():
        for item in client_batches:
            grad = gradient_for_batch(
                model, x_array, y_array, item["batch_indices"], device
            )
            client_gradients.append(grad)
            per_client_rows.append({
                "model_state": label,
                "client_id": item["client_id"],
                "training_rows": item["training_rows"],
                "training_positive_rows": item["training_positive_rows"],
                "sampled_rows": item["sampled_rows"],
                "sampled_positive_rows": item["sampled_positive_rows"],
                "gradient_l2": float(np.linalg.norm(grad)),
            })

        grad_matrix = np.stack(client_gradients, axis=0)
        client_sizes = np.asarray([x["training_rows"] for x in client_batches], dtype=np.float64)
        uniform_reference = np.mean(grad_matrix, axis=0)
        record_weighted_reference = np.average(grad_matrix, axis=0, weights=client_sizes)

        uniform_cosines = np.asarray([cosine(g, uniform_reference) for g in grad_matrix])
        record_cosines = np.asarray([cosine(g, record_weighted_reference) for g in grad_matrix])
        sampled_positive_mask = np.asarray([
            x["sampled_positive_rows"] > 0 for x in client_batches
        ], dtype=bool)

        for row, c_uniform, c_record in zip(per_client_rows, uniform_cosines, record_cosines):
            row["cosine_to_uniform_client_reference"] = c_uniform
            row["cosine_to_record_weighted_reference"] = c_record

        # Global class-conditional gradients. The positive gradient uses all
        # training positives; the background gradient uses a fixed random
        # sample of training negatives. Their combination is weighted by the
        # actual training prevalence.
        positive_indices = train_indices[y_array[train_indices] == 1]
        positive_gradient = mean_class_gradient(
            model, x_array, y_array, positive_indices, device, class_gradient_batch_size
        )
        background_gradient = mean_class_gradient(
            model, x_array, y_array, negative_indices_for_gradient, device,
            class_gradient_batch_size,
        )

    prevalence = float(len(positive_indices) / len(train_indices))
    positive_component = prevalence * positive_gradient
    background_component = (1.0 - prevalence) * background_gradient
    population_gradient = positive_component + background_component

    positive_norm = float(np.linalg.norm(positive_component))
    background_norm = float(np.linalg.norm(background_component))

    summary = {
        "model_state": label,
        "clients": int(len(client_batches)),
        "parameter_count": int(grad_matrix.shape[1]),
        "uniform_client_reference_gradient_l2": float(np.linalg.norm(uniform_reference)),
        "record_weighted_reference_gradient_l2": float(np.linalg.norm(record_weighted_reference)),
        "cosine_uniform_vs_record_weighted_reference": cosine(
            uniform_reference, record_weighted_reference
        ),
        "client_cosine_to_uniform_reference": summarize(uniform_cosines),
        "client_cosine_to_record_weighted_reference": summarize(record_cosines),
        "clients_with_sampled_positive": int(sampled_positive_mask.sum()),
        "clients_without_sampled_positive": int((~sampled_positive_mask).sum()),
        "client_cosine_to_uniform_reference_with_sampled_positive": summarize(
            uniform_cosines[sampled_positive_mask]
        ),
        "client_cosine_to_uniform_reference_without_sampled_positive": summarize(
            uniform_cosines[~sampled_positive_mask]
        ),
        "training_positive_prevalence": prevalence,
        "positive_gradient_per_example_l2": float(np.linalg.norm(positive_gradient)),
        "background_gradient_per_example_l2": float(np.linalg.norm(background_gradient)),
        "positive_prevalence_weighted_gradient_component_l2": positive_norm,
        "background_prevalence_weighted_gradient_component_l2": background_norm,
        "positive_to_background_component_norm_ratio": (
            positive_norm / background_norm if background_norm > 0 else None
        ),
        "cosine_positive_vs_background_gradient": cosine(
            positive_gradient, background_gradient
        ),
        "cosine_uniform_client_reference_vs_population_gradient": cosine(
            uniform_reference, population_gradient
        ),
        "cosine_record_weighted_reference_vs_population_gradient": cosine(
            record_weighted_reference, population_gradient
        ),
        "population_gradient_l2": float(np.linalg.norm(population_gradient)),
        "background_gradient_sample_size": int(len(negative_indices_for_gradient)),
        "positive_gradient_sample_size": int(len(positive_indices)),
    }
    return summary, per_client_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-private FedAvg gradient-alignment audit; does not train or modify the model."
    )
    parser.add_argument("--config", default="configs/fedavg_uniform_sgd_50x1.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/gradient_alignment_audit")
    parser.add_argument("--client-batch-size", type=int, default=1024)
    parser.add_argument("--background-sample-size", type=int, default=65536)
    parser.add_argument("--class-gradient-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.client_batch_size < 1 or args.background_sample_size < 1 or args.class_gradient_batch_size < 1:
        raise ValueError("All batch/sample sizes must be positive")

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    fed_cfg = cfg["federated"]
    model_cfg = cfg["model"]
    seed = int(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        Path(cfg["output_dir"]) / "checkpoints" / f"fedavg_seed{seed}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"FedAvg checkpoint not found: {checkpoint}")

    if str(fed_cfg.get("partition_mode", "institutional")).lower() != "virtual":
        raise ValueError("This audit expects federated.partition_mode: virtual")
    manifest_path = Path(fed_cfg["virtual_client_manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Virtual-client manifest not found: {manifest_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device(args.device)

    print("Loading and preprocessing the benchmark...")
    frame = load_and_label_dataset(cfg)
    x_array, feature_names = build_feature_matrix(
        frame,
        int(data_cfg.get("feature_hash_dim", 16)),
        int(cfg.get("seed", seed)),
    )
    y_array = frame["label"].to_numpy(dtype=np.float32)
    train_indices = np.flatnonzero(frame["split"].eq("train").to_numpy())
    positive_indices = train_indices[y_array[train_indices] == 1]
    negative_indices = train_indices[y_array[train_indices] == 0]
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        raise ValueError("Training split must contain both positive and background records")

    partition_map = build_virtual_client_partitions(
        frame, "train", manifest_path
    )
    client_ids = sorted(partition_map)
    if len(client_ids) != 256:
        print(f"WARNING: expected 256 virtual clients based on the prior audit; found {len(client_ids)}")

    # Fix the diagnostic batches once; the exact same rows are evaluated at
    # initialization and at the trained checkpoint.
    rng = np.random.default_rng(seed)
    client_batches = []
    for client_id in client_ids:
        indices = client_indices(partition_map[client_id])
        if len(indices) == 0:
            continue
        sample_size = min(args.client_batch_size, len(indices))
        batch_indices = rng.choice(indices, size=sample_size, replace=False)
        client_batches.append({
            "client_id": str(client_id),
            "training_rows": int(len(indices)),
            "training_positive_rows": int(np.sum(y_array[indices])),
            "sampled_rows": int(sample_size),
            "sampled_positive_rows": int(np.sum(y_array[batch_indices])),
            "batch_indices": batch_indices.astype(np.int64, copy=False),
        })

    if len(client_batches) < 2:
        raise ValueError("At least two non-empty virtual clients are required")

    bg_rng = np.random.default_rng(seed + 1)
    background_indices = bg_rng.choice(
        negative_indices,
        size=min(args.background_sample_size, len(negative_indices)),
        replace=False,
    ).astype(np.int64, copy=False)

    input_dim = int(x_array.shape[1])
    hidden_dims = tuple(model_cfg.get("hidden_dims", [128, 64]))
    model_factory = lambda: AMLClassifier(input_dim, hidden_dims)

    # run_federated seeds its initial model from the run seed. Reconstruct
    # that initialization so the initial-vs-final comparison is reproducible.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model_factory().to(device)
    initial_state = copy.deepcopy(model.state_dict())
    trained_state = load_checkpoint(checkpoint, device)

    print(f"Device: {device}; rows: {len(frame)}; features: {len(feature_names)}; clients: {len(client_batches)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Client diagnostic batch: up to {args.client_batch_size} rows/client")
    print(f"Positive gradient uses all {len(positive_indices)} training positives")
    print(f"Background gradient uses {len(background_indices)} sampled training negatives")

    all_summaries = []
    all_client_rows = []
    for label, state in [("initial_seeded_model", initial_state), ("trained_checkpoint", trained_state)]:
        model.load_state_dict(state)
        summary, rows = analyze_model_state(
            label, model, client_batches, x_array, y_array, train_indices,
            background_indices, device, args.class_gradient_batch_size,
        )
        all_summaries.append(summary)
        all_client_rows.extend(rows)
        print(f"\n=== {label.upper()} ===")
        for key in [
            "cosine_uniform_vs_record_weighted_reference",
            "client_cosine_to_uniform_reference",
            "client_cosine_to_uniform_reference_with_sampled_positive",
            "client_cosine_to_uniform_reference_without_sampled_positive",
            "cosine_positive_vs_background_gradient",
            "positive_to_background_component_norm_ratio",
            "cosine_uniform_client_reference_vs_population_gradient",
        ]:
            print(f"{key}: {summary[key]}")

    summary_path = output_dir / "gradient_alignment_summary.json"
    client_path = output_dir / "client_gradient_alignment.csv"
    summary_path.write_text(json.dumps({
        "audit_type": "non_private_gradient_alignment_diagnostic",
        "privacy_claim": "none; this is a diagnostic on training data and must not be described as a DP release",
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "seed": seed,
        "client_batch_size": args.client_batch_size,
        "background_sample_size": len(background_indices),
        "class_gradient_batch_size": args.class_gradient_batch_size,
        "training_rows": int(len(train_indices)),
        "training_positive_rows": int(len(positive_indices)),
        "virtual_clients": int(len(client_batches)),
        "model_states": all_summaries,
    }, indent=2), encoding="utf-8")
    pd.DataFrame(all_client_rows).to_csv(client_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {client_path}")


if __name__ == "__main__":
    main()
