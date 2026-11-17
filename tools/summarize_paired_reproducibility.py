from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["FedAvg", "FedAdam"]
METRICS = ["AP", "ROC_AUC", "precision", "recall", "F1", "false_positive_rate"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate matched FedAvg/FedAdam runs and summarize multi-seed metrics."
    )
    parser.add_argument(
        "--output-dir", default="outputs/paired_fedavg_fedadam_3seeds"
    )
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--rounds", type=int, default=3200)
    parser.add_argument("--clients-per-round", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.output_dir)
    expected_seeds = [int(x) for x in args.expected_seeds]
    overall_path = out / "overall_results.csv"
    typology_path = out / "typology_results.csv"
    train_path = out / "training_log.csv"
    client_path = out / "client_training_log.csv"

    for path in [overall_path, typology_path, train_path, client_path]:
        require(path.exists(), f"Missing required output: {path}")

    overall = pd.read_csv(overall_path)
    typology = pd.read_csv(typology_path)
    train = pd.read_csv(train_path)
    clients = pd.read_csv(client_path, dtype={"client_id": str})

    require(set(overall["method"].unique()) == set(METHODS), "Overall results must contain exactly FedAvg and FedAdam")
    require(set(overall["seed"].astype(int).unique()) == set(expected_seeds), "Overall results do not contain the expected seeds")
    expected_pairs = {(m, s) for m in METHODS for s in expected_seeds}
    observed_pairs = set(zip(overall["method"], overall["seed"].astype(int)))
    require(observed_pairs == expected_pairs, "Overall results are missing or duplicating method/seed combinations")
    require(len(overall) == len(expected_pairs), "Expected exactly one overall test result per method and seed")

    expected_round_rows = len(expected_seeds) * len(METHODS) * int(args.rounds)
    require(len(train) == expected_round_rows, f"Expected {expected_round_rows} training-log rows, found {len(train)}")
    expected_client_rows = expected_round_rows * int(args.clients_per_round)
    require(len(clients) == expected_client_rows, f"Expected {expected_client_rows} client-log rows, found {len(clients)}")

    required_train_cols = {"round", "method", "seed", "clients", "total_sampled_records", "initial_model_sha256"}
    required_client_cols = {"client_id", "round", "method", "seed", "sampled_records", "sampled_indices_sha256", "initial_model_sha256"}
    require(required_train_cols.issubset(train.columns), f"Training log missing columns: {sorted(required_train_cols - set(train.columns))}")
    require(required_client_cols.issubset(clients.columns), f"Client log missing columns: {sorted(required_client_cols - set(clients.columns))}")

    # Validate each method/seed run and the matched data path.
    schedule_checks = []
    initial_hash_checks = []
    sample_hash_checks = []
    per_run_checks = []
    for seed in expected_seeds:
        schedule_path = out / f"client_participation_schedule_seed{seed}.csv"
        require(schedule_path.exists(), f"Missing participation schedule: {schedule_path}")
        sched = pd.read_csv(schedule_path, dtype={"client_id": str})
        require(len(sched) == int(args.rounds) * int(args.clients_per_round), f"Wrong schedule row count for seed {seed}")
        require(sched["round"].nunique() == int(args.rounds), f"Wrong number of rounds in schedule for seed {seed}")
        counts = sched.groupby("client_id").size()
        require(len(counts) == 256, f"Expected 256 clients in schedule for seed {seed}; got {len(counts)}")
        require(counts.nunique() == 1, f"Client appearances are not balanced for seed {seed}")
        require(int(counts.iloc[0]) == int(args.rounds) * int(args.clients_per_round) // 256, f"Unexpected appearances per client for seed {seed}")
        per_round = sched.groupby("round")["client_id"].nunique()
        require((per_round == int(args.clients_per_round)).all(), f"Incorrect cohort size in schedule for seed {seed}")
        schedule_checks.append({"seed": seed, "schedule_rows": len(sched), "clients": len(counts), "appearances_per_client": int(counts.iloc[0]), "rounds": int(args.rounds)})

        run_hashes = {}
        for method in METHODS:
            tr = train[(train.method == method) & (train.seed.astype(int) == seed)].sort_values("round")
            cr = clients[(clients.method == method) & (clients.seed.astype(int) == seed)].copy()
            require(len(tr) == int(args.rounds), f"Wrong training-log rows for {method}/{seed}")
            require(tr["round"].astype(int).tolist() == list(range(1, int(args.rounds) + 1)), f"Missing/duplicate rounds for {method}/{seed}")
            require((tr["clients"].astype(int) == int(args.clients_per_round)).all(), f"Wrong clients-per-round for {method}/{seed}")
            require(len(cr) == int(args.rounds) * int(args.clients_per_round), f"Wrong client-log rows for {method}/{seed}")
            require(cr.groupby("round")["client_id"].nunique().eq(int(args.clients_per_round)).all(), f"Wrong client cohort count in logs for {method}/{seed}")
            require(cr["sampled_indices_sha256"].notna().all(), f"Missing sampled-batch fingerprints for {method}/{seed}")
            init_hashes = set(tr["initial_model_sha256"].dropna().astype(str))
            require(len(init_hashes) == 1, f"Initial model fingerprint changes within {method}/{seed}")
            run_hashes[method] = next(iter(init_hashes))
            total_sampled = int(tr["total_sampled_records"].sum())
            per_run_checks.append({"method": method, "seed": seed, "rounds": len(tr), "client_rows": len(cr), "sampled_records": total_sampled})

        same_init = run_hashes["FedAvg"] == run_hashes["FedAdam"]
        require(same_init, f"FedAvg and FedAdam did not start from identical weights for seed {seed}")
        initial_hash_checks.append({"seed": seed, "same_initial_model": True, "sha256": run_hashes["FedAvg"]})

        a = clients[(clients.method == "FedAvg") & (clients.seed.astype(int) == seed)][
            ["client_id", "round", "sampled_records", "sampled_positive_records", "sampled_indices_sha256"]
        ].copy()
        b = clients[(clients.method == "FedAdam") & (clients.seed.astype(int) == seed)][
            ["client_id", "round", "sampled_records", "sampled_positive_records", "sampled_indices_sha256"]
        ].copy()
        pair = a.merge(b, on=["client_id", "round"], suffixes=("_fedavg", "_fedadam"), validate="one_to_one")
        require(len(pair) == int(args.rounds) * int(args.clients_per_round), f"Paired client-round rows missing for seed {seed}")
        same_hash = pair["sampled_indices_sha256_fedavg"].astype(str).eq(pair["sampled_indices_sha256_fedadam"].astype(str))
        same_count = pair["sampled_records_fedavg"].eq(pair["sampled_records_fedadam"])
        same_pos = pair["sampled_positive_records_fedavg"].eq(pair["sampled_positive_records_fedadam"])
        require(same_hash.all() and same_count.all() and same_pos.all(), f"FedAvg/FedAdam sampled batches differ for seed {seed}")
        sample_hash_checks.append({"seed": seed, "paired_client_rounds": len(pair), "identical_sample_hashes": int(same_hash.sum()), "all_batches_identical": True})

    # Overall means/standard deviations across seeds.
    metric_cols = [m for m in METRICS if m in overall.columns]
    summary = overall.groupby("method")[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "multi_seed_metric_summary.csv", index=False)

    # Paired per-seed differences: FedAdam minus FedAvg.
    paired = overall.pivot(index="seed", columns="method", values=metric_cols)
    diff = pd.DataFrame(index=paired.index)
    for metric in metric_cols:
        diff[f"{metric}_FedAdam_minus_FedAvg"] = paired[(metric, "FedAdam")] - paired[(metric, "FedAvg")]
    diff = diff.reset_index()
    diff.to_csv(out / "paired_metric_differences.csv", index=False)
    diff_columns = [c for c in diff.columns if c != "seed"]
    diff_summary = pd.DataFrame({
        "metric_difference": diff_columns,
        "mean": [diff[c].mean() for c in diff_columns],
        "std": [diff[c].std() for c in diff_columns],
        "count": [int(diff[c].count()) for c in diff_columns],
    })
    diff_summary.to_csv(out / "paired_metric_difference_summary.csv", index=False)

    # Report mean/std of typology AP and ROC-AUC over seeds.
    typ_metrics = [m for m in ["AP", "ROC_AUC"] if m in typology.columns]
    typ_summary = typology.groupby(["method", "typology"])[typ_metrics].agg(["mean", "std", "count"])
    typ_summary.columns = [f"{metric}_{stat}" for metric, stat in typ_summary.columns]
    typ_summary = typ_summary.reset_index()
    typ_summary.to_csv(out / "multi_seed_typology_summary.csv", index=False)

    audit = {
        "status": "PASS",
        "privacy_status": "non_private_diagnostic_only; no DP guarantee",
        "methods": METHODS,
        "seeds": expected_seeds,
        "schedule_checks": schedule_checks,
        "initialization_pairing": initial_hash_checks,
        "sampled_batch_pairing": sample_hash_checks,
        "run_checks": per_run_checks,
        "summary_files": [
            "multi_seed_metric_summary.csv",
            "paired_metric_differences.csv",
            "paired_metric_difference_summary.csv",
            "multi_seed_typology_summary.csv",
        ],
        "interpretation": "Pairing is verified by identical initial-state fingerprints and per-client-round sampled-index hashes within each seed. This is a reproducibility audit, not a DP release or a statistical significance test.",
    }
    (out / "paired_reproducibility_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("PAIRED REPRODUCIBILITY AUDIT: PASS")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Seeds: {expected_seeds}")
    print(f"Training-log rows: {len(train)}")
    print(f"Client-log rows: {len(clients)}")
    print("Initial model weights: identical within each seed/method pair")
    print("Sampled batches: identical for every paired client-round")
    print("Wrote multi_seed_metric_summary.csv")
    print("Wrote paired_metric_differences.csv")
    print("Wrote paired_metric_difference_summary.csv")
    print("Wrote multi_seed_typology_summary.csv")
    print("Wrote paired_reproducibility_audit.json")


if __name__ == "__main__":
    main()
