from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare matched multi-seed FedAvg/FedAdam non-private runs."
    )
    parser.add_argument(
        "--source-config",
        default="configs/fedadam_partial4_3200x1.yaml",
        help="Existing validated partial-participation FedAdam config.",
    )
    parser.add_argument(
        "--output-config",
        default="configs/paired_fedavg_fedadam_3seeds.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paired_fedavg_fedadam_3seeds",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2026, 2027, 2028],
    )
    args = parser.parse_args()

    source = Path(args.source_config)
    if not source.exists():
        raise FileNotFoundError(
            f"Source config not found: {source}. Install the FedAdam patch first."
        )
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Source config must contain a YAML mapping")
    cfg = copy.deepcopy(cfg)

    if not cfg.get("data", {}).get("public_client_ids"):
        raise ValueError("The source config must contain the audited fixed public client roster")
    manifest = Path(
        cfg.get("federated", {}).get(
            "virtual_client_manifest",
            "outputs/virtual_federation/virtual_client_manifest.csv",
        )
    )
    if not manifest.exists():
        raise FileNotFoundError(
            f"Virtual-client manifest not found: {manifest}. Run the virtual-client audit/build first."
        )

    seeds = list(dict.fromkeys(int(s) for s in args.seeds))
    if len(seeds) < 2:
        raise ValueError("Use at least two independent seeds for reproducibility assessment")

    cfg["output_dir"] = args.output_dir
    cfg["seed"] = seeds[0]

    fed = cfg["federated"]
    fed.update(
        {
            "partition_mode": "virtual",
            "virtual_client_manifest": str(manifest).replace("\\", "/"),
            "max_clients": None,
            "aggregation_mode": "uniform",
            "clients_per_round": 4,
            "rounds": 3200,
            "local_steps_per_round": 1,
            "batch_size_nonprivate": 1024,
            "learning_rate": 0.001,
            "nonprivate_optimizer": "sgd",
            "participation_schedule_seed": None,
            "server_learning_rate": 0.001,
            "server_beta1": 0.9,
            "server_beta2": 0.99,
            "server_tau": 1e-8,
        }
    )

    exp = cfg["experiments"]
    exp.update(
        {
            "run_central_nn": False,
            "run_central_gbdt": False,
            "run_fedavg": True,
            "run_fedadam": True,
            "run_uniform_dp": False,
            "run_individualized_dp": False,
            "run_ta_policy_u": False,
            "run_ta_policy_d": False,
            "random_seeds": seeds,
        }
    )

    # Preserve a clear local-only status in the generated run plan.
    cfg["paired_reproducibility"] = {
        "enabled": True,
        "methods": ["FedAvg", "FedAdam"],
        "seeds": seeds,
        "same_initialization_per_seed": True,
        "same_client_schedule_per_seed": True,
        "same_sampled_batch_per_client_round": True,
        "schedule_seed_mode": "experiment_seed",
        "privacy_status": "non_private_diagnostic_only",
    }

    destination = Path(args.output_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected_existing = [
        item for item in output_dir.iterdir()
        if item.name != "paired_run_plan.json"
    ]
    if unexpected_existing:
        preview = ", ".join(str(item.name) for item in unexpected_existing[:5])
        raise FileExistsError(
            f"Output directory {output_dir} already contains run artifacts ({preview}). "
            "Choose a fresh --output-dir or move the existing run before preparing a new run."
        )
    plan = {
        "source_config": str(source),
        "output_config": str(destination),
        "output_dir": str(output_dir),
        "methods": ["FedAvg", "FedAdam"],
        "seeds": seeds,
        "federation": {
            "virtual_clients": 256,
            "clients_per_round": 4,
            "rounds": 3200,
            "local_steps_per_round": 1,
            "batch_size": 1024,
            "aggregation": "uniform",
        },
        "local_optimizer": {"name": "SGD", "learning_rate": 0.001},
        "server_optimizer": {
            "name": "FedAdam",
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.99,
            "tau": 1e-8,
        },
        "pairing": {
            "initialization": "same within each seed",
            "client_schedule": "same within each seed; schedule seed equals experiment seed",
            "sampled_batches": "same within each seed/method pair; sample-index SHA-256 fingerprints logged",
        },
        "expected_training_log_rows": len(seeds) * 2 * 3200,
        "expected_client_log_rows": len(seeds) * 2 * 3200 * 4,
        "privacy_status": "non_private_diagnostic_only; no DP claim",
        "cautions": [
            "Virtual clients are deterministic groups of source IDs, not 256 actual institutions.",
            "The four-client participation protocol is a non-private diagnostic; partial-participation DP accounting is not implemented.",
            "Sample-index hashes are local audit metadata, not a DP release.",
        ],
    }
    (output_dir / "paired_run_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )

    print(f"Created configuration: {destination}")
    print(f"Created run plan: {output_dir / 'paired_run_plan.json'}")
    print(f"Seeds: {seeds}")
    print("Methods: FedAvg + FedAdam")
    print("DP methods: disabled")
    print("Participation schedule seed: derived from each experiment seed")


if __name__ == "__main__":
    main()
