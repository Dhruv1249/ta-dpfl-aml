from pathlib import Path
import copy
import yaml

SOURCE = Path("configs/fedavg_partial4_3200x1.yaml")
DESTINATION = Path("configs/fedadam_partial4_3200x1.yaml")


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Base configuration not found: {SOURCE}. "
            "Install the partial-participation patch and generate its FedAvg config first."
        )

    with SOURCE.open("r", encoding="utf-8") as f:
        cfg = copy.deepcopy(yaml.safe_load(f))

    cfg["output_dir"] = "outputs/fedadam_partial4_3200x1"

    fed = cfg["federated"]
    fed.update({
        "partition_mode": "virtual",
        "virtual_client_manifest": "outputs/virtual_federation/virtual_client_manifest.csv",
        "max_clients": None,
        "aggregation_mode": "uniform",
        "nonprivate_optimizer": "sgd",
        "rounds": 3200,
        "local_steps_per_round": 1,
        "batch_size_nonprivate": 1024,
        "learning_rate": 0.001,
        "clients_per_round": 4,
        "participation_schedule_seed": 2026,
        "server_learning_rate": 0.001,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 1e-8,
    })

    exp = cfg["experiments"]
    exp.update({
        "run_central_nn": False,
        "run_central_gbdt": False,
        "run_fedavg": False,
        "run_fedadam": True,
        "run_uniform_dp": False,
        "run_individualized_dp": False,
        "run_ta_policy_u": False,
        "run_ta_policy_d": False,
        "random_seeds": [2026],
    })

    DESTINATION.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"Created: {DESTINATION}")
    print("Method: FedAdam (server-side Adam; non-private diagnostic)")
    for key in [
        "rounds", "clients_per_round", "local_steps_per_round",
        "batch_size_nonprivate", "learning_rate", "aggregation_mode",
        "nonprivate_optimizer", "server_learning_rate", "server_beta1",
        "server_beta2", "server_tau",
    ]:
        print(f"{key}: {fed.get(key)}")
    print("DP methods: disabled")


if __name__ == "__main__":
    main()
