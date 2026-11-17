from __future__ import annotations

import copy
from pathlib import Path
import yaml


def _base_config() -> dict:
    source = Path("configs/fedadam_partial4_3200x1.yaml")
    if not source.exists():
        raise FileNotFoundError(
            f"Base config not found: {source}. Keep the successful FedAdam config in configs/."
        )
    with source.open("r", encoding="utf-8") as f:
        return copy.deepcopy(yaml.safe_load(f))


def _configure(cfg: dict, *, rounds: int, output_dir: str, enabled_methods: list[str]) -> dict:
    cfg["output_dir"] = output_dir
    fed = cfg["federated"]
    fed.update({
        "partition_mode": "virtual",
        "virtual_client_manifest": "outputs/virtual_federation/virtual_client_manifest.csv",
        "max_clients": None,
        "rounds": int(rounds),
        "local_steps_per_round": 1,
        "clients_per_round": 4,
        "participation_schedule_seed": 2026,
        "aggregation_mode": "uniform",
        "learning_rate": 0.001,
        "batch_size_nonprivate": 1024,
        "dp_server_optimizer": "fedadam",
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
        "run_fedadam": False,
        "run_uniform_dp": "uniform_dp" in enabled_methods,
        "run_individualized_dp": "individualized_dp" in enabled_methods,
        "run_ta_policy_u": "ta_policy_u" in enabled_methods,
        "run_ta_policy_d": "ta_policy_d" in enabled_methods,
        "random_seeds": [2026],
    })
    cfg["privacy"]["require_opacus_validation"] = True
    cfg["privacy"]["secure_randomness"] = True
    return cfg


def _write(cfg: dict, path: str) -> None:
    destination = Path(path)
    destination.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Created: {destination}")
    print(f"Output: {cfg['output_dir']}")
    print(f"Rounds: {cfg['federated']['rounds']}; clients/round: 4; local steps: 1")
    print(f"Enabled DP methods: {[k for k in ('run_uniform_dp','run_individualized_dp','run_ta_policy_u','run_ta_policy_d') if cfg['experiments'][k]]}")
    print("Server optimizer: FedAdam (post-processing of protected client updates)")
    print("Non-private baselines: disabled; secure randomness and Opacus validation required")


def main() -> None:
    base = _base_config()
    # Short end-to-end check: all 256 clients participate once. This is only a
    # pipeline smoke run; it is not the paper's full training budget.
    smoke = _configure(
        copy.deepcopy(base),
        rounds=64,
        output_dir="outputs/partial_dp_fedadam_smoke_64x1",
        enabled_methods=["ta_policy_u"],
    )
    _write(smoke, "configs/partial_dp_fedadam_smoke_64x1.yaml")

    # Full budget config for the complete comparison. Do not launch until the
    # one-method smoke run and its accountant report have been reviewed.
    full = _configure(
        copy.deepcopy(base),
        rounds=3200,
        output_dir="outputs/partial_dp_fedadam_3200x1",
        enabled_methods=["uniform_dp", "individualized_dp", "ta_policy_u", "ta_policy_d"],
    )
    _write(full, "configs/partial_dp_fedadam_3200x1.yaml")
    print("\nImportant: run the 64-round Policy-U smoke configuration first.")
    print("The four-method full configuration composes all enabled releases; review its joint_release epsilon.")


if __name__ == "__main__":
    main()
