from pathlib import Path
import copy
import yaml

SOURCE = Path("configs/fedavg_virtual_learning.yaml")
if not SOURCE.exists():
    raise FileNotFoundError(
        f"Missing {SOURCE}. Create it from the prior FedAvg learning configuration first."
    )

with SOURCE.open("r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

variants = {
    "uniform": "outputs/fedavg_ablation_uniform",
    "sample_size": "outputs/fedavg_ablation_sample_size",
}

for aggregation, output_dir in variants.items():
    cfg = copy.deepcopy(base)
    exp = cfg.setdefault("experiments", {})
    exp.update({
        "run_central_nn": False,
        "run_central_gbdt": False,
        "run_fedavg": True,
        "run_uniform_dp": False,
        "run_individualized_dp": False,
        "run_ta_policy_u": False,
        "run_ta_policy_d": False,
        "random_seeds": [2026],
    })

    fed = cfg.setdefault("federated", {})
    fed.update({
        "partition_mode": "virtual",
        "virtual_client_manifest": (
            "outputs/virtual_federation/virtual_client_manifest.csv"
        ),
        "max_clients": None,
        "rounds": 10,
        "local_steps_per_round": 5,
        "batch_size_nonprivate": 128,
        "learning_rate": 0.001,
        "aggregation_mode": aggregation,
    })

    cfg["output_dir"] = output_dir
    destination = Path(f"configs/fedavg_ablation_{aggregation}.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Created {destination}")
    print(f"  aggregation_mode={aggregation}")
    print(f"  output_dir={output_dir}")
