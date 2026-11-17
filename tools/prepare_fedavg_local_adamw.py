from pathlib import Path
import copy
import yaml

candidates = [
    Path("configs/fedavg_uniform_batch1024.yaml"),
    Path("configs/fedavg_virtual_learning.yaml"),
]
source = next((p for p in candidates if p.exists()), None)
if source is None:
    raise FileNotFoundError(
        "Expected configs/fedavg_uniform_batch1024.yaml or "
        "configs/fedavg_virtual_learning.yaml."
    )
with source.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg = copy.deepcopy(cfg)
cfg["output_dir"] = "outputs/fedavg_uniform_adamw"
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
    "virtual_client_manifest": "outputs/virtual_federation/virtual_client_manifest.csv",
    "max_clients": None,
    "aggregation_mode": "uniform",
    "rounds": 10,
    "local_steps_per_round": 5,
    "batch_size_nonprivate": 1024,
    "learning_rate": 0.001,
    "nonprivate_optimizer": "adamw",
    "nonprivate_weight_decay": 0.0001,
    "nonprivate_max_grad_norm": 5.0,
})

destination = Path("configs/fedavg_uniform_adamw.yaml")
destination.write_text(
    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
print("Created:", destination)
print("Source config:", source)
print("Optimizer: AdamW (local state resets each client-round)")
print("Aggregation: uniform")
print("Rounds/local steps/batch: 10/5/1024")
print("Weight decay/gradient clip: 1e-4/5.0")
print("All DP methods disabled")
