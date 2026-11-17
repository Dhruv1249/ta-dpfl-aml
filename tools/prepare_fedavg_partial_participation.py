from pathlib import Path
import copy
import yaml

source = Path("configs/fedavg_uniform_sgd_50x1.yaml")
destination = Path("configs/fedavg_partial4_3200x1.yaml")
if not source.exists():
    raise FileNotFoundError(f"Required base config not found: {source}")
with source.open("r", encoding="utf-8") as f:
    cfg = copy.deepcopy(yaml.safe_load(f))

cfg["output_dir"] = "outputs/fedavg_partial4_3200x1"
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
})
exp = cfg["experiments"]
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
destination.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("Created:", destination)
for k in ["rounds", "local_steps_per_round", "batch_size_nonprivate", "clients_per_round", "aggregation_mode", "nonprivate_optimizer"]:
    print(f"{k}: {fed.get(k)}")
print("DP methods: disabled")
