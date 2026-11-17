from pathlib import Path
import yaml

SOURCE = Path("configs/experiment.resolved.yaml")
DEST = Path("configs/fedavg_virtual_smoke.yaml")

with SOURCE.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["output_dir"] = "outputs/fedavg_virtual_smoke"
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
fed = cfg["federated"]
fed.update({
    "partition_mode": "virtual",
    "virtual_client_manifest": "outputs/virtual_federation/virtual_client_manifest.csv",
    "max_clients": None,
    "rounds": 1,
    "local_steps_per_round": 1,
    "batch_size_nonprivate": 128,
    "learning_rate": 0.001,
})

DEST.parent.mkdir(parents=True, exist_ok=True)
DEST.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"Created {DEST}")
print("Methods: FedAvg only; all DP methods disabled")
print("Partition: 256 audited virtual clients")
print("Rounds: 1 | local steps/client: 1 | batch: 128 | learning rate: 0.001")
