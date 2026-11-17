from pathlib import Path
import argparse
import math
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Validate the virtual-client FedAvg smoke-test artifacts.")
    parser.add_argument("--output-dir", default="outputs/fedavg_virtual_smoke")
    args = parser.parse_args()
    out = Path(args.output_dir)

    required = [
        out / "overall_results.csv",
        out / "training_log.csv",
        out / "client_training_log.csv",
        out / "checkpoints" / "fedavg_seed2026.pt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("FAIL: missing artifacts:\n" + "\n".join(missing))

    overall = pd.read_csv(out / "overall_results.csv")
    round_log = pd.read_csv(out / "training_log.csv")
    client_log = pd.read_csv(out / "client_training_log.csv")

    errors = []
    if len(overall) != 1 or str(overall.iloc[0].get("method")) != "FedAvg":
        errors.append("overall_results.csv must contain exactly one FedAvg result")
    if len(round_log) != 1:
        errors.append("training_log.csv must contain exactly one round summary")
    else:
        row = round_log.iloc[0]
        if int(row.get("clients", -1)) != 256:
            errors.append("round summary must report 256 clients")
        if int(row.get("total_sampled_records", -1)) != 32768:
            errors.append("expected 32,768 sampled records (256 clients x 128)")
        for key in ["mean_local_loss", "mean_local_update_l2", "aggregated_update_l2"]:
            value = float(row.get(key, float("nan")))
            if not math.isfinite(value):
                errors.append(f"{key} must be finite")
        if float(row.get("aggregated_update_l2", 0.0)) <= 0:
            errors.append("aggregated_update_l2 must be positive")
    if len(client_log) != 256:
        errors.append("client_training_log.csv must contain 256 client rows")
    else:
        if client_log["client_id"].nunique() != 256:
            errors.append("client log must contain 256 unique clients")
        if int(client_log["training_rows"].sum()) != 3_046_861:
            errors.append("client training row counts do not sum to 3,046,861")
        if int(client_log["training_positive_rows"].sum()) != 2_297:
            errors.append("client training positive counts do not sum to 2,297")
        if not (client_log["sampled_records"] == 128).all():
            errors.append("each client should sample exactly 128 records")
        if not client_log["local_update_l2"].map(lambda x: math.isfinite(float(x))).all():
            errors.append("all local_update_l2 values must be finite")

    privacy_files = list((out / "privacy_reports").glob("*.json")) if (out / "privacy_reports").exists() else []
    if privacy_files:
        errors.append("FedAvg-only smoke test should not generate privacy report JSON files")

    if errors:
        print("FAIL")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print("PASS: artifacts, 256 client logs, row coverage, sample counts, and update diagnostics verified.")
    print("This validates integration only; it is not a convergence or DP privacy validation.")


if __name__ == "__main__":
    main()
