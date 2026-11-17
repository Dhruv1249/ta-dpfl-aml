from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import load_and_label_dataset  # noqa: E402
from tools.build_virtual_clients import build_assignments, canonical_roster, sha256_text  # noqa: E402


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Configuration YAML must contain a mapping")
    return config


def _distribution(values: pd.Series) -> dict:
    vals = values.to_numpy(dtype=np.float64)
    if len(vals) == 0:
        return {"min": 0, "median": 0, "p95": 0, "max": 0, "mean": 0, "std": 0, "coefficient_of_variation": None, "max_to_mean": None}
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    return {
        "min": int(vals.min()),
        "median": float(np.median(vals)),
        "p95": float(np.quantile(vals, 0.95)),
        "max": int(vals.max()),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": (std / mean) if mean else None,
        "max_to_mean": (float(vals.max()) / mean) if mean else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit coverage, uniqueness, and workload distribution of virtual-client mapping."
    )
    parser.add_argument("--config", default="configs/experiment.resolved.yaml")
    parser.add_argument("--manifest", default="outputs/virtual_federation/virtual_client_manifest.csv")
    parser.add_argument("--metadata", default="outputs/virtual_federation/virtual_client_manifest_metadata.json")
    parser.add_argument("--output", default="outputs/virtual_federation/virtual_client_audit.json")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    manifest_path = Path(args.manifest)
    metadata_path = Path(args.metadata)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Manifest metadata not found: {metadata_path}")

    manifest = pd.read_csv(
        manifest_path,
        dtype={"source_client_id": "string", "virtual_client_id": "string"},
        keep_default_na=False,
    )
    required = {"source_client_id", "virtual_client_id"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must include columns: {sorted(required)}")
    if manifest[list(required)].eq("").any().any():
        raise ValueError("Manifest contains empty source or virtual client IDs")
    if manifest["source_client_id"].duplicated().any():
        dupes = manifest.loc[manifest["source_client_id"].duplicated(), "source_client_id"].head(10).tolist()
        raise ValueError(f"A source client is assigned more than once: {dupes}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = load_and_label_dataset(config)

    observed_clients = set(frame["client"].astype(str).unique().tolist())
    mapped_clients = set(manifest["source_client_id"].astype(str).tolist())
    missing_clients = sorted(observed_clients - mapped_clients)
    roster_clients_without_rows = sorted(mapped_clients - observed_clients)
    if missing_clients:
        raise ValueError(
            f"{len(missing_clients)} observed source clients are missing from the manifest; "
            f"examples: {missing_clients[:10]}"
        )

    configured_roster = config.get("data", {}).get("public_client_ids")
    if configured_roster:
        configured_roster = canonical_roster(configured_roster)
        manifest_roster_matches_config = set(configured_roster) == mapped_clients
        if not manifest_roster_matches_config:
            raise ValueError("Manifest roster does not match data.public_client_ids in the audit config")
    else:
        manifest_roster_matches_config = None

    mapping = dict(zip(
        manifest["source_client_id"].astype(str),
        manifest["virtual_client_id"].astype(str),
    ))

    # Verify the manifest is exactly the deterministic mapping described by its metadata.
    seed = int(metadata["seed"])
    requested_virtual_count = int(metadata["num_virtual_clients_requested"])
    canonical_ids = canonical_roster(manifest["source_client_id"].astype(str).tolist())
    expected_assignments = build_assignments(canonical_ids, seed, requested_virtual_count)
    expected_mapping = {
        row["source_client_id"]: row["virtual_client_id"] for row in expected_assignments
    }
    if mapping != expected_mapping:
        raise ValueError("Manifest assignments do not match the recorded hash algorithm/seed/count")
    roster_sha = sha256_text(json.dumps(canonical_ids, ensure_ascii=False, separators=(",", ":")))
    if metadata.get("roster_sha256") != roster_sha:
        raise ValueError("Manifest roster SHA-256 does not match the metadata")
    assignment_text = json.dumps(expected_assignments, ensure_ascii=False, separators=(",", ":"))
    if metadata.get("assignment_sha256") != sha256_text(assignment_text):
        raise ValueError("Manifest assignment SHA-256 does not match the metadata")

    # Include every requested public virtual bucket, including any empty bucket.
    virtual_ids = [f"VC{i:04d}" for i in range(requested_virtual_count)]
    unexpected_virtual_ids = sorted(set(mapping.values()) - set(virtual_ids))
    if unexpected_virtual_ids:
        raise ValueError(f"Manifest contains unexpected virtual IDs: {unexpected_virtual_ids[:10]}")
    frame["virtual_client_id"] = pd.Categorical(
        frame["client"].astype(str).map(mapping), categories=virtual_ids
    )
    if frame["virtual_client_id"].isna().any():
        raise ValueError("One or more transactions were not assigned to a virtual client")

    if frame["source_row_id"].duplicated().any():
        raise ValueError("source_row_id is not unique; cannot establish one-to-one transaction assignment")
    # Mapping is a pure row-wise function of each row's source-client ID. Since
    # source_row_id is unique and every client has one mapping, no row is dropped
    # or duplicated by the mapping operation.
    mapped_frame = frame[["virtual_client_id", "split", "label", "group"]]

    # Every source ID maps to one and only one virtual ID.
    source_to_virtual_counts = manifest.groupby("source_client_id")["virtual_client_id"].nunique()
    if not source_to_virtual_counts.eq(1).all():
        raise AssertionError("At least one source client maps to multiple virtual clients")

    # Per-virtual-client audit. This is audit-only; label/group fields do not enter mapping.
    source_counts = manifest.groupby("virtual_client_id").size().rename("source_client_count")
    rows_by_split = (
        mapped_frame.groupby(["virtual_client_id", "split"], observed=False)
        .size().unstack(fill_value=0)
    )
    train_mask = frame["split"].eq("train")
    train = frame.loc[train_mask, ["virtual_client_id", "client", "label", "group"]]
    train_rows = train.groupby("virtual_client_id", observed=True).size().rename("train_rows")
    train_positive = (
        train[train["label"].eq(1)]
        .groupby("virtual_client_id", observed=True).size().rename("train_positive_rows")
    )
    group_counts = (
        train.groupby(["virtual_client_id", "group"], observed=True)
        .size().unstack(fill_value=0)
    )
    # Largest original source-client contribution within each virtual client.
    source_train_counts = train.groupby(["virtual_client_id", "client"], observed=True).size()
    max_source_counts = source_train_counts.groupby(level=0, observed=True).max().rename("max_source_client_train_rows")

    per_client = pd.DataFrame(index=pd.Index(virtual_ids, name="virtual_client_id"))
    for series in [source_counts, train_rows, train_positive, max_source_counts]:
        per_client = per_client.join(series, how="left")
    for col in group_counts.columns:
        per_client[f"train_group_{col}"] = group_counts[col]
    for col in ["train", "validation", "test"]:
        if col in rows_by_split.columns:
            per_client[f"{col}_rows"] = rows_by_split[col]
        else:
            per_client[f"{col}_rows"] = 0
    per_client = per_client.fillna(0)
    per_client["train_positive_fraction"] = np.divide(
        per_client["train_positive_rows"].to_numpy(dtype=float),
        per_client["train_rows"].to_numpy(dtype=float),
        out=np.zeros(len(per_client), dtype=float),
        where=per_client["train_rows"].to_numpy(dtype=float) > 0,
    )
    per_client["largest_source_client_share_of_train"] = np.divide(
        per_client["max_source_client_train_rows"].to_numpy(dtype=float),
        per_client["train_rows"].to_numpy(dtype=float),
        out=np.zeros(len(per_client), dtype=float),
        where=per_client["train_rows"].to_numpy(dtype=float) > 0,
    )

    total_by_split = {
        str(k): int(v) for k, v in frame["split"].value_counts().items()
    }
    group_counts_train = {
        str(k): int(v) for k, v in train["group"].value_counts().items()
    }
    train_dist = _distribution(per_client["train_rows"])
    all_split_row_sums = {
        split_name: int(per_client[f"{split_name}_rows"].sum())
        for split_name in ["train", "validation", "test"]
    }
    checks = {
        "all_observed_source_clients_assigned": len(missing_clients) == 0,
        "manifest_roster_matches_config_public_roster": manifest_roster_matches_config,
        "each_source_client_assigned_once": bool(source_to_virtual_counts.eq(1).all()),
        "each_transaction_assigned_once": bool(frame["source_row_id"].is_unique),
        "all_dataset_rows_preserved": len(mapped_frame) == len(frame),
        "split_row_totals_preserved": all(
            all_split_row_sums.get(split_name, 0) == total_by_split.get(split_name, 0)
            for split_name in ["train", "validation", "test"]
        ),
        "training_positive_total_preserved": int(per_client["train_positive_rows"].sum()) == int(train["label"].sum()),
    }
    passed = all(checks.values())

    report = {
        "status": "PASS" if passed else "FAIL",
        "assignment_method": metadata.get("assignment_method"),
        "seed": metadata.get("seed"),
        "num_virtual_clients_requested": metadata.get("num_virtual_clients_requested"),
        "num_virtual_clients_occupied": len(virtual_ids),
        "num_source_clients_in_manifest": int(len(manifest)),
        "num_observed_source_clients": int(len(observed_clients)),
        "num_public_roster_clients_without_observed_rows": int(len(roster_clients_without_rows)),
        "dataset_rows": int(len(frame)),
        "dataset_positive_rows": int(frame["label"].sum()),
        "dataset_split_rows": total_by_split,
        "mapped_split_row_sums": all_split_row_sums,
        "training_group_counts": group_counts_train,
        "training_positive_rows_by_virtual_client": {
            str(k): int(v) for k, v in per_client["train_positive_rows"].items()
        },
        "training_row_distribution": train_dist,
        "virtual_clients_with_zero_training_rows": int((per_client["train_rows"] == 0).sum()),
        "virtual_clients_with_zero_training_positives": int((per_client["train_positive_rows"] == 0).sum()),
        "max_largest_source_client_share_of_virtual_train": float(per_client["largest_source_client_share_of_train"].max()),
        "checks": checks,
        "all_checks_passed": passed,
        "interpretation": (
            "PASS means all observed source IDs and all processed transactions are assigned exactly once. "
            "It does not mean virtual-client sizes are balanced or that this simulation is equivalent "
            "to 30,470 separately participating institutions. Inspect the workload and concentration metrics."
        ),
        "virtual_client_summary": per_client.reset_index().to_dict(orient="records"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Audit status: {report['status']}")
    print(f"Source clients: {len(observed_clients):,}")
    print(f"Occupied virtual clients: {len(virtual_ids):,}")
    print(f"Dataset rows preserved: {len(mapped_frame):,} / {len(frame):,}")
    print(f"Training rows: {total_by_split.get('train', 0):,}")
    print(f"Training positives: {int(train['label'].sum()):,}")
    print(f"Train rows per virtual client: min={train_dist['min']:,}, median={train_dist['median']:,.1f}, p95={train_dist['p95']:,.1f}, max={train_dist['max']:,}")
    print(f"Train row max/mean: {train_dist['max_to_mean']}")
    print(f"Virtual clients with zero train rows: {report['virtual_clients_with_zero_training_rows']}")
    print(f"Virtual clients with zero train positives: {report['virtual_clients_with_zero_training_positives']}")
    print(f"Audit report: {output_path}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
