from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ta_dpfl_constants import ALL_GROUPS


def load_virtual_client_manifest(path: str | Path) -> dict[str, str]:
    manifest = pd.read_csv(
        path,
        dtype={"source_client_id": "string", "virtual_client_id": "string"},
        keep_default_na=False,
    )
    required = {"source_client_id", "virtual_client_id"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Virtual-client manifest must contain {sorted(required)}")
    if manifest[list(required)].eq("").any().any():
        raise ValueError("Virtual-client manifest contains empty IDs")
    if manifest["source_client_id"].duplicated().any():
        raise ValueError("A source client appears more than once in the virtual-client manifest")
    return dict(zip(
        manifest["source_client_id"].astype(str),
        manifest["virtual_client_id"].astype(str),
    ))


def build_virtual_client_partitions(
    frame: pd.DataFrame,
    split: str,
    manifest_path: str | Path,
) -> dict[str, dict[str, np.ndarray]]:
    """Build {virtual_client: {stratum: global positional row indices}}.

    All source IDs in frame must occur in the fixed manifest. The returned
    indices address the original x/y arrays, so this function requires a
    zero-based RangeIndex. Grouping is performed once rather than rescanning
    the entire dataframe once per virtual client.
    """
    if not isinstance(frame.index, pd.RangeIndex) or frame.index.start != 0 or frame.index.step != 1:
        raise ValueError("frame must have a zero-based RangeIndex aligned with feature arrays")
    if "split" not in frame or "group" not in frame or "client" not in frame:
        raise ValueError("frame must contain split, group, and client columns")

    mapping = load_virtual_client_manifest(manifest_path)
    source_ids = frame["client"].astype(str)
    missing = sorted(set(source_ids.unique()) - set(mapping))
    if missing:
        raise ValueError(
            f"{len(missing)} observed source clients are missing from the manifest; "
            f"examples: {missing[:10]}"
        )

    metadata_path = Path(manifest_path).with_name("virtual_client_manifest_metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected manifest metadata sidecar beside {manifest_path}: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_virtual = int(metadata["num_virtual_clients_requested"])
    virtual_roster = [f"VC{i:04d}" for i in range(n_virtual)]
    if set(mapping.values()) != set(virtual_roster):
        missing_virtual = sorted(set(virtual_roster) - set(mapping.values()))
        unexpected_virtual = sorted(set(mapping.values()) - set(virtual_roster))
        raise ValueError(
            "Manifest virtual-client roster does not match metadata; "
            f"missing={missing_virtual[:10]}, unexpected={unexpected_virtual[:10]}"
        )

    split_mask = frame["split"].eq(split).to_numpy()
    row_indices = np.flatnonzero(split_mask).astype(np.int64, copy=False)
    if row_indices.size == 0:
        raise ValueError(f"No rows found for split={split!r}")

    # Map only this split; preserve the original positional row indices.
    split_source_ids = source_ids.to_numpy()[split_mask]
    split_virtual_ids = pd.Series(split_source_ids, dtype="string").map(mapping)
    if split_virtual_ids.isna().any():
        raise ValueError("One or more split rows could not be mapped to a virtual client")
    split_groups = frame.loc[split_mask, "group"].astype(str).to_numpy()

    partitions: dict[str, dict[str, np.ndarray]] = {
        vc: {g: np.empty(0, dtype=np.int64) for g in ALL_GROUPS}
        for vc in virtual_roster
    }
    grouped = pd.DataFrame({
        "virtual_client": split_virtual_ids.astype(str).to_numpy(),
        "group": split_groups,
        "row_index": row_indices,
    }).groupby(["virtual_client", "group"], sort=False)["row_index"]

    unknown_groups = sorted(set(split_groups) - set(ALL_GROUPS))
    if unknown_groups:
        raise ValueError(f"Unknown stratum labels in frame: {unknown_groups}")
    for (virtual_client, group), indices in grouped:
        partitions[str(virtual_client)][str(group)] = indices.to_numpy(dtype=np.int64, copy=True)

    assigned = sum(len(v) for client_groups in partitions.values() for v in client_groups.values())
    if assigned != len(row_indices):
        raise RuntimeError(f"Partition coverage mismatch: assigned {assigned}, expected {len(row_indices)}")
    return partitions
