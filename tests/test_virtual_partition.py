from __future__ import annotations

import pandas as pd

from tools.build_virtual_clients import build_assignments
from federated.virtual_partition import build_virtual_client_partitions


def test_hash_assignment_is_deterministic_and_roster_complete():
    roster = ["10", "1", "100", "2"]
    first = build_assignments(roster, seed=2026, n_virtual_clients=4)
    second = build_assignments(list(reversed(roster)), seed=2026, n_virtual_clients=4)
    assert first == second
    assert {row["source_client_id"] for row in first} == set(roster)
    assert all(row["virtual_client_id"].startswith("VC") for row in first)


def test_virtual_partition_preserves_original_global_row_indices(tmp_path):
    roster = ["1", "2", "3"]
    rows = build_assignments(roster, seed=7, n_virtual_clients=2)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    (tmp_path / "virtual_client_manifest_metadata.json").write_text('{"num_virtual_clients_requested": 2}', encoding="utf-8")

    frame = pd.DataFrame({
        "client": ["1", "2", "1", "3", "2"],
        "group": ["BG", "CYCLE", "BG", "UNCL", "BG"],
        "split": ["train", "train", "test", "train", "validation"],
    })

    partitions = build_virtual_client_partitions(frame, "train", manifest)
    actual = sorted(i for groups in partitions.values() for arr in groups.values() for i in arr.tolist())
    assert actual == [0, 1, 3]
    assert sum(len(arr) for groups in partitions.values() for arr in groups.values()) == 3
