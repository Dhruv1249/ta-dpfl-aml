import pandas as pd
import pytest

from federated.partition import build_client_partitions


def test_public_roster_includes_empty_clients_and_checks_all_rows():
    frame = pd.DataFrame({
        "client": ["A", "B", "C"],
        "group": ["CYCLE", "BG", "BG"],
        "split": ["train", "train", "test"],
    })
    parts = build_client_partitions(frame, public_client_ids=["A", "B", "C", "D"])
    assert set(parts) == {"A", "B", "C", "D"}
    assert len(parts["C"]["BG"]) == 0
    assert len(parts["D"]["BG"]) == 0
    with pytest.raises(ValueError):
        build_client_partitions(frame, public_client_ids=["A", "B"])
