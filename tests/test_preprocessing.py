import pandas as pd

from data.preprocessing import load_and_label_dataset, build_feature_matrix


def test_typology_uncl_and_feature_exclusion(tmp_path):
    rows = []
    for i in range(10):
        rows.append({
            "Timestamp": f"2020-01-{i+1:02d} 00:00:00",
            "From Bank": f"BANK-{i % 2}",
            "Amount Received": float(i + 1),
            "Receiving Currency": "USD",
            "Amount Paid": float(i + 1),
            "Payment Currency": "USD",
            "Payment Format": "WIRE",
            "Is Laundering": 1 if i in (1, 5) else 0,
        })
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    annotation_path = tmp_path / "annotations.csv"
    pd.DataFrame([{"source_row_id": 1, "typology": "CYCLE"}]).to_csv(annotation_path, index=False)
    config = {
        "data": {
            "path": str(csv_path),
            "annotations_path": str(annotation_path),
            "allow_all_uncl": False,
            "columns": {
                "timestamp": "Timestamp", "client": "From Bank",
                "amount_received": "Amount Received", "amount_paid": "Amount Paid",
                "receiving_currency": "Receiving Currency", "payment_currency": "Payment Currency",
                "payment_format": "Payment Format", "label": "Is Laundering", "typology": None,
            },
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "train_end_timestamp": "2020-01-07T00:00:00Z",
            "validation_end_timestamp": "2020-01-09T00:00:00Z",
            "feature_hash_dim": 4,
        }
    }
    frame = load_and_label_dataset(config)
    assert "CYCLE" in set(frame["group"])
    assert "UNCL" in set(frame["group"])
    assert "BG" in set(frame["group"])
    x, names = build_feature_matrix(frame, hash_dim=4, seed=123)
    assert x.shape == (10, 17)
    assert not any("account" in n.lower() or "bank" in n.lower() or "typology" in n.lower() for n in names)
