from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ta_dpfl_constants import ALIASES, BG, POSITIVE_TYPOLOGIES, UNCL


def _norm_name(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ")


def _resolve_column(frame: pd.DataFrame, requested: str | None, aliases: list[str], logical: str) -> str:
    candidates = [requested] if requested else []
    candidates.extend(aliases)
    lookup = {_norm_name(c): c for c in frame.columns}
    for candidate in candidates:
        if candidate is not None and _norm_name(candidate) in lookup:
            return lookup[_norm_name(candidate)]
    raise ValueError(
        f"Could not locate required '{logical}' column. Tried {candidates}; "
        f"available columns: {list(frame.columns)}"
    )


def normalize_typology(value: Any) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    raw = str(value).strip().upper().replace("–", "-").replace("—", "-")
    raw = ALIASES.get(raw, raw)
    if raw in POSITIVE_TYPOLOGIES:
        return raw
    raise ValueError(f"Unknown non-empty typology label {value!r}. Expected one of {POSITIVE_TYPOLOGIES}.")


def load_and_label_dataset(config: dict, derive_split_cutoffs: bool = False) -> pd.DataFrame:
    data_cfg = config["data"]
    csv_path = Path(data_cfg["path"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. See data/README.md for dataset placement and schema."
        )
    raw = pd.read_csv(csv_path, low_memory=False)
    raw["source_row_id"] = np.arange(len(raw), dtype=np.int64)
    col_cfg = data_cfg.get("columns", {})

    selected = {
        "timestamp": _resolve_column(raw, col_cfg.get("timestamp"), ["Timestamp", "timestamp", "date"], "timestamp"),
        "client": _resolve_column(raw, col_cfg.get("client"), ["From Bank", "from_bank", "bank_id"], "client/originating bank"),
        "amount_received": _resolve_column(raw, col_cfg.get("amount_received"), ["Amount Received", "amount_received"], "amount received"),
        "amount_paid": _resolve_column(raw, col_cfg.get("amount_paid"), ["Amount Paid", "amount_paid"], "amount paid"),
        "receiving_currency": _resolve_column(raw, col_cfg.get("receiving_currency"), ["Receiving Currency", "receiving_currency"], "receiving currency"),
        "payment_currency": _resolve_column(raw, col_cfg.get("payment_currency"), ["Payment Currency", "payment_currency"], "payment currency"),
        "payment_format": _resolve_column(raw, col_cfg.get("payment_format"), ["Payment Format", "payment_format"], "payment format"),
        "label": _resolve_column(raw, col_cfg.get("label"), ["Is Laundering", "is_laundering", "label"], "laundering label"),
    }

    frame = pd.DataFrame({key: raw[col] for key, col in selected.items()})
    frame["source_row_id"] = raw["source_row_id"]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if frame["timestamp"].isna().any():
        bad = int(frame["timestamp"].isna().sum())
        raise ValueError(f"{bad} rows have invalid timestamps; resolve them before chronological splitting.")
    if frame["client"].isna().any():
        raise ValueError("Originating client/bank column contains missing values.")
    frame["client"] = frame["client"].astype(str)
    frame["amount_received"] = pd.to_numeric(frame["amount_received"], errors="coerce")
    frame["amount_paid"] = pd.to_numeric(frame["amount_paid"], errors="coerce")
    if frame[["amount_received", "amount_paid"]].isna().any().any():
        raise ValueError("Amount columns contain missing or non-numeric values; resolve before training.")
    if (frame[["amount_received", "amount_paid"]] < 0).any().any():
        raise ValueError("Negative transaction amounts found; log1p feature transform assumes non-negative amounts.")

    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError("Laundering label must contain only binary 0/1 values.")
    frame["label"] = labels.astype(np.int8)

    typology_column = col_cfg.get("typology")
    raw_typology_col = None
    if typology_column:
        raw_typology_col = _resolve_column(raw, typology_column, [], "typology")
    else:
        for candidate in ["typology", "laundering typology", "laundering_type", "pattern_type"]:
            match = next((c for c in raw.columns if _norm_name(c) == _norm_name(candidate)), None)
            if match is not None:
                raw_typology_col = match
                break

    typology_values = None
    if raw_typology_col is not None:
        typology_values = raw[raw_typology_col].map(normalize_typology)

    annotations_path = data_cfg.get("annotations_path")
    if annotations_path and Path(annotations_path).exists():
        annotations = pd.read_csv(annotations_path)
        required = {"source_row_id", "typology"}
        if not required.issubset(annotations.columns):
            raise ValueError(f"Annotation file must contain columns {sorted(required)}.")
        if annotations["source_row_id"].duplicated().any():
            raise ValueError("Annotation source_row_id values must be unique.")
        unknown_annotation_ids = set(annotations["source_row_id"].tolist()) - set(frame["source_row_id"].tolist())
        if unknown_annotation_ids:
            raise ValueError(f"Annotation file contains source_row_id values absent from the raw CSV: {list(unknown_annotation_ids)[:10]}")
        annotation_map = annotations.set_index("source_row_id")["typology"]
        annotation_values = frame["source_row_id"].map(annotation_map).map(normalize_typology)
        if typology_values is None:
            typology_values = annotation_values
        else:
            # Prefer explicit supplied annotations where present; otherwise retain raw labels.
            typology_values = annotation_values.combine_first(typology_values)
    elif typology_values is None:
        if not data_cfg.get("allow_all_uncl", False):
            raise ValueError(
                "No typology column or annotation file found. The paper's main experiment requires "
                "benchmark-supplied typology labels. Provide an annotation CSV; do not infer labels "
                "and describe them as supplied annotations. Set allow_all_uncl=true only for a "
                "pipeline diagnostic, not the typology-aware claim."
            )
        typology_values = pd.Series([None] * len(frame), index=frame.index)

    frame["typology"] = typology_values
    frame["group"] = np.where(frame["label"].eq(0), BG, frame["typology"].fillna(UNCL))
    invalid = sorted(set(frame.loc[frame["label"].eq(1), "group"].unique()) - set(POSITIVE_TYPOLOGIES) - {UNCL})
    if invalid:
        raise ValueError(f"Invalid positive strata found: {invalid}")
    typed_positive_count = int(frame["group"].isin(POSITIVE_TYPOLOGIES).sum())
    if typed_positive_count == 0 and int(frame["label"].sum()) > 0 and not data_cfg.get("allow_all_uncl", False):
        raise ValueError(
            "No positive transaction has a supplied typology annotation. An empty annotation file "
            "or missing typology field is not sufficient for the typology-aware research claim. "
            "Provide genuine benchmark annotations; allow_all_uncl=true is diagnostic only."
        )

    # Ensure no negative record carries a positive typology annotation.
    if (frame.loc[frame["label"].eq(0), "typology"].notna()).any():
        raise ValueError("Negative/background records contain positive typology annotations.")

    frame = frame.sort_values(["timestamp", "source_row_id"], kind="mergesort").reset_index(drop=True)
    n = len(frame)
    train_cut = data_cfg.get("train_end_timestamp")
    validation_cut = data_cfg.get("validation_end_timestamp")
    if train_cut is None or validation_cut is None:
        if not derive_split_cutoffs:
            raise ValueError(
                "Freeze data.train_end_timestamp and data.validation_end_timestamp before training. "
                "Run --audit-only to generate candidate chronological cutoffs, review them, and copy "
                "the values into configs/experiment.yaml. The cutoffs must not be recomputed for "
                "neighbouring-dataset privacy comparisons."
            )
        train_pos = int(math.floor(n * float(data_cfg.get("train_fraction", 0.60))))
        validation_pos = train_pos + int(math.floor(n * float(data_cfg.get("validation_fraction", 0.20))))
        if not (0 < train_pos < validation_pos < n):
            raise ValueError("Invalid chronological split fractions.")
        train_cut = frame.loc[train_pos, "timestamp"].isoformat()
        validation_cut = frame.loc[validation_pos, "timestamp"].isoformat()
    train_cut_ts = pd.to_datetime(train_cut, utc=True, errors="raise")
    validation_cut_ts = pd.to_datetime(validation_cut, utc=True, errors="raise")
    if not train_cut_ts < validation_cut_ts:
        raise ValueError("train_end_timestamp must be earlier than validation_end_timestamp")
    frame["split"] = np.select(
        [frame["timestamp"] < train_cut_ts, frame["timestamp"] < validation_cut_ts],
        ["train", "validation"],
        default="test",
    )
    if (frame["split"] == "train").sum() == 0 or (frame["split"] == "validation").sum() == 0 or (frame["split"] == "test").sum() == 0:
        raise ValueError("Frozen timestamp cutoffs produced an empty split. Review the cutoffs.")
    frame.attrs["split_cutoffs"] = {
        "train_end_timestamp": train_cut_ts.isoformat(),
        "validation_end_timestamp": validation_cut_ts.isoformat(),
        "semantics": "train: timestamp < train_end; validation: train_end <= timestamp < validation_end; test: timestamp >= validation_end",
    }
    return frame


def _hash_index(token: str, dimension: int, seed: int) -> tuple[int, float]:
    digest = hashlib.blake2b(f"{seed}|{token}".encode("utf-8"), digest_size=16).digest()
    integer = int.from_bytes(digest[:8], "little", signed=False)
    sign = 1.0 if (digest[8] & 1) else -1.0
    return integer % dimension, sign


def build_feature_matrix(frame: pd.DataFrame, hash_dim: int, seed: int) -> tuple[np.ndarray, list[str]]:
    if hash_dim < 2:
        raise ValueError("feature_hash_dim must be at least 2")
    n = len(frame)
    received = np.log1p(frame["amount_received"].to_numpy(dtype=np.float64)).astype(np.float32)
    paid = np.log1p(frame["amount_paid"].to_numpy(dtype=np.float64)).astype(np.float32)
    hours = frame["timestamp"].dt.hour.to_numpy(dtype=np.float32) + frame["timestamp"].dt.minute.to_numpy(dtype=np.float32) / 60.0
    hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    same_currency = frame["receiving_currency"].astype(str).eq(frame["payment_currency"].astype(str)).to_numpy(dtype=np.float32)
    numeric = np.column_stack([received, paid, hour_sin, hour_cos, same_currency]).astype(np.float32)

    hashed_blocks = []
    for field in ["receiving_currency", "payment_currency", "payment_format"]:
        block = np.zeros((n, hash_dim), dtype=np.float32)
        values = frame[field].fillna("__MISSING__").astype(str).tolist()
        for row, value in enumerate(values):
            idx, sign = _hash_index(f"{field}={value}", hash_dim, seed)
            block[row, idx] = sign
        hashed_blocks.append(block)

    x = np.concatenate([numeric] + hashed_blocks, axis=1)
    feature_names = ["log_amount_received", "log_amount_paid", "hour_sin", "hour_cos", "same_currency"]
    for field in ["receiving_currency", "payment_currency", "payment_format"]:
        feature_names.extend([f"{field}_hash_{i}" for i in range(hash_dim)])
    return x, feature_names


def write_data_audit(frame: pd.DataFrame, feature_names: list[str], out_path: str | Path) -> dict:
    counts = frame.groupby(["split", "group"], observed=False).size().unstack(fill_value=0)
    report = {
        "rows": int(len(frame)),
        "positive_rows": int(frame["label"].sum()),
        "negative_rows": int((frame["label"] == 0).sum()),
        "unique_clients": int(frame["client"].nunique()),
        "split_rows": {str(k): int(v) for k, v in frame["split"].value_counts().items()},
        "group_counts": {str(k): int(v) for k, v in frame["group"].value_counts().items()},
        "split_group_counts": {str(k): {str(kk): int(vv) for kk, vv in row.items()} for k, row in counts.iterrows()},
        "feature_names": feature_names,
        "timestamp_min": str(frame["timestamp"].min()),
        "timestamp_max": str(frame["timestamp"].max()),
        "source_row_id_definition": "zero-based row position in unchanged source CSV",
        "split_cutoffs": frame.attrs.get("split_cutoffs", {}),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
