from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ta_dpfl_constants import BG, POSITIVE_GROUPS


def select_threshold_at_fpr(y_true, scores, max_fpr: float):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")
    if len(np.unique(y_true)) < 2:
        return {"threshold": float("inf"), "validation_fpr": float("nan"), "validation_recall": 0.0, "status": "INSUFFICIENT_CLASSES"}
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = np.flatnonzero(fpr <= max_fpr)
    if len(valid) == 0:
        return {"threshold": float("inf"), "validation_fpr": 0.0, "validation_recall": 0.0, "status": "NO_VALID_THRESHOLD"}
    best_recall = np.max(tpr[valid])
    best = valid[np.isclose(tpr[valid], best_recall)]
    # With tied recall, choose the more conservative (higher) threshold.
    idx = best[np.argmax(thresholds[best])]
    return {
        "threshold": float(thresholds[idx]),
        "validation_fpr": float(fpr[idx]),
        "validation_recall": float(tpr[idx]),
        "status": "OK" if best_recall > 0 else "ZERO_RECALL_AT_CONSTRAINT",
    }


def _binary_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    result = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int((y_true == 0).sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
        "AP": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
        "ROC_AUC": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(((predictions == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())),
    }
    return result


def evaluate_overall(y_true, scores, threshold):
    return _binary_metrics(y_true, scores, threshold)


def evaluate_by_typology(groups, y_true, scores, threshold):
    groups = np.asarray(groups)
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rows = []
    bg_mask = groups == BG
    for group in POSITIVE_GROUPS:
        mask = (groups == group) | bg_mask
        y_local = (groups[mask] == group).astype(int)
        row = {"typology": group, **_binary_metrics(y_local, scores[mask], threshold)}
        rows.append(row)
    return rows
