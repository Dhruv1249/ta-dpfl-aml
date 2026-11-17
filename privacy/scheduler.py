from __future__ import annotations

import numpy as np

from ta_dpfl_constants import POSITIVE_GROUPS


def exposure_probabilities(relevance: dict, lam: float, eta: float) -> dict[str, float]:
    if not (0.0 < lam <= 1.0):
        raise ValueError("lambda_exploration must be in (0, 1]")
    if eta <= 0:
        raise ValueError("eta must be > 0")
    if set(relevance) != set(POSITIVE_GROUPS):
        missing = sorted(set(POSITIVE_GROUPS) - set(relevance))
        extra = sorted(set(relevance) - set(POSITIVE_GROUPS))
        raise ValueError(f"Relevance scores must cover exactly all positive groups; missing={missing}, extra={extra}")
    values = {g: float(relevance[g]) for g in POSITIVE_GROUPS}
    if any(not 0.0 <= x <= 1.0 for x in values.values()):
        raise ValueError("Every relevance score must be within [0, 1]")
    denom = sum(values[g] + eta for g in POSITIVE_GROUPS)
    n = len(POSITIVE_GROUPS)
    probs = {g: lam / n + (1.0 - lam) * (values[g] + eta) / denom for g in POSITIVE_GROUPS}
    # Normalize only for floating-point drift.
    total = sum(probs.values())
    return {g: p / total for g, p in probs.items()}


def largest_remainder_counts(probabilities: dict[str, float], total_steps: int) -> dict[str, int]:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    groups = list(probabilities)
    raw = np.array([probabilities[g] * total_steps for g in groups], dtype=float)
    floors = np.floor(raw).astype(int)
    remainder = total_steps - int(floors.sum())
    order = np.argsort(-(raw - floors), kind="stable")
    for idx in order[:remainder]:
        floors[idx] += 1
    counts = {g: int(v) for g, v in zip(groups, floors)}
    if sum(counts.values()) != total_steps:
        raise AssertionError("Largest-remainder allocation does not sum to total_steps")
    return counts


def make_interleaved_schedule(counts: dict[str, int]) -> list[str]:
    """Deterministic public interleaving from public counts; no private data used."""
    schedule = []
    for group, count in counts.items():
        if count < 0:
            raise ValueError("Schedule counts cannot be negative")
        if count:
            # Midpoint quantiles spread each group's entries across [0, 1].
            for j in range(count):
                schedule.append(((j + 0.5) / count, group, j))
    schedule.sort(key=lambda item: (item[0], list(counts).index(item[1]), item[2]))
    return [group for _, group, _ in schedule]


def build_schedule(relevance: dict, lam: float, eta: float, total_steps: int):
    probs = exposure_probabilities(relevance, lam, eta)
    counts = largest_remainder_counts(probs, total_steps)
    zero_groups = [g for g, count in counts.items() if count == 0]
    if zero_groups:
        raise ValueError(
            f"The configured total_steps={total_steps} gives zero focus steps to {zero_groups}. "
            "Increase total local optimizer steps or revise the predeclared public schedule before training."
        )
    return probs, counts, make_interleaved_schedule(counts)
