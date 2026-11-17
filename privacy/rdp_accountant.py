from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.special import gammaln, logsumexp


@lru_cache(maxsize=4096)
def _log_moment_base(q: float, alpha: int):
    """Cached q-dependent coefficients for one integer RDP order."""
    if q <= 0.0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    i = np.arange(alpha + 1, dtype=float)
    log_binomial = gammaln(alpha + 1) - gammaln(i + 1) - gammaln(alpha - i + 1)
    if q == 1.0:
        base = np.full(alpha + 1, -np.inf, dtype=float)
        base[-1] = 0.0
    else:
        base = log_binomial + i * math.log(q) + (alpha - i) * math.log1p(-q)
    # Opacus / TF-Privacy integer-order SGM expression uses
    # (i^2 - i) / (2 * sigma^2).  Keep the full (i^2 - i) here
    # because the caller multiplies by 1 / (2 * sigma^2).
    exponent = i * i - i
    return base, exponent


def compute_rdp_poisson_gaussian(q: float, sigma: float, orders: Iterable[int]) -> np.ndarray:
    """Integer-order RDP for a Poisson-sampled Gaussian mechanism.

    Uses the exact integer-alpha log-moment expression for the add/remove
    sampled Gaussian mechanism. Orders must be integers >= 2. Composition is
    performed by summing these per-step RDP values.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and > 0")
    orders = np.asarray(list(orders), dtype=int)
    if len(orders) == 0 or np.any(orders < 2):
        raise ValueError("RDP orders must be non-empty integers >= 2")
    if q == 0.0:
        return np.zeros(len(orders), dtype=float)
    out = np.empty(len(orders), dtype=float)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    for j, alpha in enumerate(orders):
        if q == 1.0:
            out[j] = float(alpha) * inv_two_sigma_sq
            continue
        base, exponent = _log_moment_base(float(q), int(alpha))
        out[j] = float(logsumexp(base + exponent * inv_two_sigma_sq) / (alpha - 1.0))
    return out


def rdp_to_epsilon(orders: Iterable[int], rdp: Iterable[float], delta: float) -> tuple[float, int]:
    orders = np.asarray(list(orders), dtype=int)
    rdp = np.asarray(list(rdp), dtype=float)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if len(orders) != len(rdp) or len(orders) == 0:
        raise ValueError("orders and rdp must have equal non-zero length")
    if np.all(rdp == 0):
        return 0.0, int(np.max(orders))
    eps_values = rdp + math.log(1.0 / delta) / (orders - 1.0)
    idx = int(np.argmin(eps_values))
    return float(eps_values[idx]), int(orders[idx])


def account_typology_schedule(
    counts: dict[str, int],
    q_focus: dict[str, float],
    q_background: float,
    sigmas: dict[str, float],
    orders: Iterable[int],
    delta: float,
) -> dict:
    orders = np.asarray(list(orders), dtype=int)
    group_rdp = {}
    group_epsilon = {}
    for group, steps in counts.items():
        rho = compute_rdp_poisson_gaussian(q_focus[group], sigmas[group], orders) * int(steps)
        epsilon, best_order = rdp_to_epsilon(orders, rho, delta)
        group_rdp[group] = rho
        group_epsilon[group] = {"epsilon": epsilon, "best_order": best_order}

    bg_rho = np.zeros(len(orders), dtype=float)
    for group, steps in counts.items():
        bg_rho += compute_rdp_poisson_gaussian(q_background, sigmas[group], orders) * int(steps)
    bg_epsilon, bg_best_order = rdp_to_epsilon(orders, bg_rho, delta)
    return {
        "positive_groups": group_epsilon,
        "background": {"epsilon": bg_epsilon, "best_order": bg_best_order},
        "rdp_positive": {g: v.tolist() for g, v in group_rdp.items()},
        "rdp_background": bg_rho.tolist(),
        "orders": orders.tolist(),
    }


def validate_against_opacus(q_values=(0.001, 0.01, 0.1, 1.0), sigma_values=(0.8, 1.2, 2.0), orders=None, tolerance=1e-5):
    """Compare this accountant with Opacus' independent RDP implementation."""
    try:
        from opacus.accountants.analysis.rdp import compute_rdp as opacus_compute_rdp
    except Exception as exc:
        return {"status": "UNAVAILABLE", "detail": f"Could not import Opacus RDP accountant: {exc}"}

    if orders is None:
        orders = [2, 3, 4, 5, 8, 16, 32, 64]
    max_abs = 0.0
    comparisons = 0
    for q in q_values:
        for sigma in sigma_values:
            ours = compute_rdp_poisson_gaussian(q, sigma, orders)
            theirs = np.asarray(
                opacus_compute_rdp(q=q, noise_multiplier=sigma, steps=1, orders=orders),
                dtype=float,
            )
            diff = float(np.max(np.abs(ours - theirs)))
            max_abs = max(max_abs, diff)
            comparisons += 1
    status = "PASS" if max_abs <= tolerance else "FAIL"
    return {"status": status, "max_absolute_difference": max_abs, "comparisons": comparisons, "tolerance": tolerance}
