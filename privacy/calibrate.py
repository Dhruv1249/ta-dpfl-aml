from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, minimize

from privacy.rdp_accountant import (
    account_typology_schedule,
    compute_rdp_poisson_gaussian,
    rdp_to_epsilon,
    validate_against_opacus,
)


def _epsilon(q: float, sigma: float, steps: int, orders, delta: float) -> float:
    rho = compute_rdp_poisson_gaussian(q, sigma, orders) * int(steps)
    return rdp_to_epsilon(orders, rho, delta)[0]


def calibrate_uniform_noise(q: float, steps: int, epsilon_target: float, delta: float, orders):
    if not 0 < q <= 1:
        raise ValueError("q_uniform must be in (0, 1]")
    if steps < 1:
        raise ValueError("steps must be positive")
    if epsilon_target <= 0:
        raise ValueError("epsilon target must be positive")
    def f(log_sigma):
        return _epsilon(q, math.exp(log_sigma), steps, orders, delta) - epsilon_target
    low, high = math.log(0.01), math.log(1_000.0)
    if f(high) > 0:
        raise RuntimeError("Could not meet uniform privacy target even at sigma=1000; configuration infeasible.")
    root = brentq(f, low, high, maxiter=200)
    sigma = math.exp(root)
    return sigma, _epsilon(q, sigma, steps, orders, delta)


def calibrate_typology_noise(
    counts: dict[str, int],
    q_focus: dict[str, float],
    q_background: float,
    targets: dict[str, float],
    background_target: float,
    delta: float,
    orders,
    require_opacus_validation: bool = True,
):
    if require_opacus_validation:
        validation = validate_against_opacus(orders=orders)
        if validation["status"] != "PASS":
            raise RuntimeError(
                "Independent RDP accountant validation did not pass. "
                f"Result: {validation}. No formal privacy guarantee will be reported."
            )
    groups = list(counts)
    if set(groups) != set(q_focus) or set(groups) != set(targets):
        raise ValueError("counts, q_focus and targets must cover identical positive groups")
    if any(counts[g] < 1 for g in groups):
        raise ValueError("Every positive stratum must receive at least one focus step")
    if any(not 0 < q_focus[g] <= 1 for g in groups) or not 0 < q_background <= 1:
        raise ValueError("Sampling probabilities must be in (0, 1]")
    if any(targets[g] <= 0 for g in groups) or background_target <= 0:
        raise ValueError("All privacy targets must be positive")

    orders = np.asarray(list(orders), dtype=int)
    step_vector = np.asarray([counts[g] for g in groups], dtype=float)

    def eps_for(q, sigma, steps):
        rho = compute_rdp_poisson_gaussian(q, sigma, orders) * int(steps)
        return rdp_to_epsilon(orders, rho, delta)[0]

    # Build a feasible starting point by independently meeting positive-group
    # targets, then scaling all sigmas until the cumulative BG target is met.
    initial_sigmas = {}
    for group in groups:
        def positive_constraint(log_sigma):
            return eps_for(q_focus[group], math.exp(log_sigma), counts[group]) - targets[group]
        low, high = math.log(0.01), math.log(1_000.0)
        if positive_constraint(high) > 0:
            raise RuntimeError(f"Positive-stratum privacy target infeasible for {group}")
        if positive_constraint(low) <= 0:
            initial_sigmas[group] = math.exp(low)
        else:
            initial_sigmas[group] = math.exp(brentq(positive_constraint, low, high, maxiter=100))

    def background_epsilon(scale):
        rho = np.zeros(len(orders), dtype=float)
        for group in groups:
            rho += compute_rdp_poisson_gaussian(q_background, initial_sigmas[group] * scale, orders) * counts[group]
        return rdp_to_epsilon(orders, rho, delta)[0]

    if background_epsilon(1.0) > background_target:
        lo, hi = 1.0, 2.0
        while background_epsilon(hi) > background_target and hi < 1_000.0:
            hi *= 2.0
        if background_epsilon(hi) > background_target:
            raise RuntimeError("Background privacy target infeasible within sigma search bound")
        scale = brentq(lambda z: background_epsilon(z) - background_target, lo, hi, maxiter=100)
        initial_sigmas = {g: s * scale for g, s in initial_sigmas.items()}

    x0 = np.log([initial_sigmas[g] for g in groups])

    def unpack(z):
        return {g: float(np.exp(z[j])) for j, g in enumerate(groups)}

    def compute_report(sigmas):
        group_epsilon = {}
        for group in groups:
            value = eps_for(q_focus[group], sigmas[group], counts[group])
            group_epsilon[group] = {"epsilon": value}
        bg_rho = np.zeros(len(orders), dtype=float)
        for group in groups:
            bg_rho += compute_rdp_poisson_gaussian(q_background, sigmas[group], orders) * counts[group]
        bg_eps, bg_order = rdp_to_epsilon(orders, bg_rho, delta)
        for group in groups:
            _, best_order = rdp_to_epsilon(
                orders,
                compute_rdp_poisson_gaussian(q_focus[group], sigmas[group], orders) * counts[group],
                delta,
            )
            group_epsilon[group]["best_order"] = best_order
        return group_epsilon, {"epsilon": bg_eps, "best_order": bg_order}, bg_rho

    def objective(z):
        sigmas = np.exp(z)
        return float(np.dot(step_vector, sigmas * sigmas))

    def constraints(z):
        sigmas = unpack(z)
        group_eps, bg_eps, _ = compute_report(sigmas)
        values = [targets[g] - group_eps[g]["epsilon"] for g in groups]
        values.append(background_target - bg_eps["epsilon"])
        return np.asarray(values, dtype=float)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(math.log(0.01), math.log(1_000.0))] * len(groups),
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"maxiter": 120, "ftol": 1e-8, "disp": False},
    )
    candidate_z = result.x if np.all(constraints(result.x) >= -1e-6) else x0
    sigmas = unpack(candidate_z)
    group_eps, bg_eps, bg_rho = compute_report(sigmas)
    positive_rdp = {
        g: (compute_rdp_poisson_gaussian(q_focus[g], sigmas[g], orders) * counts[g]).tolist()
        for g in groups
    }
    global_epsilon = max([v["epsilon"] for v in group_eps.values()] + [bg_eps["epsilon"]])
    satisfied = all(group_eps[g]["epsilon"] <= targets[g] + 1e-5 for g in groups) and bg_eps["epsilon"] <= background_target + 1e-5
    if not satisfied:
        raise RuntimeError("Calibrated noise schedule failed a privacy constraint")

    return {
        "positive_groups": group_eps,
        "background": bg_eps,
        "rdp_positive": positive_rdp,
        "rdp_background": bg_rho.tolist(),
        "orders": orders.tolist(),
        "noise_multipliers": sigmas,
        "targets": {"positive_groups": targets, "background": background_target},
        "objective_total_variance_proxy": objective(candidate_z),
        "optimizer_message": str(result.message),
        "optimizer_success": bool(result.success and np.all(constraints(result.x) >= -1e-6)),
        "calibration_optimality_certified": False,
        "calibration_note": "Feasible numerical calibration; SLSQP result is not a proof of global optimality.",
        "opacus_validation": validate_against_opacus(orders=orders),
        "global_epsilon": global_epsilon,
        "target_satisfied": bool(satisfied),
    }
