"""Per-record RDP accounting for fixed, public partial-client participation.

This module accounts each transaction at its owning virtual client. A client
record receives privacy loss only on rounds in which that client participates.
The participation schedule and typology-focus schedule must be fixed
independently of private training records and must be exactly the schedules
used by the trainer.

No privacy amplification is credited for client participation: inactive rounds
are simply omitted from that record's composition. Poisson amplification is
accounted only for the record-level Bernoulli sampling inside each active local
step.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import brentq, minimize

from privacy.rdp_accountant import compute_rdp_poisson_gaussian, rdp_to_epsilon


def build_client_focus_exposure_counts(
    participation_schedule: list[list[str]],
    focus_schedule: list[str],
    client_ids: Iterable[str],
    local_steps_per_round: int = 1,
) -> dict:
    """Count focus-group steps experienced by every client under exact schedules.

    The focus schedule is global and indexed by (round * local_steps + local
    step), matching run_federated's schedule lookup. Every participating
    client in a round receives the same public focus group(s) for that round.
    """
    ids = sorted(str(x) for x in client_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("client_ids must be nonempty and unique")
    if not participation_schedule:
        raise ValueError("participation_schedule must not be empty")
    if int(local_steps_per_round) < 1:
        raise ValueError("local_steps_per_round must be positive")
    local_steps = int(local_steps_per_round)
    expected_focus_steps = len(participation_schedule) * local_steps
    if len(focus_schedule) != expected_focus_steps:
        raise ValueError(
            f"focus_schedule has {len(focus_schedule)} entries; "
            f"expected {expected_focus_steps}"
        )

    allowed = set(ids)
    counters = {cid: Counter() for cid in ids}
    appearances = Counter()
    cohort_size = None
    normalized_schedule: list[list[str]] = []

    for round_index, cohort in enumerate(participation_schedule):
        current = [str(x) for x in cohort]
        if not current:
            raise ValueError(f"Round {round_index + 1} has an empty cohort")
        if len(current) != len(set(current)):
            raise ValueError(f"Round {round_index + 1} has duplicate clients")
        if not set(current).issubset(allowed):
            raise ValueError(f"Round {round_index + 1} contains unknown clients")
        if cohort_size is None:
            cohort_size = len(current)
        elif len(current) != cohort_size:
            raise ValueError("All participation cohorts must have the same size")
        normalized_schedule.append(current)
        active_groups = focus_schedule[round_index * local_steps:(round_index + 1) * local_steps]
        for client_id in current:
            appearances[client_id] += 1
            counters[client_id].update(str(group) for group in active_groups)

    missing = sorted(allowed - set(appearances))
    if missing:
        raise ValueError(f"Some eligible clients never participate: {missing[:5]}")
    counts_values = set(appearances.values())
    if len(counts_values) != 1:
        raise ValueError("Participation schedule must have equal appearances per client")

    groups = sorted({str(g) for g in focus_schedule})
    if not groups:
        raise ValueError("focus_schedule must contain at least one group")
    matrix = {
        cid: {group: int(counters[cid].get(group, 0)) for group in groups}
        for cid in ids
    }
    global_focus_counts = {
        group: int(sum(matrix[cid][group] for cid in ids) // cohort_size)
        for group in groups
    }
    return {
        "client_focus_counts": matrix,
        "client_participation_counts": {cid: int(appearances[cid]) for cid in ids},
        "num_clients": len(ids),
        "rounds": len(participation_schedule),
        "clients_per_round": int(cohort_size),
        "local_steps_per_round": local_steps,
        "appearances_per_client": int(next(iter(counts_values))),
        "focus_counts_per_global_step_schedule": {
            group: int(sum(1 for g in focus_schedule if str(g) == group))
            for group in groups
        },
        "mean_client_focus_exposures": {
            group: float(np.mean([matrix[cid][group] for cid in ids]))
            for group in groups
        },
        "max_client_focus_exposures": {
            group: int(max(matrix[cid][group] for cid in ids))
            for group in groups
        },
        "min_client_focus_exposures": {
            group: int(min(matrix[cid][group] for cid in ids))
            for group in groups
        },
        "schedule_semantics": (
            "For client c and positive stratum g, only local steps where c is "
            "selected and the public focus schedule equals g contribute q_focus[g] "
            "privacy loss. A background record receives q_background exposure on "
            "every local step in which its client participates, with the noise "
            "multiplier associated with that step's focus group."
        ),
    }


def _eps(rdp: np.ndarray, orders: np.ndarray, delta: float) -> tuple[float, int]:
    return rdp_to_epsilon(orders, rdp, delta)


def account_partial_typology_schedule(
    client_focus_counts: Mapping[str, Mapping[str, int]],
    q_focus: Mapping[str, float],
    q_background: float,
    sigmas: Mapping[str, float],
    orders: Iterable[int],
    delta: float,
    composition_repetitions: int = 1,
) -> dict:
    """Compose exact per-client RDP for TA focus/background mechanisms.

    Positive stratum g at client c: n[c,g] repetitions of the sampled Gaussian
    mechanism (q_focus[g], sigma[g]).

    BG at client c: for each active focus group g, n[c,g] repetitions of
    (q_background, sigma[g]).

    `composition_repetitions` composes independently released model runs/seeds
    on the same training data. It is not client participation amplification.
    """
    if not client_focus_counts:
        raise ValueError("client_focus_counts must not be empty")
    groups = list(q_focus)
    if not groups or set(groups) != set(sigmas):
        raise ValueError("q_focus and sigmas must cover the same nonempty groups")
    if any(g not in row for row in client_focus_counts.values() for g in groups):
        raise ValueError("Every client exposure row must include every focus group")
    if any(int(row[g]) < 0 for row in client_focus_counts.values() for g in groups):
        raise ValueError("Exposure counts cannot be negative")
    if any(not 0.0 < float(q_focus[g]) <= 1.0 for g in groups):
        raise ValueError("Every q_focus must be in (0, 1]")
    if not 0.0 < float(q_background) <= 1.0:
        raise ValueError("q_background must be in (0, 1]")
    if any(float(sigmas[g]) <= 0 or not math.isfinite(float(sigmas[g])) for g in groups):
        raise ValueError("All noise multipliers must be finite and positive")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    repetitions = int(composition_repetitions)
    if repetitions < 1:
        raise ValueError("composition_repetitions must be >= 1")

    orders_array = np.asarray(list(orders), dtype=int)
    if len(orders_array) == 0 or np.any(orders_array < 2):
        raise ValueError("RDP orders must be integers >= 2")
    clients = sorted(str(c) for c in client_focus_counts)

    positive_rdp_by_client: dict[str, dict[str, list[float]]] = {g: {} for g in groups}
    positive_eps_by_client: dict[str, dict[str, float]] = {g: {} for g in groups}
    positive_orders_by_client: dict[str, dict[str, int]] = {g: {} for g in groups}
    group_reports = {}

    step_rdp_focus = {
        g: compute_rdp_poisson_gaussian(float(q_focus[g]), float(sigmas[g]), orders_array)
        for g in groups
    }
    step_rdp_bg = {
        g: compute_rdp_poisson_gaussian(float(q_background), float(sigmas[g]), orders_array)
        for g in groups
    }

    for group in groups:
        eps_values = []
        for client_id in clients:
            count = int(client_focus_counts[client_id][group]) * repetitions
            rho = step_rdp_focus[group] * count
            epsilon, best_order = _eps(rho, orders_array, delta)
            positive_rdp_by_client[group][client_id] = rho.tolist()
            positive_eps_by_client[group][client_id] = float(epsilon)
            positive_orders_by_client[group][client_id] = int(best_order)
            eps_values.append((float(epsilon), client_id, count, int(best_order)))
        worst = max(eps_values, key=lambda x: x[0])
        group_reports[group] = {
            "epsilon_worst_client": worst[0],
            "worst_client_id": worst[1],
            "worst_client_focus_steps_per_run": int(worst[2] // repetitions),
            "worst_client_focus_steps_across_composed_runs": int(worst[2]),
            "best_order_worst_client": worst[3],
            "epsilon_min_client": float(min(x[0] for x in eps_values)),
            "epsilon_median_client": float(np.median([x[0] for x in eps_values])),
            "clients_with_zero_focus_steps": int(sum(x[2] == 0 for x in eps_values)),
        }

    background_rdp_by_client: dict[str, list[float]] = {}
    background_eps_by_client: dict[str, float] = {}
    background_order_by_client: dict[str, int] = {}
    background_steps_by_client: dict[str, int] = {}
    for client_id in clients:
        rho = np.zeros(len(orders_array), dtype=float)
        steps = 0
        for group in groups:
            count_per_run = int(client_focus_counts[client_id][group])
            steps += count_per_run
            rho += step_rdp_bg[group] * (count_per_run * repetitions)
        epsilon, best_order = _eps(rho, orders_array, delta)
        background_rdp_by_client[client_id] = rho.tolist()
        background_eps_by_client[client_id] = float(epsilon)
        background_order_by_client[client_id] = int(best_order)
        background_steps_by_client[client_id] = int(steps)

    worst_bg_client = max(background_eps_by_client, key=background_eps_by_client.get)
    background_report = {
        "epsilon_worst_client": background_eps_by_client[worst_bg_client],
        "worst_client_id": worst_bg_client,
        "worst_client_steps_per_run": int(background_steps_by_client[worst_bg_client]),
        "worst_client_steps_across_composed_runs": int(background_steps_by_client[worst_bg_client] * repetitions),
        "best_order_worst_client": background_order_by_client[worst_bg_client],
        "epsilon_min_client": float(min(background_eps_by_client.values())),
        "epsilon_median_client": float(np.median(list(background_eps_by_client.values()))),
    }
    global_epsilon = max(
        [v["epsilon_worst_client"] for v in group_reports.values()]
        + [background_report["epsilon_worst_client"]]
    )

    return {
        "positive_groups": group_reports,
        "background": background_report,
        "global_epsilon": float(global_epsilon),
        "delta": float(delta),
        "orders": orders_array.tolist(),
        "composition_repetitions": repetitions,
        "num_clients": len(clients),
        "noise_multipliers": {g: float(sigmas[g]) for g in groups},
        "q_focus": {g: float(q_focus[g]) for g in groups},
        "q_background": float(q_background),
        "positive_rdp_by_client": positive_rdp_by_client,
        "background_rdp_by_client": background_rdp_by_client,
        "positive_epsilon_by_client": positive_eps_by_client,
        "background_epsilon_by_client": background_eps_by_client,
    }


def calibrate_partial_typology_noise(
    client_focus_counts: Mapping[str, Mapping[str, int]],
    q_focus: Mapping[str, float],
    q_background: float,
    targets: Mapping[str, float],
    background_target: float,
    delta: float,
    orders: Iterable[int],
    composition_repetitions: int = 1,
    require_opacus_validation: bool = True,
) -> dict:
    """Find feasible sigmas for exact schedule-specific worst-client targets.

    The optimizer is numerical; feasibility is checked explicitly. No global
    optimality claim is made for the SLSQP objective.
    """
    from privacy.rdp_accountant import validate_against_opacus

    groups = list(q_focus)
    if set(groups) != set(targets):
        raise ValueError("targets and q_focus must cover the same positive groups")
    if not client_focus_counts:
        raise ValueError("client_focus_counts must not be empty")
    if any(float(targets[g]) <= 0 for g in groups) or float(background_target) <= 0:
        raise ValueError("All privacy targets must be positive")
    if require_opacus_validation:
        validation = validate_against_opacus(orders=orders)
        if validation["status"] != "PASS":
            raise RuntimeError(
                "Independent RDP validation did not pass; refusing partial-DP calibration: "
                f"{validation}"
            )
    else:
        validation = {"status": "NOT_REQUIRED"}

    repetitions = int(composition_repetitions)
    if repetitions < 1:
        raise ValueError("composition_repetitions must be >= 1")
    orders_array = np.asarray(list(orders), dtype=int)
    max_counts = {
        g: max(int(client_focus_counts[cid][g]) for cid in client_focus_counts) * repetitions
        for g in groups
    }
    if any(max_counts[g] <= 0 for g in groups):
        absent = [g for g in groups if max_counts[g] <= 0]
        raise ValueError(
            f"No client receives a focus step for {absent}; this schedule cannot train/account those strata."
        )

    def epsilon_for(q: float, sigma: float, steps: int) -> float:
        rho = compute_rdp_poisson_gaussian(q, sigma, orders_array) * int(steps)
        return rdp_to_epsilon(orders_array, rho, delta)[0]

    initial = {}
    for group in groups:
        target = float(targets[group])
        q = float(q_focus[group])
        steps = max_counts[group]
        def constraint(log_sigma):
            return epsilon_for(q, math.exp(log_sigma), steps) - target
        low, high = math.log(0.01), math.log(1_000.0)
        if constraint(high) > 0:
            raise RuntimeError(f"Positive privacy target for {group} is infeasible")
        initial[group] = math.exp(brentq(constraint, low, high, maxiter=200))

    clients = sorted(str(c) for c in client_focus_counts)
    exposure_matrix = np.asarray(
        [[int(client_focus_counts[cid][group]) * repetitions for group in groups] for cid in clients],
        dtype=np.float64,
    )
    epsilon_conversion = math.log(1.0 / float(delta)) / (orders_array - 1.0)

    def worst_background_epsilon(sigmas: Mapping[str, float]) -> float:
        # Vectorized across clients: RDP(client, alpha) = sum_g n[c,g] *
        # RDP(q_bg, sigma_g, alpha). This avoids a Python loop over all
        # clients for every SLSQP constraint evaluation.
        group_rdp = np.stack([
            compute_rdp_poisson_gaussian(
                float(q_background), float(sigmas[group]), orders_array
            )
            for group in groups
        ], axis=0)
        client_rdp = exposure_matrix @ group_rdp
        client_eps = np.min(client_rdp + epsilon_conversion[None, :], axis=1)
        return float(np.max(client_eps))

    if worst_background_epsilon(initial) > float(background_target):
        def bg_constraint(log_scale):
            scale = math.exp(log_scale)
            scaled = {g: initial[g] * scale for g in groups}
            return worst_background_epsilon(scaled) - float(background_target)
        low, high = 0.0, math.log(2.0)
        while bg_constraint(high) > 0 and high < math.log(1_000.0):
            high += math.log(2.0)
        if bg_constraint(high) > 0:
            raise RuntimeError("Background privacy target is infeasible within sigma search bounds")
        scale = math.exp(brentq(bg_constraint, low, high, maxiter=200))
        initial = {g: initial[g] * scale for g in groups}

    x0 = np.log([initial[g] for g in groups])

    def unpack(log_sigmas):
        return {g: float(math.exp(log_sigmas[i])) for i, g in enumerate(groups)}

    def objective(log_sigmas):
        sigmas = np.exp(log_sigmas)
        return float(sum(max_counts[g] * sigmas[i] ** 2 for i, g in enumerate(groups)))

    def constraints(log_sigmas):
        sigmas = unpack(log_sigmas)
        values = []
        for group in groups:
            values.append(
                float(targets[group])
                - epsilon_for(float(q_focus[group]), sigmas[group], max_counts[group])
            )
        values.append(float(background_target) - worst_background_epsilon(sigmas))
        return np.asarray(values, dtype=float)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(math.log(0.01), math.log(1_000.0))] * len(groups),
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"maxiter": 160, "ftol": 1e-9, "disp": False},
    )
    chosen = result.x if np.all(constraints(result.x) >= -1e-6) else x0
    sigmas = unpack(chosen)
    report = account_partial_typology_schedule(
        client_focus_counts, q_focus, q_background, sigmas,
        orders_array, delta, composition_repetitions=repetitions,
    )
    positive_satisfied = all(
        report["positive_groups"][g]["epsilon_worst_client"] <= float(targets[g]) + 1e-5
        for g in groups
    )
    background_satisfied = report["background"]["epsilon_worst_client"] <= float(background_target) + 1e-5
    if not positive_satisfied or not background_satisfied:
        raise RuntimeError("Partial typology noise calibration failed a per-record privacy target")

    report.update({
        "targets": {"positive_groups": {g: float(targets[g]) for g in groups}, "background": float(background_target)},
        "max_focus_steps_per_client_across_composed_runs": max_counts,
        "objective_total_variance_proxy": float(objective(chosen)),
        "optimizer_success": bool(result.success and np.all(constraints(result.x) >= -1e-6)),
        "optimizer_message": str(result.message),
        "calibration_optimality_certified": False,
        "calibration_note": "Feasible numerical SLSQP calibration; global optimality is not certified.",
        "independent_rdp_validation": validation,
        "target_satisfied": True,
        "accounting_protocol": "per-record, per-owning-virtual-client; public partial participation; no client-subsampling amplification credited",
    })
    return report


def account_partial_uniform(
    q: float,
    sigma: float,
    client_participation_counts: Mapping[str, int],
    local_steps_per_round: int,
    orders: Iterable[int],
    delta: float,
    composition_repetitions: int = 1,
) -> dict:
    """Account uniform Poisson DP-SGD for records in their owning clients."""
    if not client_participation_counts:
        raise ValueError("client_participation_counts must not be empty")
    if not 0.0 < float(q) <= 1.0 or not math.isfinite(float(sigma)) or float(sigma) <= 0:
        raise ValueError("q must be in (0,1] and sigma must be finite and positive")
    if int(local_steps_per_round) < 1:
        raise ValueError("local_steps_per_round must be >= 1")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0,1)")
    repetitions = int(composition_repetitions)
    if repetitions < 1:
        raise ValueError("composition_repetitions must be >= 1")
    counts = {
        str(cid): int(n) * int(local_steps_per_round) * repetitions
        for cid, n in client_participation_counts.items()
    }
    if any(v < 0 for v in counts.values()):
        raise ValueError("Participation counts cannot be negative")
    orders_array = np.asarray(list(orders), dtype=int)
    step_rdp = compute_rdp_poisson_gaussian(float(q), float(sigma), orders_array)
    client_rdp = {cid: (step_rdp * count).tolist() for cid, count in counts.items()}
    eps = {cid: float(rdp_to_epsilon(orders_array, step_rdp * count, delta)[0]) for cid, count in counts.items()}
    worst_client = max(eps, key=eps.get)
    return {
        "global_epsilon": eps[worst_client],
        "worst_client_id": worst_client,
        "worst_client_steps_across_composed_runs": counts[worst_client],
        "epsilon_min_client": float(min(eps.values())),
        "epsilon_median_client": float(np.median(list(eps.values()))),
        "client_epsilon": eps,
        "client_rdp": client_rdp,
        "orders": orders_array.tolist(),
        "delta": float(delta),
        "q": float(q),
        "sigma": float(sigma),
        "composition_repetitions": repetitions,
    }


def compose_partial_record_rdp(
    reports: Iterable[dict],
    orders: Iterable[int],
    delta: float,
) -> dict:
    """Compose record-level RDP across multiple released partial-DP methods.

    Each report must include positive_rdp_by_client and background_rdp_by_client
    (as returned by account_partial_typology_schedule), or a compatible
    `record_rdp` structure. This is useful when multiple DP method checkpoints
    trained on the same records are released together.
    """
    reports = list(reports)
    if not reports:
        raise ValueError("At least one report is required")
    orders_array = np.asarray(list(orders), dtype=int)
    # Generic report contract: record_rdp_positive[group][client] and
    # record_rdp_background[client], arrays already composed over repetitions.
    all_groups = set()
    all_clients = None
    for report in reports:
        if "positive_rdp_by_client" not in report or "background_rdp_by_client" not in report:
            raise ValueError("Report is missing per-client RDP arrays")
        all_groups.update(report["positive_rdp_by_client"])
        report_clients = set(report["background_rdp_by_client"])
        if all_clients is None:
            all_clients = report_clients
        elif report_clients != all_clients:
            raise ValueError("All composed reports must describe exactly the same client roster")
        for group, by_client in report["positive_rdp_by_client"].items():
            if set(by_client) != report_clients:
                raise ValueError(f"Positive-stratum RDP report for {group} has inconsistent client IDs")
        for values in report["background_rdp_by_client"].values():
            if len(values) != len(orders_array):
                raise ValueError("Background RDP vector length does not match orders")
        for by_client in report["positive_rdp_by_client"].values():
            for values in by_client.values():
                if len(values) != len(orders_array):
                    raise ValueError("Positive-stratum RDP vector length does not match orders")
    if not all_clients:
        raise ValueError("Reports must describe the same nonempty client roster")

    worst = {"epsilon": -1.0}
    for client_id in sorted(all_clients):
        bg_rho = np.zeros(len(orders_array), dtype=float)
        for report in reports:
            bg_rho += np.asarray(report["background_rdp_by_client"][client_id], dtype=float)
        eps, best = rdp_to_epsilon(orders_array, bg_rho, delta)
        if eps > worst["epsilon"]:
            worst = {"epsilon": float(eps), "client_id": client_id, "stratum": "BG", "best_order": int(best)}
        for group in all_groups:
            rho = np.zeros(len(orders_array), dtype=float)
            for report in reports:
                if group in report["positive_rdp_by_client"]:
                    rho += np.asarray(report["positive_rdp_by_client"][group][client_id], dtype=float)
            eps, best = rdp_to_epsilon(orders_array, rho, delta)
            if eps > worst["epsilon"]:
                worst = {"epsilon": float(eps), "client_id": client_id, "stratum": group, "best_order": int(best)}
    return {
        "global_epsilon_joint_release": worst["epsilon"],
        "worst_record_location": {k: v for k, v in worst.items() if k != "epsilon"},
        "delta": float(delta),
        "orders": orders_array.tolist(),
        "number_of_composed_methods": len(reports),
        "warning": "This is the bound for releasing all listed method outputs together; it is not the per-method comparison bound.",
    }


def account_partial_stratified_uniform(
    q: float,
    sigma_by_group: Mapping[str, float],
    client_participation_counts: Mapping[str, int],
    local_steps_per_round: int,
    orders: Iterable[int],
    delta: float,
    composition_repetitions: int = 1,
) -> dict:
    """Account a separate sampled-Gaussian mechanism for each fixed stratum.

    The implementation must Poisson-sample each stratum independently, clip
    and sum within that stratum, add one independent Gaussian vector for that
    stratum on every local step (including empty samples), and only then sum
    the privatized stratum vectors. A record affects only its fixed stratum's
    Gaussian mechanism; summing the stratum outputs is post-processing.
    """
    if not sigma_by_group or "BG" not in sigma_by_group:
        raise ValueError("sigma_by_group must include BG and all positive strata")
    repetitions = int(composition_repetitions)
    if repetitions < 1:
        raise ValueError("composition_repetitions must be >= 1")
    orders_array = np.asarray(list(orders), dtype=int)
    step_counts = {
        str(cid): int(n) * int(local_steps_per_round) * repetitions
        for cid, n in client_participation_counts.items()
    }
    if not step_counts or any(n < 0 for n in step_counts.values()):
        raise ValueError("Invalid client participation counts")
    if not 0.0 < float(q) <= 1.0:
        raise ValueError("q must be in (0,1]")
    if int(local_steps_per_round) < 1:
        raise ValueError("local_steps_per_round must be >= 1")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0,1)")
    reports = {}
    positive_rdp_by_client: dict[str, dict[str, list[float]]] = {}
    positive_epsilon_by_client: dict[str, dict[str, float]] = {}
    background_rdp_by_client = {}
    background_epsilon_by_client = {}
    for group, sigma in sigma_by_group.items():
        if float(sigma) <= 0 or not math.isfinite(float(sigma)):
            raise ValueError(f"Invalid sigma for stratum {group}")
        step_rdp = compute_rdp_poisson_gaussian(float(q), float(sigma), orders_array)
        eps_by_client = {
            cid: float(rdp_to_epsilon(orders_array, step_rdp * n, delta)[0])
            for cid, n in step_counts.items()
        }
        rdp_by_client = {cid: (step_rdp * n).tolist() for cid, n in step_counts.items()}
        worst = max(eps_by_client, key=eps_by_client.get)
        reports[group] = {
            "epsilon_worst_client": eps_by_client[worst],
            "worst_client_id": worst,
            "epsilon_min_client": float(min(eps_by_client.values())),
            "epsilon_median_client": float(np.median(list(eps_by_client.values()))),
            "sigma": float(sigma),
            "q": float(q),
            "steps_per_client_across_composed_runs": step_counts[worst],
        }
        if group == "BG":
            background_rdp_by_client = rdp_by_client
            background_epsilon_by_client = eps_by_client
        else:
            positive_rdp_by_client[group] = rdp_by_client
            positive_epsilon_by_client[group] = eps_by_client
    global_epsilon = max(v["epsilon_worst_client"] for v in reports.values())
    return {
        "strata": reports,
        "positive_rdp_by_client": positive_rdp_by_client,
        "background_rdp_by_client": background_rdp_by_client,
        "positive_epsilon_by_client": positive_epsilon_by_client,
        "background_epsilon_by_client": background_epsilon_by_client,
        "global_epsilon": float(global_epsilon),
        "orders": orders_array.tolist(),
        "delta": float(delta),
        "q": float(q),
        "composition_repetitions": repetitions,
        "accounting_protocol": "per-stratum Poisson sampled Gaussian; no client-subsampling amplification credited",
    }
