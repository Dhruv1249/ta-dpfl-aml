from __future__ import annotations

import copy
import hashlib
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from privacy.dp_sgd import private_typology_step, private_uniform_step, private_individualized_step


def build_balanced_participation_schedule(
    client_ids,
    rounds: int,
    clients_per_round: int,
    seed: int,
) -> list[list[str]]:
    """Create a deterministic, label-independent balanced client schedule.

    Each cycle shuffles the full client roster and partitions it into disjoint
    cohorts. The caller must choose rounds so every client appears exactly the
    same number of times. Client selection depends only on public client IDs,
    the requested participation budget, and the fixed seed.
    """
    ids = sorted(str(cid) for cid in client_ids)
    n_clients = len(ids)
    rounds = int(rounds)
    clients_per_round = int(clients_per_round)
    if n_clients < 1:
        raise ValueError("client_ids must not be empty")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not 1 <= clients_per_round <= n_clients:
        raise ValueError("clients_per_round must be between 1 and the number of clients")
    total_slots = rounds * clients_per_round
    if total_slots % n_clients != 0:
        raise ValueError(
            "Balanced schedule requires rounds * clients_per_round to be "
            "divisible by the number of clients; choose a compatible round count"
        )
    # For the simple permutation-block construction, each complete cycle
    # visits every client exactly once and creates n_clients / clients_per_round rounds.
    if n_clients % clients_per_round != 0:
        raise ValueError(
            "This balanced scheduler requires the number of clients to be "
            "divisible by clients_per_round"
        )
    rounds_per_cycle = n_clients // clients_per_round
    if rounds % rounds_per_cycle != 0:
        raise ValueError(
            "rounds must contain an integer number of complete balanced cycles"
        )

    rng = np.random.default_rng(int(seed))
    schedule: list[list[str]] = []
    for _ in range(rounds // rounds_per_cycle):
        permutation = rng.permutation(np.asarray(ids, dtype=object)).tolist()
        for start in range(0, n_clients, clients_per_round):
            cohort = [str(x) for x in permutation[start:start + clients_per_round]]
            if len(cohort) != clients_per_round:
                raise RuntimeError("Internal error: incomplete participation cohort")
            schedule.append(cohort)

    if len(schedule) != rounds:
        raise RuntimeError(f"Schedule has {len(schedule)} rounds; expected {rounds}")
    counts = Counter(cid for cohort in schedule for cid in cohort)
    if set(counts) != set(ids) or len(set(counts.values())) != 1:
        raise RuntimeError("Generated participation schedule is not exactly balanced")
    return schedule


def _validate_participation_schedule(
    participation_schedule, client_ids, rounds, mode, allow_private_partial_participation=False
):
    if participation_schedule is None:
        return None
    private_modes = {"uniform_dp", "ta_dpfl", "individualized_dp"}
    if mode in private_modes and not allow_private_partial_participation:
        raise ValueError(
            "Partial participation for DP modes requires the runner to complete "
            "schedule-specific per-client record-level accounting first."
        )
    if mode not in {"fedavg", "fedadam"} | private_modes:
        raise ValueError(f"Unsupported partial-participation mode: {mode}")
    if len(participation_schedule) != int(rounds):
        raise ValueError("participation_schedule length must equal rounds")
    allowed = set(client_ids)
    normalized = []
    expected_size = None
    counts = Counter()
    for round_number, cohort in enumerate(participation_schedule, start=1):
        cohort = [str(cid) for cid in cohort]
        if not cohort:
            raise ValueError(f"Round {round_number} has no participating clients")
        if len(set(cohort)) != len(cohort):
            raise ValueError(f"Round {round_number} contains duplicate client IDs")
        if not set(cohort).issubset(allowed):
            unknown = sorted(set(cohort) - allowed)
            raise ValueError(f"Round {round_number} contains unknown clients: {unknown[:5]}")
        if expected_size is None:
            expected_size = len(cohort)
        elif len(cohort) != expected_size:
            raise ValueError("All rounds must have the same number of participating clients")
        counts.update(cohort)
        normalized.append(cohort)
    if set(counts) != allowed or len(set(counts.values())) != 1:
        raise ValueError("Participation schedule must include every client equally often")
    return normalized

def _all_client_indices(group_indices):
    arrays = [v for v in group_indices.values() if len(v)]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)


def _sample_indices_sha256(indices: np.ndarray) -> str:
    """Fingerprint a sampled batch for paired non-private reproducibility audits.

    The raw row indices are never written to the experiment logs. This digest is
    diagnostic metadata only; it is not a privacy guarantee and must not be
    exported as a DP release.
    """
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _state_dict_sha256(state_dict) -> str:
    """Fingerprint model initialization for paired optimizer comparisons."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _nonprivate_step(
    model, x_array, y_array, candidates, batch_size, learning_rate,
    device, rng, optimizer=None, max_grad_norm=5.0,
):
    if len(candidates) == 0:
        return {"sampled_count": 0, "sampled_positive_count": 0, "loss": None}
    size = min(int(batch_size), len(candidates))
    batch = rng.choice(candidates, size=size, replace=False)
    x = torch.as_tensor(x_array[batch], dtype=torch.float32, device=device)
    y = torch.as_tensor(y_array[batch], dtype=torch.float32, device=device)

    if optimizer is None:
        model.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad(set_to_none=True)

    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()

    if optimizer is None:
        # Original manual-SGD implementation retained for the baseline.
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-learning_rate)
    else:
        # AdamW local optimization; gradient clipping matches the central MLP v2.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
        optimizer.step()

    return {
        "sampled_count": int(size),
        "sampled_positive_count": int(np.asarray(y_array[batch]).sum()),
        "loss": float(loss.detach().cpu()),
        "sampled_indices_sha256": _sample_indices_sha256(batch),
    }


def _state_delta_l2(local_state, global_state):
    total = 0.0
    for key in global_state:
        diff = local_state[key].detach().float() - global_state[key].detach().float()
        total += float(torch.sum(diff * diff).item())
    return float(np.sqrt(total))


def _fedadam_server_update(
    global_state,
    aggregated_delta,
    first_moment,
    second_moment,
    step,
    learning_rate,
    beta1=0.9,
    beta2=0.99,
    tau=1e-8,
):
    """Apply one FedAdam server update to a client-averaged model delta.

    Client deltas are defined as local_parameters - global_parameters, i.e.
    they already point in the direction in which the global parameters should
    move. Moments are therefore accumulated over deltas and ADDED to the server
    parameters. This is equivalent to applying Adam to the negative pseudo-
    gradient (global - client_average) with a conventional subtractive update.
    """
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("FedAdam beta1 and beta2 must be in [0, 1)")
    if learning_rate <= 0 or tau <= 0 or step < 1:
        raise ValueError("FedAdam learning_rate, tau and step must be positive")

    updated = {}
    applied_update_sq = 0.0
    for key, parameter in global_state.items():
        delta = aggregated_delta[key]
        if not torch.is_floating_point(parameter):
            # This model currently has no integer buffers; preserve any future
            # non-floating state without applying adaptive arithmetic to it.
            updated[key] = parameter + delta
            continue

        m = first_moment[key]
        v = second_moment[key]
        m.mul_(beta1).add_(delta, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(delta, delta, value=1.0 - beta2)

        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        server_update = float(learning_rate) * m_hat / (torch.sqrt(v_hat) + float(tau))
        updated[key] = parameter + server_update
        applied_update_sq += float(torch.sum(server_update.float() ** 2).item())

    return updated, float(np.sqrt(applied_update_sq))


def run_federated(
    model_factory,
    x_array: np.ndarray,
    y_array: np.ndarray,
    client_partitions: dict,
    mode: str,
    rounds: int,
    local_steps_per_round: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    batch_size_nonprivate: int = 256,
    schedule: list[str] | None = None,
    q_focus: dict[str, float] | None = None,
    q_background: float | None = None,
    q_uniform: float | None = None,
    sigmas: dict[str, float] | None = None,
    sigma_uniform: float | None = None,
    clip_norm: float | None = None,
    normalization: float | None = None,
    secure_randomness: bool = True,
    groups_array: np.ndarray | None = None,
    sigma_by_group: dict[str, float] | None = None,
    aggregation_mode: str = "uniform",
    nonprivate_optimizer: str = "sgd",
    nonprivate_weight_decay: float = 1e-4,
    nonprivate_max_grad_norm: float = 5.0,
    participation_schedule: list[list[str]] | None = None,
    server_learning_rate: float = 0.001,
    server_beta1: float = 0.9,
    server_beta2: float = 0.99,
    server_tau: float = 1e-8,
    server_optimizer: str | None = None,
    partial_dp_accounting_enabled: bool = False,
):
    """Simulate FedAvg with all-client or explicitly scheduled participation.

    FedAvg applies the client-averaged delta directly. FedAdam applies persistent
    server-side first/second moments to the same averaged delta. For DP modes,
    partial participation is permitted only when the caller explicitly confirms
    that exact per-client/per-record exposure accounting has been completed.
    A FedAdam server update on DP-protected client deltas is post-processing of
    those protected messages; the local DP mechanisms remain the privacy source.
    """
    if mode not in {"fedavg", "fedadam", "uniform_dp", "ta_dpfl", "individualized_dp"}:
        raise ValueError(f"Unsupported federated mode: {mode}")
    if rounds < 1 or local_steps_per_round < 1:
        raise ValueError("rounds and local_steps_per_round must both be positive")
    if mode == "ta_dpfl" and (schedule is None or len(schedule) != rounds * local_steps_per_round):
        raise ValueError("TA-DPFL schedule length must equal rounds * local_steps_per_round")
    if server_optimizer is None:
        server_optimizer = "fedadam" if mode == "fedadam" else "fedavg"
    server_optimizer = str(server_optimizer).lower()
    if server_optimizer not in {"fedavg", "fedadam"}:
        raise ValueError("server_optimizer must be 'fedavg' or 'fedadam'")
    if mode == "fedadam" and server_optimizer != "fedadam":
        raise ValueError("mode='fedadam' requires server_optimizer='fedadam'")
    if mode in {"fedavg", "fedadam"} and mode == "fedavg" and server_optimizer != "fedavg":
        raise ValueError("Use mode='fedadam' for the non-private FedAdam diagnostic")

    # Seed model initialization explicitly so paired aggregation ablations
    # begin from identical weights. NumPy sampling is seeded separately below.
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if aggregation_mode not in {"uniform", "sample_size"}:
        raise ValueError("aggregation_mode must be 'uniform' or 'sample_size'")
    if mode != "fedavg" and aggregation_mode != "uniform":
        raise ValueError("Non-uniform aggregation is permitted only for the non-private FedAvg diagnostic")
    if nonprivate_optimizer not in {"sgd", "adamw"}:
        raise ValueError("nonprivate_optimizer must be 'sgd' or 'adamw'")
    if mode == "fedadam" and nonprivate_optimizer != "sgd":
        raise ValueError("FedAdam diagnostic requires local SGD so only the server optimizer changes")
    if mode not in {"fedavg", "fedadam"} and nonprivate_optimizer != "sgd":
        raise ValueError("Non-private optimizer ablations are supported only for FedAvg; DP methods retain their existing mechanism")
    if mode == "fedadam" and aggregation_mode != "uniform":
        raise ValueError("FedAdam diagnostic requires uniform aggregation across the selected cohort")
    if mode == "fedadam" and (server_learning_rate <= 0 or server_tau <= 0):
        raise ValueError("FedAdam server_learning_rate and server_tau must be positive")
    if mode == "fedadam" and not (0.0 <= server_beta1 < 1.0 and 0.0 <= server_beta2 < 1.0):
        raise ValueError("FedAdam server_beta1 and server_beta2 must be in [0, 1)")
    if nonprivate_weight_decay < 0 or nonprivate_max_grad_norm <= 0:
        raise ValueError("weight decay must be nonnegative and max_grad_norm must be positive")

    global_model = model_factory().to(device)
    initial_model_sha256 = _state_dict_sha256(global_model.state_dict())
    client_ids = sorted(str(cid) for cid in client_partitions)
    if not client_ids:
        raise ValueError("No training clients found")
    participation_schedule = _validate_participation_schedule(
        participation_schedule,
        client_ids,
        rounds,
        mode,
        allow_private_partial_participation=bool(partial_dp_accounting_enabled),
    )
    master_rng = np.random.default_rng(seed)
    # FedAdam moments persist across communication rounds and are initialized
    # once from the global model schema. They are never reset per client/round.
    first_moment = None
    second_moment = None
    if server_optimizer == "fedadam":
        first_moment = {
            key: torch.zeros_like(value, dtype=torch.float32, device=device)
            for key, value in global_model.state_dict().items()
            if torch.is_floating_point(value)
        }
        second_moment = {
            key: torch.zeros_like(value, dtype=torch.float32, device=device)
            for key, value in global_model.state_dict().items()
            if torch.is_floating_point(value)
        }
    history = []
    for round_index in range(rounds):
        round_client_ids = (
            client_ids if participation_schedule is None
            else participation_schedule[round_index]
        )
        round_deltas = []
        round_client_diagnostics = []
        client_stats = []
        global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        sampled_total = 0
        sampled_positive_total = 0
        local_losses = []

        for client_id in round_client_ids:
            local_model = copy.deepcopy(global_model)
            local_rng = np.random.default_rng(int(master_rng.integers(0, 2**32 - 1)))
            groups = client_partitions[client_id]
            candidates = _all_client_indices(groups)
            # A fresh local optimizer is created for each client in each round.
            # Its AdamW moments persist over that client's local steps only;
            # model parameters are refreshed from the global model next round.
            local_optimizer = None
            if mode == "fedavg" and nonprivate_optimizer == "adamw" and len(candidates):
                local_optimizer = torch.optim.AdamW(
                    local_model.parameters(),
                    lr=float(learning_rate),
                    weight_decay=float(nonprivate_weight_decay),
                )
            client_step_stats = []
            for local_step_index in range(local_steps_per_round):
                if mode in {"fedavg", "fedadam"}:
                    stat = _nonprivate_step(
                        local_model, x_array, y_array, candidates,
                        batch_size_nonprivate, learning_rate, device, local_rng,
                        optimizer=local_optimizer,
                        max_grad_norm=nonprivate_max_grad_norm,
                    )
                elif mode == "uniform_dp":
                    stat = private_uniform_step(
                        local_model, x_array, y_array, candidates, q_uniform, sigma_uniform,
                        clip_norm, normalization, learning_rate, device,
                        secure_randomness=secure_randomness, test_rng=local_rng,
                    )
                elif mode == "ta_dpfl":
                    focus_group = schedule[round_index * local_steps_per_round + local_step_index]
                    stat = private_typology_step(
                        local_model, x_array, y_array, groups, focus_group,
                        q_focus[focus_group], q_background, sigmas[focus_group],
                        clip_norm, normalization, learning_rate, device,
                        secure_randomness=secure_randomness, test_rng=local_rng,
                    )
                else:
                    # Group membership is already represented by the client's
                    # mutually-exclusive group_indices mapping. A second global
                    # groups_array is not needed by the stratum-sum mechanism.
                    if sigma_by_group is None:
                        raise ValueError("individualized_dp requires sigma_by_group")
                    stat = private_individualized_step(
                        local_model, x_array, y_array, groups,
                        q_uniform, sigma_by_group, clip_norm, normalization,
                        learning_rate, device, secure_randomness=secure_randomness,
                        test_rng=local_rng,
                    )
                client_step_stats.append(stat)
                client_stats.append(stat)
                sampled_total += int(stat.get("sampled_count", 0))
                if mode in {"fedavg", "fedadam"}:
                    sampled_positive_total += int(stat.get("sampled_positive_count", 0))
                if stat.get("loss") is not None:
                    local_losses.append(float(stat["loss"]))

            local_state = local_model.state_dict()
            local_update_l2 = _state_delta_l2(local_state, global_state)
            delta = {k: local_state[k].detach() - global_state[k] for k in global_state}
            round_deltas.append(delta)

            if mode in {"fedavg", "fedadam"}:
                round_client_diagnostics.append({
                    "client_id": str(client_id),
                    "training_rows": int(len(candidates)),
                    "training_positive_rows": int(np.asarray(y_array[candidates]).sum()) if len(candidates) else 0,
                    "sampled_records": int(sum(s.get("sampled_count", 0) for s in client_step_stats)),
                    "sampled_positive_records": int(sum(s.get("sampled_positive_count", 0) for s in client_step_stats)),
                    "mean_local_loss": float(np.mean([s["loss"] for s in client_step_stats if s.get("loss") is not None])) if any(s.get("loss") is not None for s in client_step_stats) else None,
                    "local_update_l2": local_update_l2,
                    # For the paired non-private FedAvg/FedAdam experiment,
                    # this verifies that each client saw the identical batch.
                    "sampled_indices_sha256": hashlib.sha256(
                        "|".join(
                            str(s.get("sampled_indices_sha256", ""))
                            for s in client_step_stats
                        ).encode("ascii")
                    ).hexdigest(),
                    "initial_model_sha256": initial_model_sha256,
                })

        # Uniform is the paper's declared client objective. sample_size is a
        # non-private diagnostic control to isolate the impact of virtual-client
        # workload imbalance; it must never be used for DP comparisons.
        if aggregation_mode == "uniform":
            client_weights = np.full(len(round_client_ids), 1.0 / len(round_client_ids), dtype=np.float64)
        else:
            client_sizes = np.asarray(
                [len(_all_client_indices(client_partitions[c])) for c in round_client_ids],
                dtype=np.float64,
            )
            if client_sizes.sum() <= 0:
                raise ValueError("Cannot sample-size-weight clients with zero total training rows")
            client_weights = client_sizes / client_sizes.sum()

        with torch.no_grad():
            aggregated_delta = {}
            aggregate_sq = 0.0
            for key in global_state:
                weighted_delta = torch.zeros_like(global_state[key])
                for weight, delta in zip(client_weights, round_deltas):
                    weighted_delta.add_(delta[key], alpha=float(weight))
                aggregate_sq += float(torch.sum(weighted_delta.float() ** 2).item())
                aggregated_delta[key] = weighted_delta

            if server_optimizer == "fedadam":
                updated, server_update_l2 = _fedadam_server_update(
                    global_state,
                    aggregated_delta,
                    first_moment,
                    second_moment,
                    step=round_index + 1,
                    learning_rate=float(server_learning_rate),
                    beta1=float(server_beta1),
                    beta2=float(server_beta2),
                    tau=float(server_tau),
                )
            else:
                updated = {
                    key: global_state[key] + aggregated_delta[key]
                    for key in global_state
                }
                server_update_l2 = float(np.sqrt(aggregate_sq))
            global_model.load_state_dict(updated)

        is_private_mode = mode in {"uniform_dp", "ta_dpfl", "individualized_dp"}
        round_record = {
            "round": round_index + 1,
            "clients": len(round_client_ids),
            "initial_model_sha256": initial_model_sha256,
            "eligible_clients": len(client_ids),
            "participation_mode": "balanced_partial" if participation_schedule is not None else "all_clients",
            "aggregation_mode": aggregation_mode,
            "local_optimizer": nonprivate_optimizer if mode in {"fedavg", "fedadam"} else "record_level_dp_sgd",
            "server_optimizer": server_optimizer,
            "server_learning_rate": float(server_learning_rate) if server_optimizer == "fedadam" else None,
            "server_beta1": float(server_beta1) if server_optimizer == "fedadam" else None,
            "server_beta2": float(server_beta2) if server_optimizer == "fedadam" else None,
            # Norms are post-processing of the protected client deltas in DP modes.
            "aggregated_client_delta_l2": float(np.sqrt(aggregate_sq)),
            "server_update_l2": float(server_update_l2),
            # Raw local loss and realized Poisson sample counts depend on private
            # records. They are deliberately not written to DP experiment logs.
            "mean_sampled_per_local_step": None if is_private_mode else (float(np.mean([s.get("sampled_count", 0) for s in client_stats])) if client_stats else 0.0),
            "total_sampled_records": None if is_private_mode else int(sampled_total),
            "mean_local_loss": None if is_private_mode else (float(np.mean(local_losses)) if local_losses else None),
            "mean_local_update_l2": float(np.mean([c["local_update_l2"] for c in round_client_diagnostics])) if round_client_diagnostics else None,
            "aggregated_update_l2": float(np.sqrt(aggregate_sq)),
        }
        if mode in {"fedavg", "fedadam"}:
            round_record["sampled_positive_records"] = int(sampled_positive_total)
            round_record["clients_with_sampled_positive"] = int(sum(c["sampled_positive_records"] > 0 for c in round_client_diagnostics))
            round_record["client_diagnostics"] = round_client_diagnostics
        history.append(round_record)

    return global_model, history
