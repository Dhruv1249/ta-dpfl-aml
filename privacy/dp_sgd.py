from __future__ import annotations

import math
import secrets

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ndtri


def poisson_sample_indices(indices: np.ndarray, q: float, secure: bool = True, rng: np.random.Generator | None = None) -> np.ndarray:
    """Independent Bernoulli inclusion (Poisson sampling) over `indices`.

    In secure mode, geometric gaps are generated from Python's OS-backed
    `secrets` source. This avoids using a publicly seeded PRNG for the private
    sampling decisions. In test mode, NumPy's Generator can be used instead.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if not 0.0 <= q <= 1.0:
        raise ValueError("Poisson inclusion probability q must be in [0, 1]")
    if q == 0.0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    if q == 1.0:
        return indices.copy()
    if not secure:
        if rng is None:
            raise ValueError("Pass a NumPy Generator when secure=False")
        mask = rng.random(len(indices)) < q
        return indices[mask]

    log_failure = math.log1p(-q)
    selected_positions = []
    position = -1
    n = len(indices)
    scale = float(1 << 53)
    while True:
        u = (secrets.randbits(53) + 0.5) / scale
        failures = int(math.floor(math.log(u) / log_failure))
        position += failures + 1
        if position >= n:
            break
        selected_positions.append(position)
    return indices[np.asarray(selected_positions, dtype=np.int64)]


def _secure_standard_normal(shape, device, dtype=torch.float32) -> torch.Tensor:
    count = int(np.prod(shape))
    if count == 0:
        return torch.empty(shape, device=device, dtype=dtype)
    raw = secrets.token_bytes(count * 8)
    integers = np.frombuffer(raw, dtype=np.uint64, count=count)
    # Midpoint mapping avoids exactly 0 and 1 before inverse-normal transform.
    uniform = ((integers >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)
    normal = ndtri(uniform).astype(np.float32).reshape(shape)
    return torch.from_numpy(normal).to(device=device, dtype=dtype)


def per_sample_clipped_gradient(model, x_one: torch.Tensor, y_one: torch.Tensor, clip_norm: float):
    """Return one example's globally L2-clipped gradient over all parameters."""
    model.zero_grad(set_to_none=True)
    logit = model(x_one.unsqueeze(0))
    loss = F.binary_cross_entropy_with_logits(logit, y_one.reshape(1), reduction="sum")
    grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=False, create_graph=False)
    squared_norm = torch.zeros((), device=x_one.device)
    for grad in grads:
        squared_norm = squared_norm + torch.sum(grad.detach() ** 2)
    norm = torch.sqrt(squared_norm)
    factor = torch.clamp(torch.as_tensor(clip_norm, device=x_one.device) / (norm + 1e-12), max=1.0)
    return [grad.detach() * factor for grad in grads], float(norm.detach().cpu())


def private_sgd_step_from_indices(
    model,
    x_array: np.ndarray,
    y_array: np.ndarray,
    sampled_indices: np.ndarray,
    clip_norm: float,
    noise_multiplier: float,
    normalization: float,
    learning_rate: float,
    device: torch.device,
    secure_randomness: bool = True,
):
    """One per-example clipped, summed-gradient Gaussian mechanism update."""
    if clip_norm <= 0 or noise_multiplier <= 0 or normalization <= 0 or learning_rate <= 0:
        raise ValueError("clip_norm, noise_multiplier, normalization and learning_rate must be > 0")
    params = tuple(model.parameters())
    grad_sums = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]
    observed_norms = []

    for idx in np.asarray(sampled_indices, dtype=np.int64):
        x_one = torch.as_tensor(x_array[idx], dtype=torch.float32, device=device)
        y_one = torch.as_tensor(float(y_array[idx]), dtype=torch.float32, device=device)
        clipped, norm = per_sample_clipped_gradient(model, x_one, y_one, clip_norm)
        observed_norms.append(norm)
        for acc, grad in zip(grad_sums, clipped):
            acc.add_(grad)

    # The focus and BG samples are already combined in sampled_indices. Add
    # exactly one noise tensor to this combined clipped sum.
    with torch.no_grad():
        for parameter, grad_sum in zip(params, grad_sums):
            if secure_randomness:
                noise = _secure_standard_normal(parameter.shape, parameter.device, parameter.dtype)
            else:
                noise = torch.randn_like(parameter)
            privatized_gradient = (grad_sum + noise * (noise_multiplier * clip_norm)) / normalization
            parameter.add_(privatized_gradient, alpha=-learning_rate)

    return {
        "sampled_count": int(len(sampled_indices)),
        "max_unclipped_norm_observed": max(observed_norms) if observed_norms else 0.0,
    }


def private_typology_step(
    model,
    x_array,
    y_array,
    group_indices: dict[str, np.ndarray],
    focus_group: str,
    q_focus: float,
    q_background: float,
    noise_multiplier: float,
    clip_norm: float,
    normalization: float,
    learning_rate: float,
    device: torch.device,
    secure_randomness: bool = True,
    test_rng: np.random.Generator | None = None,
):
    focus = poisson_sample_indices(group_indices[focus_group], q_focus, secure_randomness, test_rng)
    background = poisson_sample_indices(group_indices["BG"], q_background, secure_randomness, test_rng)
    sampled = np.concatenate([focus, background])
    return private_sgd_step_from_indices(
        model, x_array, y_array, sampled, clip_norm, noise_multiplier,
        normalization, learning_rate, device, secure_randomness,
    )


def private_uniform_step(
    model, x_array, y_array, all_indices, q_uniform, noise_multiplier,
    clip_norm, normalization, learning_rate, device,
    secure_randomness=True, test_rng=None,
):
    sampled = poisson_sample_indices(all_indices, q_uniform, secure_randomness, test_rng)
    return private_sgd_step_from_indices(
        model, x_array, y_array, sampled, clip_norm, noise_multiplier,
        normalization, learning_rate, device, secure_randomness,
    )


def private_individualized_step(
    model,
    x_array: np.ndarray,
    y_array: np.ndarray,
    group_indices: dict[str, np.ndarray],
    q_uniform: float,
    sigma_by_group: dict[str, float],
    clip_norm: float,
    normalization: float,
    learning_rate: float,
    device: torch.device,
    secure_randomness: bool = True,
    test_rng: np.random.Generator | None = None,
):
    """Stratum-specific Gaussian baseline for individualized record privacy.

    For each mutually-exclusive fixed stratum, independently Poisson-sample
    records at q_uniform, sum their globally clipped per-example gradients,
    and add one independent Gaussian vector with that stratum's multiplier.
    A noise vector is added for every stratum on every step, including empty
    samples. The privatized stratum sums are then added and normalized.

    A neighboring change to one transaction affects only its own stratum
    mechanism. Releasing the sum is post-processing of the tuple of
    per-stratum sampled Gaussian mechanisms. This is an explicit operational
    individualized-DP-SGD-style comparator, not a claim to reproduce a unique
    external implementation.
    """
    if not 0.0 < float(q_uniform) <= 1.0:
        raise ValueError("q_uniform must be in (0, 1]")
    if clip_norm <= 0 or normalization <= 0 or learning_rate <= 0:
        raise ValueError("clip_norm, normalization and learning_rate must be > 0")
    if not sigma_by_group:
        raise ValueError("sigma_by_group must not be empty")
    if set(sigma_by_group) != set(group_indices):
        raise ValueError(
            "sigma_by_group and local group_indices must cover exactly the same strata"
        )

    params = tuple(model.parameters())
    privatized_sum = [
        torch.zeros_like(p, memory_format=torch.preserve_format) for p in params
    ]
    total_sampled = 0
    max_norm_observed = 0.0

    for group, sigma_value in sigma_by_group.items():
        sigma = float(sigma_value)
        if sigma <= 0 or not math.isfinite(sigma):
            raise ValueError(f"Invalid noise multiplier for stratum {group}")
        candidates = np.asarray(group_indices[group], dtype=np.int64)
        sampled = poisson_sample_indices(
            candidates, float(q_uniform), secure_randomness, test_rng
        )
        total_sampled += int(len(sampled))
        grad_sums = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]
        for idx in sampled:
            x_one = torch.as_tensor(x_array[idx], dtype=torch.float32, device=device)
            y_one = torch.as_tensor(float(y_array[idx]), dtype=torch.float32, device=device)
            clipped, unclipped_norm = per_sample_clipped_gradient(
                model, x_one, y_one, clip_norm
            )
            max_norm_observed = max(max_norm_observed, float(unclipped_norm))
            for acc, grad in zip(grad_sums, clipped):
                acc.add_(grad)

        # Independent Gaussian mechanism for this fixed stratum, even when
        # its sampled batch is empty. Noise is added once per stratum, not once
        # per sampled record.
        with torch.no_grad():
            for acc, grad_sum in zip(privatized_sum, grad_sums):
                if secure_randomness:
                    noise = _secure_standard_normal(
                        grad_sum.shape, grad_sum.device, grad_sum.dtype
                    )
                else:
                    noise = torch.randn_like(grad_sum)
                acc.add_(grad_sum + noise * (sigma * clip_norm))

    with torch.no_grad():
        for parameter, grad_sum in zip(params, privatized_sum):
            parameter.add_(grad_sum / normalization, alpha=-learning_rate)

    return {
        "sampled_count": int(total_sampled),
        "max_unclipped_norm_observed": float(max_norm_observed),
        "mechanism": "independent stratum-level sampled Gaussian sums; one noise vector per stratum per local step",
    }
