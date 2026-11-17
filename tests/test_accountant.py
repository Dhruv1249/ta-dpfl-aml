import numpy as np
import pytest

from privacy.rdp_accountant import (
    compute_rdp_poisson_gaussian,
    rdp_to_epsilon,
    validate_against_opacus,
)


def test_zero_sampling_has_zero_rdp_and_epsilon():
    orders = [2, 3, 4, 8]
    rdp = compute_rdp_poisson_gaussian(0.0, 1.3, orders)
    assert np.all(rdp == 0)
    eps, _ = rdp_to_epsilon(orders, rdp, 1e-6)
    assert eps == 0.0


def test_full_sampling_is_gaussian_rdp():
    orders = np.array([2, 3, 4, 8])
    sigma = 1.7
    actual = compute_rdp_poisson_gaussian(1.0, sigma, orders)
    expected = orders / (2.0 * sigma * sigma)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)



def test_subsampled_gaussian_rdp_matches_independent_integer_order_expression():
    """Regression test for the (i^2-i)/(2 sigma^2) exponent.

    The q=1 test alone cannot catch mistakes in the subsampling expression.
    This direct reference calculation deliberately uses a separate loop.
    """
    from math import comb, log
    from scipy.special import logsumexp

    q = 0.01
    sigma = 1.2
    orders = [2, 3, 4, 8, 16]
    actual = compute_rdp_poisson_gaussian(q, sigma, orders)

    expected = []
    for alpha in orders:
        terms = []
        for i in range(alpha + 1):
            log_probability = (
                log(comb(alpha, i))
                + i * log(q)
                + (alpha - i) * log(1.0 - q)
            )
            gaussian_exponent = (i * i - i) / (2.0 * sigma * sigma)
            terms.append(log_probability + gaussian_exponent)
        expected.append(float(logsumexp(terms)) / (alpha - 1))

    assert np.allclose(actual, np.asarray(expected), rtol=1e-12, atol=1e-12)

def test_opacus_independent_validation_when_installed():
    result = validate_against_opacus()
    if result["status"] == "UNAVAILABLE":
        pytest.skip(result["detail"])
    assert result["status"] == "PASS", result


def test_typology_noise_calibration_meets_all_stratum_and_bg_targets():
    from privacy.calibrate import calibrate_typology_noise
    from privacy.scheduler import build_schedule
    from ta_dpfl_constants import POSITIVE_GROUPS

    relevance = {g: 0.5 for g in POSITIVE_GROUPS}
    _, counts, _ = build_schedule(relevance, lam=0.2, eta=0.01, total_steps=50)
    q_focus = {g: 0.001 for g in POSITIVE_GROUPS}
    targets = {g: 8.0 for g in POSITIVE_GROUPS}
    result = calibrate_typology_noise(
        counts, q_focus, 0.001, targets, 8.0, 1e-6,
        [2, 3, 4, 5, 8, 16, 32, 64], require_opacus_validation=False,
    )
    assert result["target_satisfied"]
    assert result["background"]["epsilon"] <= 8.00001
    assert all(v["epsilon"] <= 8.00001 for v in result["positive_groups"].values())
    assert result["calibration_optimality_certified"] is False
