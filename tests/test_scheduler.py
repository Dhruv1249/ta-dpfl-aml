from ta_dpfl_constants import POSITIVE_GROUPS
from privacy.scheduler import build_schedule


def test_schedule_is_public_positive_and_sums_to_t():
    relevance = {g: 0.5 for g in POSITIVE_GROUPS}
    probabilities, counts, schedule = build_schedule(relevance, lam=0.2, eta=0.01, total_steps=50)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-10
    assert all(v > 0 for v in probabilities.values())
    assert all(counts[g] >= 1 for g in POSITIVE_GROUPS)
    assert sum(counts.values()) == 50
    assert len(schedule) == 50
    for group, count in counts.items():
        assert schedule.count(group) == count
