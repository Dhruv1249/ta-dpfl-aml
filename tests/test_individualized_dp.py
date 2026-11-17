import numpy as np
import torch

from models.aml_classifier import AMLClassifier
from privacy.dp_sgd import private_individualized_step


def test_individualized_baseline_runs_without_typology_schedule():
    torch.manual_seed(1)
    model = AMLClassifier(input_dim=3, hidden_dims=(4,))
    x = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.9, 0.2, 0.5]], dtype=np.float32)
    y = np.array([1, 0, 0], dtype=np.float32)
    groups = np.array(["CYCLE", "BG", "BG"])
    group_indices = {"CYCLE": np.array([0]), "BG": np.array([1, 2])}
    before = [p.detach().clone() for p in model.parameters()]
    result = private_individualized_step(
        model, x, y, group_indices, q_uniform=1.0,
        sigma_by_group={"CYCLE": 0.5, "BG": 1.0},
        clip_norm=1.0, normalization=3.0, learning_rate=0.01,
        device=torch.device("cpu"), secure_randomness=False,
        test_rng=np.random.default_rng(3),
    )
    assert result["sampled_count"] == 3
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
