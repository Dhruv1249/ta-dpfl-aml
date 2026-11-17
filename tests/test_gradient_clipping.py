import torch

from models.aml_classifier import AMLClassifier
from privacy.dp_sgd import per_sample_clipped_gradient


def test_per_example_global_norm_is_clipped():
    torch.manual_seed(4)
    model = AMLClassifier(input_dim=5, hidden_dims=(4,))
    x = torch.randn(5)
    y = torch.tensor(1.0)
    clip = 0.25
    clipped, _ = per_sample_clipped_gradient(model, x, y, clip)
    norm = torch.sqrt(sum(torch.sum(g ** 2) for g in clipped))
    assert float(norm) <= clip + 1e-6
