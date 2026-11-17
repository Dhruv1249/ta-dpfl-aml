import torch
from torch import nn


class AMLClassifier(nn.Module):
    """Transaction-local MLP; typology, account and bank IDs are not inputs."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU()])
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)
