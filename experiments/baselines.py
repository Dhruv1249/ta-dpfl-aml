from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader, TensorDataset



def train_central_nn(
    model_factory,
    x,
    y,
    epochs,
    batch_size,
    learning_rate,
    device,
    seed,
    pos_weight=1.0,
):
    torch.manual_seed(seed)

    if pos_weight <= 0:
        raise ValueError("pos_weight must be positive")

    model = model_factory().to(device)

    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    positive_weight = torch.tensor(
        float(pos_weight),
        dtype=torch.float32,
        device=device,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_records = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)

            loss = F.binary_cross_entropy_with_logits(
                logits,
                yb,
                pos_weight=positive_weight,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += loss.item() * len(yb)
            total_records += len(yb)

        average_loss = total_loss / max(1, total_records)

        print(
            f"Central MLP | Epoch {epoch}/{epochs} "
            f"| Mean loss: {average_loss:.8f} "
            f"| pos_weight: {pos_weight}"
        )

    return model



def train_central_gbdt(x, y, max_rows, max_iter, seed):
    """Train an auditable capped-row GBDT reference; actual row count is returned."""
    n = len(y)
    rng = np.random.default_rng(seed)
    if n > max_rows:
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        pos_take = min(len(positives), max(1, max_rows // 2))
        neg_take = max_rows - pos_take
        chosen = np.concatenate([
            rng.choice(positives, size=pos_take, replace=False) if pos_take else np.empty(0, dtype=int),
            rng.choice(negatives, size=min(neg_take, len(negatives)), replace=False),
        ])
        rng.shuffle(chosen)
    else:
        chosen = np.arange(n)
    model = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)
    model.fit(x[chosen], y[chosen])
    return model, int(len(chosen))
