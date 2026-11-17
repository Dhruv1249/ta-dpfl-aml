from __future__ import annotations

import numpy as np
import pandas as pd

from ta_dpfl_constants import ALL_GROUPS


def build_client_partitions(frame: pd.DataFrame, split: str = "train", max_clients: int | None = None, public_client_ids: list[str] | None = None):
    """Return {client_id: {group: row_indices}} over the global feature matrix."""
    subset = frame.loc[frame["split"].eq(split), ["client", "group"]]
    observed_clients = set(frame["client"].astype(str).unique().tolist())
    if public_client_ids is None:
        clients = sorted(observed_clients)
    else:
        clients = [str(v) for v in public_client_ids]
        if len(clients) != len(set(clients)):
            raise ValueError("public_client_ids must not contain duplicates")
        if not observed_clients.issubset(set(clients)):
            unexpected = sorted(observed_clients - set(clients))
            raise ValueError(f"Observed clients missing from predeclared public roster: {unexpected[:10]}")
    if not clients:
        raise ValueError("The public client roster is empty")
    if max_clients is not None:
        if max_clients < 1:
            raise ValueError("max_clients must be positive or null")
        clients = clients[:max_clients]
    index_map = {}
    for client in clients:
        client_rows = frame.index[frame["split"].eq(split) & frame["client"].astype(str).eq(client)].to_numpy(dtype=np.int64)
        groups = frame.loc[client_rows, "group"].to_numpy()
        index_map[client] = {g: client_rows[groups == g] for g in ALL_GROUPS}
    return dict(index_map)
