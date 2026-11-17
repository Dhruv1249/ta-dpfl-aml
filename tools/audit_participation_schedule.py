import argparse
import pandas as pd
import sys
from pathlib import Path

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated.training import build_balanced_participation_schedule

p = argparse.ArgumentParser()
p.add_argument("--clients", type=int, default=256)
p.add_argument("--rounds", type=int, default=3200)
p.add_argument("--clients-per-round", type=int, default=4)
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--output", default="outputs/partial_participation_schedule_preview.csv")
a = p.parse_args()
ids = [f"VC{i:04d}" for i in range(a.clients)]
schedule = build_balanced_participation_schedule(ids, a.rounds, a.clients_per_round, a.seed)
rows = [{"round": r, "client_id": cid} for r, cohort in enumerate(schedule, 1) for cid in cohort]
df = pd.DataFrame(rows)
counts = df.groupby("client_id").size()
assert len(df) == a.rounds * a.clients_per_round
assert counts.nunique() == 1
assert counts.iloc[0] == a.rounds * a.clients_per_round // a.clients
out = pd.io.common.stringify_path(a.output)
from pathlib import Path
Path(out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("Schedule PASS")
print("rounds:", a.rounds)
print("clients per round:", a.clients_per_round)
print("total client-round participations:", len(df))
print("appearances per client:", int(counts.iloc[0]))
print("saved:", out)
