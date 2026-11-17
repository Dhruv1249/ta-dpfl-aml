from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml

from ta_dpfl_constants import POSITIVE_GROUPS


def _normalize_max(scores):
    maximum = max(scores.values())
    if maximum <= 0:
        return {g: 1.0 for g in scores}
    return {g: float(v / maximum) for g, v in scores.items()}


def make_controls(config_path: str, output_dir: str, seed: int = 2026):
    with open(config_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    relevance = base.get("privacy", {}).get("relevance_scores")
    if not isinstance(relevance, dict) or set(relevance) != set(POSITIVE_GROUPS):
        raise ValueError("Base config must contain relevance_scores for all nine positive strata")
    base_scores = {g: float(relevance[g]) for g in POSITIVE_GROUPS}
    if any(v < 0 or v > 1 for v in base_scores.values()):
        raise ValueError("All base relevance scores must be within [0, 1]")

    rng = np.random.default_rng(seed)
    vals = np.asarray([base_scores[g] for g in POSITIVE_GROUPS], dtype=float)
    variants = {
        "uniform": {g: 1.0 for g in POSITIVE_GROUPS},
        "inverted": _normalize_max({g: 1.0 - base_scores[g] for g in POSITIVE_GROUPS}),
        "permuted": _normalize_max(dict(zip(POSITIVE_GROUPS, rng.permutation(vals).tolist()))),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, scores in variants.items():
        config = copy.deepcopy(base)
        config["privacy"]["relevance_scores"] = scores
        config["output_dir"] = f"outputs/ablation_{name}"
        path = output_dir / f"relevance_{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        written.append(str(path))
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate relevance-score ablation configs.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="configs/ablations")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for path in make_controls(args.config, args.output_dir, args.seed):
        print(path)


if __name__ == "__main__":
    main()
