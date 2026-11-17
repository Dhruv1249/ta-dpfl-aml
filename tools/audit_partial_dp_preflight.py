from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated.training import build_balanced_participation_schedule
from privacy.partial_participation_accountant import build_client_focus_exposure_counts
from privacy.rdp_accountant import validate_against_opacus
from privacy.scheduler import build_schedule
from ta_dpfl_constants import POSITIVE_GROUPS
from run_experiment import _build_partial_dp_privacy_reports, _strip_internal_rdp_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the partial-participation DP schedules and accountant.")
    parser.add_argument("--config", default="configs/partial_dp_fedadam_3200x1.yaml")
    parser.add_argument("--output", default="outputs/partial_dp_preflight.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fed = cfg["federated"]
    privacy = cfg["privacy"]
    experiments = cfg["experiments"]
    if fed.get("clients_per_round") is None:
        raise ValueError("Partial-DP preflight requires clients_per_round")
    if fed.get("participation_schedule_seed") is None:
        raise ValueError("A fixed participation_schedule_seed is required")
    if any(experiments.get(k, False) for k in (
        "run_central_nn", "run_central_gbdt", "run_fedavg", "run_fedadam"
    )):
        raise ValueError("Non-private methods must be disabled in this DP output config")

    ids = cfg["data"].get("public_client_ids")
    if not ids:
        raise ValueError("The predeclared public source-client roster is missing")
    manifest = Path(fed["virtual_client_manifest"])
    if not manifest.exists():
        raise FileNotFoundError(f"Virtual-client manifest not found: {manifest}")
    import pandas as pd
    mapping = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required_columns = {"source_client_id", "virtual_client_id"}
    if not required_columns.issubset(mapping.columns):
        raise ValueError(f"Manifest must contain {sorted(required_columns)}")
    if mapping["source_client_id"].duplicated().any():
        raise ValueError("Manifest assigns at least one source client more than once")
    configured_roster = set(str(value) for value in ids)
    manifest_roster = set(mapping["source_client_id"].astype(str))
    if configured_roster != manifest_roster:
        raise ValueError(
            "Virtual-client manifest source IDs do not exactly match the configured public roster "
            f"(config={len(configured_roster)}, manifest={len(manifest_roster)})"
        )
    client_col_candidates = [c for c in mapping.columns if c.lower() in {"virtual_client_id", "virtual_client", "virtual_id"}]
    if len(client_col_candidates) != 1:
        raise ValueError(f"Could not identify unique virtual client ID column in {mapping.columns.tolist()}")
    virtual_ids = sorted(mapping[client_col_candidates[0]].dropna().astype(str).unique().tolist())

    rounds = int(fed["rounds"])
    local_steps = int(fed["local_steps_per_round"])
    participants = int(fed["clients_per_round"])
    participation_seed = int(fed["participation_schedule_seed"])
    participation = build_balanced_participation_schedule(
        virtual_ids, rounds, participants, participation_seed
    )
    relevance = {g: float(privacy["relevance_scores"][g]) for g in POSITIVE_GROUPS}
    probabilities, focus_counts, focus_schedule = build_schedule(
        relevance,
        float(privacy["lambda_exploration"]),
        float(privacy["eta"]),
        rounds * local_steps,
    )
    exposure = build_client_focus_exposure_counts(
        participation, focus_schedule, virtual_ids, local_steps
    )
    independent = validate_against_opacus(orders=privacy["alpha_orders"])
    if privacy.get("require_opacus_validation", True) and independent.get("status") != "PASS":
        raise RuntimeError(f"Independent Opacus validation failed: {independent}")

    # Run the same client-exposure-aware calibration used by the training runner.
    # The previous version referenced accountant_summary without defining it,
    # which caused the preflight to fail after schedule and Opacus checks.
    privacy_reports, comparison, sigma_uniform, individualized_sigmas, policy_u, policy_d = (
        _build_partial_dp_privacy_reports(
            privacy,
            fed,
            experiments,
            exposure,
            focus_schedule,
            rounds * local_steps,
            local_steps,
            independent,
        )
    )

    accountant_summary = {
        "calibration_status": "PASS",
        "global_epsilon_by_method": {
            name: float(report["global_epsilon"])
            for name, report in privacy_reports.items()
        },
        "target_satisfied": {
            "uniform_dp": bool(
                float(privacy_reports["uniform_dp"]["global_epsilon"])
                <= float(privacy["epsilon_uniform"]) + 1e-5
            ),
            "individualized_dp": bool(
                float(privacy_reports["individualized_dp"]["global_epsilon"])
                <= float(privacy["epsilon_max"]) + 1e-5
            ),
            "ta_policy_u": bool(policy_u.get("target_satisfied", False)),
            "ta_policy_d": bool(policy_d.get("target_satisfied", False)),
        },
        "uniform_noise_multiplier": float(sigma_uniform),
        "individualized_noise_multipliers": {
            key: float(value) for key, value in individualized_sigmas.items()
        },
        "ta_policy_u_noise_multipliers": {
            key: float(value) for key, value in policy_u["noise_multipliers"].items()
        },
        "ta_policy_d_noise_multipliers": {
            key: float(value) for key, value in policy_d["noise_multipliers"].items()
        },
        "regime_a_setup_satisfied": bool(comparison["regime_a_setup_satisfied"]),
        "matched_maximum_privacy_cap_per_method": bool(
            comparison["matched_maximum_privacy_cap_per_method"]
        ),
        "joint_release_all_enabled_dp_methods_and_seeds": comparison.get(
            "joint_release_all_enabled_dp_methods_and_seeds"
        ),
        "comparison_interpretation": comparison["interpretation"],
    }

    if not all(accountant_summary["target_satisfied"].values()):
        raise RuntimeError(
            "At least one partial-DP calibration failed its configured target: "
            f"{accountant_summary['target_satisfied']}"
        )

    report = {
        "status": "PASS",
        "config": str(config_path),
        "virtual_clients": len(virtual_ids),
        "rounds": rounds,
        "clients_per_round": participants,
        "appearances_per_client": exposure["appearances_per_client"],
        "participation_seed": participation_seed,
        "focus_step_counts": focus_counts,
        "focus_probabilities": probabilities,
        "max_client_focus_exposures": exposure["max_client_focus_exposures"],
        "min_client_focus_exposures": exposure["min_client_focus_exposures"],
        "independent_rdp_validation": independent,
        "accountant_calibration": accountant_summary,
        "privacy_comparison": comparison,
        "privacy_reports": {
            name: _strip_internal_rdp_arrays(value)
            for name, value in privacy_reports.items()
        },
        "non_private_methods_disabled": True,
        "secure_randomness_enabled": bool(privacy.get("secure_randomness")),
        "training_performed": False,
        "privacy_claim_note": (
            "Preflight validates schedule construction, feasible target calibration, and the accountant's "
            "independent RDP check when Opacus is available. It does not validate the complete end-to-end "
            "mechanism, operating environment, or research assumptions, and is not by itself a formal privacy certification."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Preflight report saved: {output}")


if __name__ == "__main__":
    main()
