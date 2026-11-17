from __future__ import annotations

import argparse
import copy
import json
from logging import config
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.preprocessing import load_and_label_dataset, build_feature_matrix, write_data_audit
from evaluation.metrics import select_threshold_at_fpr, evaluate_overall, evaluate_by_typology
from evaluation.visualize_results import generate_result_plots
from experiments.baselines import train_central_nn, train_central_gbdt
from federated.partition import build_client_partitions
from federated.virtual_partition import build_virtual_client_partitions, load_virtual_client_manifest
from federated.training import run_federated, build_balanced_participation_schedule
from models.aml_classifier import AMLClassifier
from privacy.calibrate import calibrate_uniform_noise, calibrate_typology_noise
from privacy.rdp_accountant import compute_rdp_poisson_gaussian, rdp_to_epsilon, validate_against_opacus
from privacy.scheduler import build_schedule
from privacy.partial_participation_accountant import (
    build_client_focus_exposure_counts,
    calibrate_partial_typology_noise,
    account_partial_uniform,
    account_partial_stratified_uniform,
    compose_partial_record_rdp,
)
from ta_dpfl_constants import POSITIVE_GROUPS


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Configuration YAML must contain a mapping")
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _required_privacy_values(cfg):
    required = ["epsilon_uniform", "epsilon_min", "epsilon_max", "relevance_scores"]
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ValueError(
            "Complete configs/experiment.yaml privacy settings before DP experiments. "
            f"Missing: {missing}. The paper does not specify these numerical values."
        )
    return


def _q_map(value):
    if isinstance(value, dict):
        return {g: float(value[g]) for g in POSITIVE_GROUPS}
    return {g: float(value) for g in POSITIVE_GROUPS}


def _scores(model, x, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        batch_size = 65536
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=float)


def _predict_sklearn(model, x):
    return model.predict_proba(x)[:, 1]


def _record_results(method, seed, model, x_val, y_val, groups_val, x_test, y_test, groups_test, max_fpr, device, output_dir, privacy=None):
    if isinstance(model, torch.nn.Module):
        val_scores = _scores(model, x_val, device)
        test_scores = _scores(model, x_test, device)
    else:
        val_scores = _predict_sklearn(model, x_val)
        test_scores = _predict_sklearn(model, x_test)

    threshold_info = select_threshold_at_fpr(y_val, val_scores, max_fpr)
    overall = evaluate_overall(y_test, test_scores, threshold_info["threshold"])
    overall.update({
        "method": method,
        "seed": seed,
        "threshold": threshold_info["threshold"],
        "validation_fpr": threshold_info["validation_fpr"],
        "validation_recall": threshold_info["validation_recall"],
        "threshold_status": threshold_info["status"],
    })
    if privacy:
        overall["global_epsilon"] = privacy.get("global_epsilon")
        overall["delta"] = privacy.get("delta")
    typology = evaluate_by_typology(groups_test, y_test, test_scores, threshold_info["threshold"])
    for row in typology:
        row.update({"method": method, "seed": seed, "threshold": threshold_info["threshold"]})
    return overall, typology


def _save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")




def _strip_internal_rdp_arrays(report):
    """Remove detailed per-client RDP vectors from human-readable report files."""
    cleaned = copy.deepcopy(report)
    for key in (
        "positive_rdp_by_client",
        "background_rdp_by_client",
        "client_rdp",
    ):
        cleaned.pop(key, None)
    return cleaned


def _build_partial_dp_privacy_reports(
    privacy_cfg,
    fed_cfg,
    experiment_cfg,
    client_exposure,
    focus_schedule,
    total_steps,
    local_steps_per_round,
    independent_validation,
):
    """Calibrate all partial-participation DP methods against exact client exposure."""
    orders = [int(a) for a in privacy_cfg["alpha_orders"]]
    delta = float(privacy_cfg["delta"])
    q_focus = _q_map(privacy_cfg["q_focus"])
    q_background = float(privacy_cfg["q_background"])
    q_uniform = float(privacy_cfg["q_uniform"])
    relevance = {g: float(privacy_cfg["relevance_scores"][g]) for g in POSITIVE_GROUPS}
    repetitions = len(experiment_cfg.get("random_seeds", [int(2026)]))
    client_counts = client_exposure["client_participation_counts"]
    max_client_steps_per_run = max(client_counts.values()) * int(local_steps_per_round)
    composed_steps = max_client_steps_per_run * repetitions

    # Uniform-DP: each record is Poisson-sampled from its owning client's local
    # population at every one of that client's actual participation steps.
    sigma_uniform, _ = calibrate_uniform_noise(
        q_uniform, composed_steps, float(privacy_cfg["epsilon_uniform"]), delta, orders
    )
    uniform_report = account_partial_uniform(
        q_uniform, sigma_uniform, client_counts, local_steps_per_round,
        orders, delta, composition_repetitions=repetitions,
    )
    # A record of any fixed stratum has the same uniform-DP RDP vector.
    uniform_report["positive_rdp_by_client"] = {
        group: copy.deepcopy(uniform_report["client_rdp"]) for group in POSITIVE_GROUPS
    }
    uniform_report["background_rdp_by_client"] = copy.deepcopy(uniform_report["client_rdp"])
    uniform_report.update({
        "sigma": float(sigma_uniform),
        "target_epsilon": float(privacy_cfg["epsilon_uniform"]),
        "steps_per_record_per_run_max": int(max_client_steps_per_run),
        "composition_repetitions": repetitions,
        "mechanism": "Poisson-sampled clipped-sum Gaussian mechanism at each active client step",
        "independent_accountant_validation": independent_validation,
        "formal_privacy_claim_permitted": False,
        "privacy_claim_readiness": (
            "accountant validation and secure-randomness prerequisites passed; "
            "independent end-to-end mechanism, adjacency, transcript, and implementation review remains required"
        ),
    })

    # Individualized comparator: use one independent Gaussian sum per fixed
    # stratum, including BG. Its operational definition is recorded explicitly.
    individualized_targets = {
        group: float(privacy_cfg["epsilon_min"])
        + (float(privacy_cfg["epsilon_max"]) - float(privacy_cfg["epsilon_min"])) * relevance[group]
        for group in POSITIVE_GROUPS
    }
    individualized_targets["BG"] = float(privacy_cfg["epsilon_min"])
    individualized_sigmas = {}
    for group, target in individualized_targets.items():
        individualized_sigmas[group], _ = calibrate_uniform_noise(
            q_uniform, composed_steps, target, delta, orders
        )
    individualized_report = account_partial_stratified_uniform(
        q_uniform, individualized_sigmas, client_counts,
        local_steps_per_round, orders, delta,
        composition_repetitions=repetitions,
    )
    individualized_report.update({
        "targets": individualized_targets,
        "noise_multipliers": individualized_sigmas,
        "steps_per_record_per_run_max": int(max_client_steps_per_run),
        "composition_repetitions": repetitions,
        "baseline_protocol": (
            "At each active local step, each mutually exclusive stratum is "
            "Poisson-sampled at q_uniform, its clipped gradients are summed, "
            "one independent Gaussian vector is added for that stratum (including "
            "empty samples), and the privatized stratum sums are combined."
        ),
        "not_exact_reproduction": (
            "This is an explicit, record-level-accountable stratum-specific "
            "Gaussian comparator. The source paper names an individualized-DP-SGD-style "
            "baseline but does not prescribe a unique noise-placement implementation."
        ),
        "independent_accountant_validation": independent_validation,
        "formal_privacy_claim_permitted": False,
        "privacy_claim_readiness": (
            "accountant validation and secure-randomness prerequisites passed; "
            "independent end-to-end mechanism, adjacency, transcript, and implementation review remains required"
        ),
    })

    targets_u = {group: float(privacy_cfg["epsilon_uniform"]) for group in POSITIVE_GROUPS}
    policy_u = calibrate_partial_typology_noise(
        client_exposure["client_focus_counts"], q_focus, q_background,
        targets_u, float(privacy_cfg["epsilon_uniform"]), delta, orders,
        composition_repetitions=repetitions,
        require_opacus_validation=bool(privacy_cfg.get("require_opacus_validation", True)),
    )
    policy_u.update({
        "focus_schedule_global_steps": int(total_steps),
        "participation_schedule_seed": int(fed_cfg["participation_schedule_seed"]),
        "independent_accountant_validation": independent_validation,
        "formal_privacy_claim_permitted": False,
        "privacy_claim_readiness": (
            "accountant validation and secure-randomness prerequisites passed; "
            "independent end-to-end mechanism, adjacency, transcript, and implementation review remains required"
        ),
    })

    targets_d = {
        group: float(privacy_cfg["epsilon_min"])
        + (float(privacy_cfg["epsilon_max"]) - float(privacy_cfg["epsilon_min"])) * relevance[group]
        for group in POSITIVE_GROUPS
    }
    policy_d = calibrate_partial_typology_noise(
        client_exposure["client_focus_counts"], q_focus, q_background,
        targets_d, float(privacy_cfg["epsilon_min"]), delta, orders,
        composition_repetitions=repetitions,
        require_opacus_validation=bool(privacy_cfg.get("require_opacus_validation", True)),
    )
    policy_d.update({
        "focus_schedule_global_steps": int(total_steps),
        "participation_schedule_seed": int(fed_cfg["participation_schedule_seed"]),
        "independent_accountant_validation": independent_validation,
        "formal_privacy_claim_permitted": False,
        "privacy_claim_readiness": (
            "accountant validation and secure-randomness prerequisites passed; "
            "independent end-to-end mechanism, adjacency, transcript, and implementation review remains required"
        ),
    })

    reports = {
        "uniform_dp": uniform_report,
        "individualized_dp": individualized_report,
        "ta_policy_u": policy_u,
        "ta_policy_d": policy_d,
    }
    max_relevance = max(relevance.values())
    regime_a_setup = (
        abs(max_relevance - 1.0) <= 1e-8
        and abs(float(privacy_cfg["epsilon_max"]) - float(privacy_cfg["epsilon_uniform"])) <= 1e-8
    )
    tolerance = float(privacy_cfg.get("privacy_matching_tolerance", 0.02))
    epsilon_values = {name: float(report["global_epsilon"]) for name, report in reports.items()}
    uniform_epsilon = epsilon_values["uniform_dp"]
    matched = all(abs(value - uniform_epsilon) <= tolerance for value in epsilon_values.values())

    enabled_methods = []
    if experiment_cfg.get("run_uniform_dp", False):
        enabled_methods.append(reports["uniform_dp"])
    if experiment_cfg.get("run_individualized_dp", False):
        enabled_methods.append(reports["individualized_dp"])
    if experiment_cfg.get("run_ta_policy_u", False):
        enabled_methods.append(reports["ta_policy_u"])
    if experiment_cfg.get("run_ta_policy_d", False):
        enabled_methods.append(reports["ta_policy_d"])
    joint_release = compose_partial_record_rdp(enabled_methods, orders, delta) if enabled_methods else None

    comparison = {
        "accounting_protocol": "exact public partial participation; per-record composition at the owning virtual client; no client-subsampling amplification credited",
        "composition_repetitions_per_method": repetitions,
        "regime_a_setup_satisfied": bool(regime_a_setup),
        "privacy_matching_tolerance": tolerance,
        "global_epsilon_bounds_per_method_across_configured_seeds": epsilon_values,
        "matched_maximum_privacy_cap_per_method": bool(regime_a_setup and matched),
        "joint_release_all_enabled_dp_methods_and_seeds": joint_release,
        "joint_release_warning": (
            "Each method is calibrated to its own matched cap across its configured seeds. "
            "If checkpoints from multiple DP methods are released together, their privacy "
            "losses compose; consult joint_release_all_enabled_dp_methods_and_seeds."
        ),
        "interpretation": (
            "Per-method matched-cap comparison" if regime_a_setup and matched
            else "Report as a privacy-utility frontier; the per-method worst-case bounds do not match within tolerance."
        ),
    }
    return reports, comparison, sigma_uniform, individualized_sigmas, policy_u, policy_d


def main():
    parser = argparse.ArgumentParser(description="Run the paper-aligned TA-DPFL AML experiment.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--audit-only", action="store_true", help="Preprocess and audit data, but do not train.")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiments"]
    fed_cfg = config["federated"]
    model_cfg = config["model"]
    privacy_cfg = config["privacy"]
    need_dp = any(exp_cfg.get(k, False) for k in [
        "run_uniform_dp", "run_individualized_dp", "run_ta_policy_u", "run_ta_policy_d"
    ])

    output_dir = Path(config.get("output_dir", "outputs"))

    # A DP output directory must not contain artifacts from a prior non-private
    # or differently configured run. Model checkpoints, raw logs, or other
    # diagnostics from a non-private baseline would not be covered by this DP run.
    if need_dp:
        nonprivate_enabled = any([
            exp_cfg.get("run_central_nn", True),
            exp_cfg.get("run_central_gbdt", True),
            exp_cfg.get("run_fedavg", True),
            exp_cfg.get("run_fedadam", False),
        ])
        if nonprivate_enabled:
            raise ValueError(
                "Do not mix non-private baselines/checkpoints with DP artifacts in one output directory. "
                "Set run_central_nn, run_central_gbdt, run_fedavg and run_fedadam to false; "
                "the DP server optimizer is configured separately with federated.dp_server_optimizer."
            )
        if args.audit_only:
            raise ValueError(
                "--audit-only writes raw label/typology counts and cannot be combined with a DP output run. "
                "Run dataset auditing under a separate non-private config/output directory."
            )
        if output_dir.exists():
            existing_files = [path for path in output_dir.rglob("*") if path.is_file()]
            if existing_files:
                raise RuntimeError(
                    f"DP output directory {output_dir} is not empty ({len(existing_files)} files found). "
                    "Choose a fresh output directory to avoid mixing stale or non-private artifacts."
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["checkpoints", "logs", "privacy_reports", "results"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    clients_per_round_cfg = fed_cfg.get("clients_per_round")
    if clients_per_round_cfg is not None and not any([
        exp_cfg.get("run_fedavg", False),
        exp_cfg.get("run_fedadam", False),
        need_dp,
    ]):
        raise ValueError("Partial participation requires at least one federated method")
    if clients_per_round_cfg is not None and need_dp:
        if fed_cfg.get("participation_schedule_seed") is None:
            raise ValueError(
                "Partial-DP requires a fixed public federated.participation_schedule_seed. "
                "The same predeclared schedule is used for calibration and every seed."
            )
        if not config["data"].get("public_client_ids"):
            raise ValueError("Partial-DP requires a predeclared public source-client roster")

    if need_dp:
        configured_seeds = [int(value) for value in exp_cfg.get("random_seeds", [config.get("seed", 2026)])]
        if not configured_seeds:
            raise ValueError("DP experiments require at least one configured random seed")
        if len(set(configured_seeds)) != len(configured_seeds):
            raise ValueError("DP random_seeds must be unique; duplicate model releases compose and overwrite checkpoint names")
        _required_privacy_values(privacy_cfg)
        if float(privacy_cfg["epsilon_min"]) > float(privacy_cfg["epsilon_max"]):
            raise ValueError("epsilon_min must be less than or equal to epsilon_max")
        if float(privacy_cfg["epsilon_uniform"]) <= 0 or float(privacy_cfg["epsilon_min"]) <= 0:
            raise ValueError("Privacy epsilon targets must be positive")
        if not privacy_cfg.get("secure_randomness", True):
            raise ValueError(
                "The experiment runner requires OS-backed secure_randomness for DP methods. "
                "Use deterministic randomness only in isolated unit tests."
            )
        if not config["data"].get("public_client_ids"):
            raise ValueError(
                "Set data.public_client_ids to a predeclared public client roster before a DP run. "
                "The participating institution roster is treated as public/fixed in this implementation."
            )
        if fed_cfg.get("aggregation_mode", "uniform") != "uniform":
            raise ValueError("DP methods require uniform aggregation across the selected client cohort")

    frame = load_and_label_dataset(config, derive_split_cutoffs=args.audit_only)
    x, feature_names = build_feature_matrix(
        frame,
        int(config["data"].get("feature_hash_dim", 16)),
        int(config.get("seed", 2026)),
    )

    # The raw audit includes label and typology counts. Keep it out of a DP
    # mechanism output directory. The user can retain the already-generated
    # local audit separately for dataset documentation.
    if not need_dp:
        audit = write_data_audit(frame, feature_names, output_dir / "data_audit.json")
        print(json.dumps({k: audit[k] for k in [
            "rows", "positive_rows", "negative_rows", "unique_clients", "split_rows", "group_counts"
        ]}, indent=2))
    else:
        if (output_dir / "data_audit.json").exists():
            raise RuntimeError(
                f"DP output directory {output_dir} already contains data_audit.json. "
                "Use a fresh output directory so raw audit counts are not bundled with DP artifacts."
            )
        print("DP run: raw data_audit.json omitted from this output directory (local audit counts are not a DP release).")

    if args.audit_only:
        candidate_roster = sorted(frame["client"].astype(str).unique().tolist())
        _save_json(output_dir / "public_client_ids_candidate.json", candidate_roster)
        print("Audit complete. Review outputs/data_audit.json and outputs/public_client_ids_candidate.json.")
        print("Copy frozen split cutoffs and the approved public client roster into configs/experiment.yaml, then run training.")
        return

    train_mask = frame["split"].eq("train").to_numpy()
    val_mask = frame["split"].eq("validation").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)
    y_all = frame["label"].to_numpy(dtype=np.float32)
    groups_all = frame["group"].to_numpy()
    x_train, y_train = x[train_idx], y_all[train_idx]
    x_val, y_val, groups_val = x[val_idx], y_all[val_idx], groups_all[val_idx]
    x_test, y_test, groups_test = x[test_idx], y_all[test_idx], groups_all[test_idx]

    federated_enabled = any([
        exp_cfg.get("run_fedavg", True),
        exp_cfg.get("run_fedadam", False),
        exp_cfg.get("run_uniform_dp", False),
        exp_cfg.get("run_individualized_dp", False),
        exp_cfg.get("run_ta_policy_u", False),
        exp_cfg.get("run_ta_policy_d", False),
    ])

    max_clients = fed_cfg.get("max_clients")
    public_client_ids = config["data"].get("public_client_ids")

    if federated_enabled:
        partition_mode = str(fed_cfg.get("partition_mode", "institutional")).lower()
        if partition_mode == "virtual":
            if max_clients is not None:
                raise ValueError("Set federated.max_clients: null when using a pre-audited virtual-client manifest.")
            manifest_path = fed_cfg.get("virtual_client_manifest")
            if not manifest_path:
                raise ValueError("federated.virtual_client_manifest is required for partition_mode: virtual")
            if need_dp:
                manifest_mapping = load_virtual_client_manifest(manifest_path)
                configured_roster = set(str(value) for value in public_client_ids)
                if set(manifest_mapping) != configured_roster:
                    raise ValueError(
                        "Partial-DP requires the virtual-client manifest source IDs to match "
                        "the fixed public roster exactly."
                    )
            client_partitions = build_virtual_client_partitions(
                frame, "train", manifest_path
            )
            print(f"Federation partition: virtual; manifest={manifest_path}")
        elif partition_mode == "institutional":
            client_partitions = build_client_partitions(
                frame,
                "train",
                max_clients=max_clients,
                public_client_ids=public_client_ids,
            )
            if max_clients is not None:
                print("WARNING: max_clients is set; only the selected source clients participate.")
            print("Federation partition: original source-client IDs")
        else:
            raise ValueError("federated.partition_mode must be 'institutional' or 'virtual'")
    else:
        client_partitions = {}
        print("Federated methods disabled. Skipping client partition construction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; clients: {len(client_partitions)}; features: {x.shape[1]}")
    input_dim = int(x.shape[1])
    hidden_dims = tuple(model_cfg.get("hidden_dims", [128, 64]))
    def model_factory():
        return AMLClassifier(input_dim, hidden_dims)

    rounds = int(fed_cfg["rounds"])
    local_steps = int(fed_cfg["local_steps_per_round"])
    total_steps = rounds * local_steps
    lr = float(fed_cfg["learning_rate"])
    clients_per_round = fed_cfg.get("clients_per_round")
    partial_participation_schedule = None
    partial_participation_schedule_seed = None
    partial_client_exposure = None
    partial_dp_accounting_ready = False
    if clients_per_round is not None and fed_cfg.get("participation_schedule_seed") is not None:
        partial_participation_schedule_seed = int(fed_cfg["participation_schedule_seed"])
        partial_participation_schedule = build_balanced_participation_schedule(
            client_partitions.keys(), rounds, int(clients_per_round),
            partial_participation_schedule_seed,
        )
    results = []
    typology_results = []
    privacy_reports = {}
    train_logs = []
    client_train_logs = []

    if need_dp:
        orders = [int(a) for a in privacy_cfg["alpha_orders"]]
        delta = float(privacy_cfg["delta"])
        q_focus = _q_map(privacy_cfg["q_focus"])
        q_bg = float(privacy_cfg["q_background"])
        q_uniform = float(privacy_cfg["q_uniform"])
        relevance = {g: float(privacy_cfg["relevance_scores"][g]) for g in POSITIVE_GROUPS}
        probs, counts, schedule = build_schedule(
            relevance,
            float(privacy_cfg["lambda_exploration"]),
            float(privacy_cfg["eta"]),
            total_steps,
        )
        schedule_df = pd.DataFrame({"step": np.arange(1, len(schedule) + 1), "focus_group": schedule})
        schedule_df.to_csv(output_dir / "typology_schedule.csv", index=False)
        _save_json(output_dir / "schedule_config.json", {
            "exposure_probabilities": probs,
            "global_focus_step_counts": counts,
            "schedule": schedule,
            "schedule_is_public_and_label_independent": True,
        })
        independent = validate_against_opacus(orders=orders)
        if privacy_cfg.get("require_opacus_validation", True) and independent["status"] != "PASS":
            raise RuntimeError(f"Independent RDP accountant validation failed: {independent}")
        formal_privacy_claim_permitted = bool(
            privacy_cfg.get("secure_randomness", True) and independent["status"] == "PASS"
        )

        if partial_participation_schedule is not None:
            # Per-client exposure counts are derived from the exact public
            # participation and focus schedules used by the trainer.
            partial_client_exposure = build_client_focus_exposure_counts(
                partial_participation_schedule,
                schedule,
                client_partitions.keys(),
                local_steps_per_round=local_steps,
            )
            _save_json(
                output_dir / "partial_dp_client_exposure_counts.json",
                partial_client_exposure,
            )
            privacy_reports, comparison, sigma_uniform, individualized_sigmas, policy_u, policy_d = (
                _build_partial_dp_privacy_reports(
                    privacy_cfg,
                    fed_cfg,
                    exp_cfg,
                    partial_client_exposure,
                    schedule,
                    total_steps,
                    local_steps,
                    independent,
                )
            )
            partial_dp_accounting_ready = True
            _save_json(output_dir / "privacy_comparison.json", comparison)
            for name, report in privacy_reports.items():
                _save_json(
                    output_dir / "privacy_reports" / f"{name}.json",
                    _strip_internal_rdp_arrays(report),
                )
            _save_json(
                output_dir / "partial_dp_accounting_assumptions.json",
                {
                    "privacy_unit": "one transaction record under add/remove adjacency",
                    "ownership": "each record is assigned to exactly one fixed public virtual client",
                    "client_participation_seed": partial_participation_schedule_seed,
                    "client_participation_is_public_and_data_independent": True,
                    "client_subsampling_amplification_credited": False,
                    "positive_record_accounting": "compose only focus steps at the owning client where the public focus schedule equals the record's fixed typology",
                    "background_record_accounting": "compose q_background Gaussian mechanisms at every active step of its owning client, using that step's focus-group noise multiplier",
                    "one_combined_noise_vector_for_focus_plus_background": True,
                    "fixed_normalization": float(privacy_cfg["normalization"]),
                    "composition_repetitions_per_method": len(exp_cfg.get("random_seeds", [int(config.get("seed", 2026))])),
                    "opacus_rdp_validation": independent,
                    "raw_training_loss_and_realized_sample_counts_logged": False,
                    "formal_privacy_claim_permitted": False,
                    "claim_readiness": "candidate mechanism/accounting; independent end-to-end review required before making a formal DP claim",
                    "claim_scope": "training-record mechanism only; validation/test records and any separately published raw dataset audit are outside this training DP guarantee",
                    "joint_release_warning": comparison.get("joint_release_warning"),
                },
            )
        else:
            orders = [int(a) for a in privacy_cfg["alpha_orders"]]
            delta = float(privacy_cfg["delta"])
            q_focus = _q_map(privacy_cfg["q_focus"])
            q_bg = float(privacy_cfg["q_background"])
            q_uniform = float(privacy_cfg["q_uniform"])
            relevance = {g: float(privacy_cfg["relevance_scores"][g]) for g in POSITIVE_GROUPS}
            probs, counts, schedule = build_schedule(
                relevance,
                float(privacy_cfg["lambda_exploration"]),
                float(privacy_cfg["eta"]),
                total_steps,
            )
            schedule_df = pd.DataFrame({"step": np.arange(1, len(schedule) + 1), "focus_group": schedule})
            schedule_df.to_csv(output_dir / "typology_schedule.csv", index=False)
            _save_json(output_dir / "schedule_config.json", {"exposure_probabilities": probs, "step_counts": counts, "schedule": schedule})
            independent = validate_against_opacus(orders=orders)
            if privacy_cfg.get("require_opacus_validation", True) and independent["status"] != "PASS":
                raise RuntimeError(f"Independent RDP accountant validation failed: {independent}")
            formal_privacy_claim_permitted = bool(
                privacy_cfg.get("secure_randomness", True) and independent["status"] == "PASS"
            )
            sigma_uniform, eps_uniform_actual = calibrate_uniform_noise(
                q_uniform, total_steps, float(privacy_cfg["epsilon_uniform"]), delta, orders
            )
            uniform_rdp = compute_rdp_poisson_gaussian(q_uniform, sigma_uniform, orders) * total_steps
            uniform_eps, uniform_order = rdp_to_epsilon(orders, uniform_rdp, delta)
            privacy_reports["uniform_dp"] = {
                "sigma": sigma_uniform,
                "global_epsilon": uniform_eps,
                "target_epsilon": float(privacy_cfg["epsilon_uniform"]),
                "delta": delta,
                "best_order": uniform_order,
                "steps_per_record": total_steps,
                "q": q_uniform,
                "opacus_validation": validate_against_opacus(orders=orders),
            }
            individualized_targets = {
                g: float(privacy_cfg["epsilon_min"]) +
                   (float(privacy_cfg["epsilon_max"]) - float(privacy_cfg["epsilon_min"])) * relevance[g]
                for g in POSITIVE_GROUPS
            }
            individualized_sigmas = {}
            individualized_epsilons = {}
            for group in POSITIVE_GROUPS:
                sig, actual = calibrate_uniform_noise(q_uniform, total_steps, individualized_targets[group], delta, orders)
                individualized_sigmas[group] = sig
                individualized_epsilons[group] = actual
            sigma_bg_ind, epsilon_bg_ind = calibrate_uniform_noise(
                q_uniform, total_steps, float(privacy_cfg["epsilon_min"]), delta, orders
            )
            individualized_sigmas["BG"] = sigma_bg_ind
            individualized_epsilons["BG"] = epsilon_bg_ind
            privacy_reports["individualized_dp"] = {
                "noise_multipliers": individualized_sigmas,
                "positive_groups": {g: {"epsilon": individualized_epsilons[g], "target": individualized_targets[g]} for g in POSITIVE_GROUPS},
                "background": {"epsilon": epsilon_bg_ind, "target": float(privacy_cfg["epsilon_min"])},
                "global_epsilon": max(individualized_epsilons.values()),
                "delta": delta,
                "q_uniform": q_uniform,
                "steps_per_record": total_steps,
                "opacus_validation": validate_against_opacus(orders=orders),
                "baseline_protocol": "same Poisson q for every record; per-record Gaussian noise multiplier by fixed stratum; no typology focus schedule",
                "not_exact_reproduction": "The paper names an individualized-DP-SGD-style comparator but does not specify a unique noise-placement/sampling algorithm; this is an explicit operationalization.",
            }
            common = {
                "delta": delta,
                "orders": orders,
                "require_opacus_validation": bool(privacy_cfg.get("require_opacus_validation", True)),
            }
            targets_u = {g: float(privacy_cfg["epsilon_uniform"]) for g in POSITIVE_GROUPS}
            policy_u = calibrate_typology_noise(counts, q_focus, q_bg, targets_u,
                                                float(privacy_cfg["epsilon_uniform"]), **common)
            policy_u["delta"] = delta
            privacy_reports["ta_policy_u"] = policy_u
            targets_d = {
                g: float(privacy_cfg["epsilon_min"]) +
                   (float(privacy_cfg["epsilon_max"]) - float(privacy_cfg["epsilon_min"])) * relevance[g]
                for g in POSITIVE_GROUPS
            }
            policy_d = calibrate_typology_noise(counts, q_focus, q_bg, targets_d,
                                                float(privacy_cfg["epsilon_min"]), **common)
            policy_d["delta"] = delta
            privacy_reports["ta_policy_d"] = policy_d
            for report in privacy_reports.values():
                report["independent_accountant_validation"] = independent
                report["formal_privacy_claim_permitted"] = formal_privacy_claim_permitted
            max_relevance = max(relevance.values())
            regime_a_setup = (
                abs(max_relevance - 1.0) <= 1e-8
                and abs(float(privacy_cfg["epsilon_max"]) - float(privacy_cfg["epsilon_uniform"])) <= 1e-8
            )
            tolerance = float(privacy_cfg.get("privacy_matching_tolerance", 0.02))
            epsilon_values = {
                "uniform_dp": uniform_eps,
                "individualized_dp": privacy_reports["individualized_dp"]["global_epsilon"],
                "ta_policy_u": policy_u["global_epsilon"],
                "ta_policy_d": policy_d["global_epsilon"],
            }
            matched_to_uniform = all(
                abs(value - uniform_eps) <= tolerance for value in epsilon_values.values()
            )
            comparison = {
                "regime_a_setup_satisfied": regime_a_setup,
                "privacy_matching_tolerance": tolerance,
                "global_epsilon_bounds": epsilon_values,
                "matched_maximum_privacy_cap": bool(regime_a_setup and matched_to_uniform),
                "interpretation": (
                    "matched-cap comparison" if regime_a_setup and matched_to_uniform
                    else "report as a privacy-utility frontier; do not claim strictly matched privacy"
                ),
            }
            _save_json(output_dir / "privacy_comparison.json", comparison)
            for name, report in privacy_reports.items():
                _save_json(output_dir / "privacy_reports" / f"{name}.json", report)

    for seed in exp_cfg.get("random_seeds", [int(config.get("seed", 2026))]):
        seed = int(seed)
        set_seed(seed)
        methods = []
        if exp_cfg.get("run_central_nn", True):
            model = train_central_nn(
                model_factory,
                x_train,
                y_train,
                int(exp_cfg.get("central_epochs", 5)),
                int(exp_cfg.get("central_batch_size", 1024)),
                float(exp_cfg.get("central_learning_rate", 0.001)),
                device,
                seed,
                float(exp_cfg.get("central_pos_weight", 1.0)),
            )
            methods.append(("Central-NN", model, None))
            torch.save(model.state_dict(), output_dir / "checkpoints" / f"central_nn_seed{seed}.pt")

        if exp_cfg.get("run_central_gbdt", True):
            model, used_rows = train_central_gbdt(x_train, y_train,
                                                  int(exp_cfg.get("gbdt_max_rows", 100000)),
                                                  int(exp_cfg.get("gbdt_max_iter", 50)), seed)
            methods.append(("Central-GBDT", model, None))
            train_logs.append({"method": "Central-GBDT", "seed": seed, "training_rows": used_rows})
            import joblib
            joblib.dump(model, output_dir / "checkpoints" / f"central_gbdt_seed{seed}.joblib")

        federated_specs = []
        if exp_cfg.get("run_fedavg", True):
            federated_specs.append(("FedAvg", "fedavg", {}))
        if exp_cfg.get("run_fedadam", False):
            federated_specs.append(("FedAdam", "fedadam", {}))
        if exp_cfg.get("run_uniform_dp", False):
            federated_specs.append(("Uniform-DP-FL", "uniform_dp", {
                "q_uniform": q_uniform, "sigma_uniform": sigma_uniform,
                "clip_norm": float(privacy_cfg["clip_norm"]),
                "normalization": float(privacy_cfg["normalization"]),
                "secure_randomness": bool(privacy_cfg.get("secure_randomness", True)),
            }))
        if exp_cfg.get("run_individualized_dp", False):
            federated_specs.append(("Individualized-DP-SGD-style", "individualized_dp", {
                "q_uniform": q_uniform,
                "sigma_by_group": individualized_sigmas,
                "clip_norm": float(privacy_cfg["clip_norm"]),
                "normalization": float(privacy_cfg["normalization"]),
                "secure_randomness": bool(privacy_cfg.get("secure_randomness", True)),
            }))
        if exp_cfg.get("run_ta_policy_u", False):
            federated_specs.append(("TA-DPFL-Policy-U", "ta_dpfl", {
                "schedule": schedule, "q_focus": q_focus, "q_background": q_bg,
                "sigmas": policy_u["noise_multipliers"],
                "clip_norm": float(privacy_cfg["clip_norm"]),
                "normalization": float(privacy_cfg["normalization"]),
                "secure_randomness": bool(privacy_cfg.get("secure_randomness", True)),
            }))
        if exp_cfg.get("run_ta_policy_d", False):
            federated_specs.append(("TA-DPFL-Policy-D", "ta_dpfl", {
                "schedule": schedule, "q_focus": q_focus, "q_background": q_bg,
                "sigmas": policy_d["noise_multipliers"],
                "clip_norm": float(privacy_cfg["clip_norm"]),
                "normalization": float(privacy_cfg["normalization"]),
                "secure_randomness": bool(privacy_cfg.get("secure_randomness", True)),
            }))

        for label, mode, extra in federated_specs:
            participation_schedule = None
            schedule_seed = None
            clients_per_round = fed_cfg.get("clients_per_round")
            if clients_per_round is not None:
                if partial_participation_schedule is not None:
                    participation_schedule = partial_participation_schedule
                    schedule_seed = partial_participation_schedule_seed
                else:
                    configured_schedule_seed = fed_cfg.get("participation_schedule_seed")
                    schedule_seed = seed if configured_schedule_seed is None else int(configured_schedule_seed)
                    participation_schedule = build_balanced_participation_schedule(
                        client_partitions.keys(),
                        rounds=rounds,
                        clients_per_round=int(clients_per_round),
                        seed=schedule_seed,
                    )
                schedule_rows = [
                    {"seed": seed, "round": round_number, "client_id": client_id,
                     "clients_per_round": int(clients_per_round),
                     "schedule_seed": schedule_seed}
                    for round_number, cohort in enumerate(participation_schedule, start=1)
                    for client_id in cohort
                ]
                pd.DataFrame(schedule_rows).to_csv(
                    output_dir / f"client_participation_schedule_seed{seed}.csv",
                    index=False,
                )
                _save_json(
                    output_dir / f"client_participation_schedule_seed{seed}.json",
                    {
                        "method": "balanced_random_permutation_blocks",
                        "seed": schedule_seed,
                        "rounds": rounds,
                        "eligible_clients": len(client_partitions),
                        "clients_per_round": int(clients_per_round),
                        "appearances_per_client": rounds * int(clients_per_round) // len(client_partitions),
                        "schedule_is_public_and_data_independent": True,
                        "privacy_status": (
                            "used_by_partial_record_level_accountant"
                            if mode in {"uniform_dp", "ta_dpfl", "individualized_dp"}
                            else "non_private_or_public_schedule_metadata"
                        ),
                    },
                )

            is_private_mode = mode in {"uniform_dp", "ta_dpfl", "individualized_dp"}
            if mode == "fedadam":
                server_optimizer = "fedadam"
            elif is_private_mode:
                server_optimizer = str(fed_cfg.get("dp_server_optimizer", "fedadam")).lower()
            else:
                server_optimizer = "fedavg"

            model, history = run_federated(
                model_factory, x, y_all, client_partitions, mode,
                rounds, local_steps, lr, device, seed,
                batch_size_nonprivate=int(fed_cfg.get("batch_size_nonprivate", 256)),
                aggregation_mode=str(fed_cfg.get("aggregation_mode", "uniform")),
                nonprivate_optimizer=str(fed_cfg.get("nonprivate_optimizer", "sgd")),
                nonprivate_weight_decay=float(fed_cfg.get("nonprivate_weight_decay", 1e-4)),
                nonprivate_max_grad_norm=float(fed_cfg.get("nonprivate_max_grad_norm", 5.0)),
                participation_schedule=participation_schedule,
                server_learning_rate=float(fed_cfg.get("server_learning_rate", 0.001)),
                server_beta1=float(fed_cfg.get("server_beta1", 0.9)),
                server_beta2=float(fed_cfg.get("server_beta2", 0.99)),
                server_tau=float(fed_cfg.get("server_tau", 1e-8)),
                server_optimizer=server_optimizer,
                partial_dp_accounting_enabled=bool(
                    is_private_mode
                    and participation_schedule is not None
                    and partial_dp_accounting_ready
                ),
                **extra,
            )
            if label == "Uniform-DP-FL":
                method_privacy = privacy_reports.get("uniform_dp")
            elif label == "Individualized-DP-SGD-style":
                method_privacy = privacy_reports.get("individualized_dp")
            elif "TA-DPFL" in label:
                method_privacy = privacy_reports.get("ta_policy_u" if label.endswith("U") else "ta_policy_d")
            else:
                method_privacy = None
            methods.append((label, model, method_privacy))
            for entry in history:
                summary = {k: v for k, v in entry.items() if k != "client_diagnostics"}
                train_logs.append({**summary, "method": label, "seed": seed})
                for client_entry in entry.get("client_diagnostics", []):
                    client_train_logs.append({**client_entry, "round": entry["round"], "method": label, "seed": seed})
            torch.save(model.state_dict(), output_dir / "checkpoints" / f"{label.lower().replace('-', '_')}_seed{seed}.pt")

        for method, model, privacy_report in methods:
            overall, per_typology = _record_results(
                method, seed, model, x_val, y_val, groups_val,
                x_test, y_test, groups_test,
                float(exp_cfg.get("validation_fpr_limit", 0.01)), device,
                output_dir, privacy_report,
            )
            results.append(overall)
            typology_results.extend(per_typology)

    pd.DataFrame(results).to_csv(output_dir / "overall_results.csv", index=False)
    pd.DataFrame(typology_results).to_csv(output_dir / "typology_results.csv", index=False)
    pd.DataFrame(train_logs).to_csv(output_dir / "training_log.csv", index=False)
    if client_train_logs:
        pd.DataFrame(client_train_logs).to_csv(output_dir / "client_training_log.csv", index=False)
    plot_paths = generate_result_plots(output_dir)
    print(f"Completed. Results saved under {output_dir.resolve()}")
    if plot_paths:
        print("Generated plots:")
        print("\n".join(plot_paths))
    if privacy_reports:
        print("Worst-case reported privacy bounds:")
        print(json.dumps({k: v.get("global_epsilon") for k, v in privacy_reports.items()}, indent=2))


if __name__ == "__main__":
    main()
