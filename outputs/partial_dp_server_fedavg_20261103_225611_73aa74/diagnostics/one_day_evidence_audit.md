# TA-DPFL one-day evidence audit

- Run directory: `D:\Study\Academics\Research and Others\Capstone_2027\TA-DPFL_RTX5050_Windows11_Optimized\TA-DPFL_bundle\outputs\partial_dp_server_fedavg_20261103_225611_73aa74`
- Config: `D:\Study\Academics\Research and Others\Capstone_2027\TA-DPFL_RTX5050_Windows11_Optimized\TA-DPFL_bundle\configs\partial_dp_server_fedavg_20261103_225611_73aa74.yaml`

## Execution and integrity checks

- **all_overall_ap_at_or_above_prevalence**: `False`
- **all_raw_auc_at_or_above_half**: `False`
- **single_seed_only**: `True`
- **typology_rows_all_auc_below_half**: `True`
- **typology_rows_all_ap_below_stratum_prevalence**: `True`
- **all_methods_have_expected_rounds**: `True`
- **all_rounds_have_expected_client_count**: `True`
- **server_optimizer_matches_config**: `True`
- **raw_private_losses_and_sample_counts_not_logged**: `True`
- **focus_schedule_length_matches_rounds**: `True`
- **focus_count_sum_matches_rounds**: `True`
- **focus_schedule_is_public_label_independent**: `True`
- **all_clients_exactly_expected_appearances**: `True`
- **per_method_privacy_caps_matched**: `True`
- **opacus_rdp_crosscheck_passed**: `True`
- **formal_dp_certification_available**: `False`

## Overall test metrics

| Method | AP | AP / prevalence | ROC-AUC | Precision | Recall | ε |
|---|---:|---:|---:|---:|---:|---:|
| Uniform-DP-FL | 0.0011089138 | 0.6265 | 0.275229 | 0.000260485 | 0.00111235 | 4.0000000000000036 |
| TA-DPFL-Policy-U | 0.0011089409 | 0.6266 | 0.275244 | 0.000259168 | 0.00111235 | 4.0 |

## Approximate DP-noise-only update scale

| Method | Approx. noise-only update L2 | Observed aggregate update L2 | Observed / noise-only |
|---|---:|---:|---:|
| Uniform-DP-FL | 0.000302985 | 0.00031879 | 1.052 |
| TA-DPFL-Policy-U | 0.000503327 | 0.000513683 | 1.021 |

This is a scale comparison under the listed assumptions, not a formal signal-to-noise estimate or proof that all observed updates are noise.

## Privacy interpretation

The RDP numerical cross-check is accountant validation, not independent end-to-end DP certification. Report transaction-level add/remove scope and composition across jointly released models.

## Detection interpretation

Ranking is below chance for at least one method; do not claim successful AML detection.

## Typology-aware utility interpretation

No demonstrated overall AP lift over prevalence in this run.

## Limitations

- 256 hashed virtual clients are a simulation, not 256 independently participating institutions.
- Single-seed comparisons are descriptive, not evidence of statistical significance.
- Validation/test records and separately released raw dataset audits are outside the training-record DP claim.
- Do not invert scores, alter thresholds, or tune on the test set based on these diagnostics.

Machine-generated audit. Review source files and confirm interpretations before submission.