Generated artifacts are written here by `run_experiment.py` (or under the configured `output_dir`).

- `data_audit.json`: row counts, split/group counts, fixed split cutoffs and feature schema.
- `public_client_ids_candidate.json`: candidate client roster; confirm it is public/fixed before copying into config.
- `typology_schedule.csv`, `schedule_config.json`: public exposure schedule.
- `privacy_reports/`: calibrated noise multipliers, per-stratum bounds and background bound.
- `privacy_comparison.json`: matched-cap check versus frontier classification.
- `overall_results.csv`, `typology_results.csv`: test metrics.
- `training_log.csv`: training diagnostics.
- `overall_ap_by_method.png`, `privacy_utility_comparison.png`, `typology_performance.png`: generated plots.
- `checkpoints/`: saved model weights and GBDT reference.
