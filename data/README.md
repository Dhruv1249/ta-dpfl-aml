# Dataset and typology annotation requirements

1. Obtain the IBM AML-Data HI-Small transaction CSV under its applicable
   licence/terms and place it at the path in `configs/experiment.yaml`.
2. The pipeline recognizes the standard HI-Small columns, including
   `Timestamp`, `From Bank`, `Amount Received`, `Receiving Currency`,
   `Amount Paid`, `Payment Currency`, `Payment Format`, and `Is Laundering`.
   If names differ, change the column mapping in the YAML file.
3. **Typology annotations are required for the paper's main claim.** If the
   raw CSV has a typology column, set `data.columns.typology` to its exact name.
   Otherwise provide `data/typology_annotations.csv` with columns:

       source_row_id,typology
       0,CYCLE
       1,FAN-OUT

   `source_row_id` is the zero-based row position in the unchanged raw CSV.
   Preserve the original file and row order. Do not regenerate row IDs after
   sorting or filtering. Each annotation row must be unique. Valid typologies
   are CYCLE, GATHER-SCATTER, BIPARTITE, FAN-OUT, SCATTER-GATHER, STACK,
   RANDOM, and FAN-IN. Missing typology for a positive is assigned UNCL.
   Unknown non-empty labels cause an error rather than being silently mapped.
4. The original HI-Small release may not contain the typology labels assumed by
   the paper. Do not infer them from account graphs and then describe them as
   benchmark-supplied annotations. If you do not have the separate annotation
   source, the intended typology-aware experiment cannot be reproduced as
   specified. `allow_all_uncl: true` is only a pipeline diagnostic; it does
   not validate the paper's typology hypothesis.
5. The code assigns each transaction to one simulated client using the
   configured client/bank column. This implements the paper's single-owner
   simulation assumption, not a real-world bank data-governance arrangement.

6. Run `python run_experiment.py --audit-only` after preparing the raw data and
   annotations. Review `outputs/data_audit.json` and
   `outputs/public_client_ids_candidate.json`. Copy the two UTC timestamp
   cutoffs into `data.train_end_timestamp` and `data.validation_end_timestamp`
   in the YAML configuration. Copy the approved, fixed simulated institution
   roster into `data.public_client_ids` as a YAML list of strings. The roster
   is treated as public and fixed for the record-level neighbouring-dataset
   analysis. Do not derive these values afresh for each privacy comparison.
7. The timestamp cutoffs are strict boundaries: train is before the first
   cutoff, validation is from the first cutoff up to (but not including) the
   second, and test is at or after the second. Equal timestamps stay in the
   same split, so realized percentages can differ slightly from 60/20/20.
