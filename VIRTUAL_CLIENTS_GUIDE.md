# TA-DPFL virtual-client construction and audit

## Scope

This package adds a **data-independent source-client grouping utility**. It does not modify the training loop, privacy accountant, sampling rates, clipping, Gaussian noise, model, or existing real-client partitioner.

The mapping assigns each public source-client identifier (`From Bank`) to one virtual client using BLAKE2b hashing of the identifier and a fixed seed. The assignment does **not** use transaction rows, labels, typologies, or per-client record counts.

This is intentional. A row-count-balanced greedy assignment would use data-dependent client sizes. Unless those sizes are explicitly established as public metadata and included in the privacy assumptions, selecting the mapping from transaction counts creates an additional data-dependent preprocessing step. The hash mapping avoids that issue, but can yield uneven transaction loads when some source clients have much larger datasets. The audit measures this; do not call the buckets balanced unless the measured distribution supports that description.

## Install

Extract this ZIP into the project root so the `TA-DPFL_Virtual_Client_Tools` folder is beside `run_experiment.py`. Then run:

```powershell
.\TA-DPFL_Virtual_Client_Tools\install_virtual_client_tools.ps1
```

The installer copies the two utilities, the partition helper, its test, and this guide. It does not modify `run_experiment.py`, the existing `federated/partition.py`, the training loop, or any privacy code.

Run commands from the project root, using the full resolved config that contains the frozen timestamp cutoffs and complete 30,470-ID roster.

## Build the mapping

```powershell
.\.venv\Scripts\python.exe .\tools\build_virtual_clients.py `
  --config .\configs\experiment.resolved.yaml `
  --num-virtual-clients 256 `
  --seed 2026 `
  --output-dir .\outputs\virtual_federation
```

Expected outputs:

- `outputs/virtual_federation/virtual_client_manifest.csv`
- `outputs/virtual_federation/virtual_client_manifest_metadata.json`

The CSV has one row per source-client ID, not one row per transaction. The same source-client ID always maps to the same virtual-client ID for the same seed and number of virtual clients.

## Audit the mapping against the dataset

```powershell
.\.venv\Scripts\python.exe .\tools\audit_virtual_clients.py `
  --config .\configs\experiment.resolved.yaml `
  --manifest .\outputs\virtual_federation\virtual_client_manifest.csv `
  --metadata .\outputs\virtual_federation\virtual_client_manifest_metadata.json `
  --output .\outputs\virtual_federation\virtual_client_audit.json
```

The audit reloads the dataset through the project's `load_and_label_dataset()` function. It verifies that every observed source ID has exactly one mapping and that all processed transaction rows are represented exactly once. It reports training/validation/test counts, train positives and typology counts, virtual-client workload distribution, zero-data buckets, and the largest original source-client share within any bucket.

A `PASS` indicates assignment coverage and row preservation. It does **not** certify workload balance, useful statistical heterogeneity, or equivalence to 30,470 separately participating banks.

## Important interpretation

- The 256 virtual clients are an engineering simulation of grouped source IDs, not 256 real institutions.
- The grouping changes the client partition and therefore the simulated federation. It must be reported as such.
- The fixed roster and mapping seed must remain unchanged across methods and seeds in a matched comparison. If testing multiple mappings, treat mapping choice as an explicit experimental factor and report it.
- Do not change `max_clients` to 256 while still using the original source IDs. That would select only 256 IDs and drop the others. Use the manifest-based helper instead.
- Do not launch DP experiments merely because the mapping audit passes. First inspect workload concentration, verify local-step semantics and privacy accounting under the virtual-client federation, then run a small non-private smoke test.
- The current training loop performs every local step on every client in every round and uniformly averages client deltas. With 256 clients, 5 rounds and 10 local steps, this is 12,800 client-local optimizer-step calls per method per seed. It is substantially smaller than 1,523,500 calls for 30,470 clients, but still needs a pilot and runtime measurement.

## Integration helper

`federated/virtual_partition.py` exposes:

```python
from federated.virtual_partition import build_virtual_client_partitions

client_partitions = build_virtual_client_partitions(
    frame,
    split="train",
    manifest_path="outputs/virtual_federation/virtual_client_manifest.csv",
)
```

It returns the same `{client_id: {group: global_row_indices}}` interface used by `run_federated()`. It only constructs a temporary partition frame and preserves the original global row indices. It is not yet wired into `run_experiment.py`; integrate only after reviewing the audit report.
