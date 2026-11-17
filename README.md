# TA-DPFL: Typology-Aware Differentially Private Federated Learning for Anti-Money Laundering

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)](tests/)

Official implementation of **TA-DPFL**: *Typology-Aware Privacy Scheduling for Differentially Private Federated AML Transaction Detection*.

---

## Overview & Core Breakthroughs

Cross-institutional financial crime detection operates under extreme class imbalance (prevalence $< 0.1\%$), rare multi-hop laundering topologies (`CYCLE`, `GATHER-SCATTER`, `SCATTER-GATHER`), and strict regulatory privacy requirements.

This project delivers a complete, mathematically verified federated learning framework under **Differential Privacy ($\varepsilon = 4.0, \delta = 10^{-8}$)** that achieves:

1. **Resolution of the Catastrophic Gradient Collapse Paradox:**
   * *The Problem:* Standard unweighted FedAvg and DP-SGD catastrophically collapse into an inverted score distribution ($\text{ROC-AUC} \approx 0.275$), while server FedAdam achieves state-of-the-art performance ($\text{ROC-AUC} = 0.938$).
   * *The Root Cause:* Under $0.075\%$ positive prevalence, default zero-bias initialization ($b_0 = 0$) creates a $1,332:1$ negative gradient dominance that forces unweighted gradient descent to suppress positive features, predicting lower output probabilities for large laundering amounts than for benign transactions.
   * *The Fix:* Initializing the output layer bias to the public class-prior log-odds:
     $$b_0 = \log\left(\frac{\pi}{1 - \pi}\right) \approx -7.19 \quad (\text{zero DP cost})$$
     combined with server-side FedAdam (via the formal **DP Post-Processing Property**) restores exact gradient equilibrium and natively lifts the private model to **$\text{ROC-AUC} = \mathbf{0.8955}$** on the full 1,015,882-record test split without score inversion.

2. **Resolution of Signal Dilution via Typology Scheduling (Policy U):**
   * Classic Individualized-DP-SGD uses uniform Poisson sampling ($q = 0.001$). Under extreme imbalance, over $99.5\%$ of client batches contain **zero positive samples**, allowing isotropic Gaussian noise to wash out fragile minority gradients ($\text{ROC} = 0.8147$).
   * TA-DPFL **Policy U** concentrates exposure into scheduled focus rounds ($q_{\text{focus}} = 0.05$), delivering high Signal-to-Noise Ratio (SNR) updates for rare structures and achieving **$\text{ROC-AUC} = \mathbf{0.8955}$** (with `CYCLE` detection at **$0.9353$**).

3. **Empirical Validation of the Background Privacy Tax (Policy D):**
   * Enforcing tighter privacy on the non-illicit background class ($\varepsilon_{\text{BG}} = 2.0$) forces the global noise multiplier to $\sigma = 1.15$. Because background transactions participate across every round, this higher noise floor degrades every client parameter delta, dropping overall $\text{ROC-AUC}$ from **$0.8955 \to \mathbf{0.8873}$**.

---

## Benchmark Results

Evaluated on the frozen, held-out test split of the IBM AML benchmark (**1,015,882 records, 1,798 positives, prevalence = 0.001770**). All metrics are directly reproducible via `tools/verify_all_models.py`.

### Master Model Comparison

| Method / Architecture | Privacy ($\varepsilon$) | Server Optimizer | ROC-AUC | Average Precision (AP) | AP / Prevalence | Mean Positive Logit | Mean Negative Logit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Central-NN (Class-Weighted)** | Non-private | AdamW | **0.9418** | **0.0741** | 41.89× | 0.3207 | 0.0235 |
| **Central-NN (Unweighted)** | Non-private | AdamW | **0.9391** | **0.0828** | 46.77× | 0.0262 | 0.0015 |
| **FedAdam (Non-private)** | Non-private | FedAdam | **0.9380** | **0.0859** | 48.52× | 0.0194 | 0.0008 |
| *FedAvg (Collapsed Baseline)* | Non-private | FedAvg | *0.2796* | *0.0011* | 0.63× | 0.0031 | 0.0163 |
| *Uniform-DP-FL (Collapsed)* | $\varepsilon = 4.0$ | FedAvg | *0.2752* | *0.0011* | 0.63× | 0.1723 | 0.2283 |
| *TA-DPFL Policy U (Collapsed)* | $\varepsilon = 4.0$ | FedAvg | *0.2752* | *0.0011* | 0.63× | 0.1720 | 0.2279 |
| **Uniform-DP-FL (Fixed)** | $\varepsilon = 4.0$ | FedAdam + Prior | **0.8319** | **0.0058** | 3.28× | 0.0045 | 0.0018 |
| **Individualized-DP-SGD (Fixed)** | $\varepsilon \in [2, 4]$ | FedAdam + Prior | **0.8147** | **0.0151** | 8.53× | 0.0024 | 0.0005 |
| **TA-DPFL Policy U (Fixed)** | $\varepsilon = 4.0$ | FedAdam + Prior | **0.8955** | **0.0145** | 8.19× | 0.0093 | 0.0019 |
| **TA-DPFL Policy D (Fixed)** | $\varepsilon \in [2, 4]$ | FedAdam + Prior | **0.8873** | **0.0122** | 6.87× | 0.0071 | 0.0020 |

---

### 4-Way Differentially Private Typology Breakdown (One-vs-Background Test ROC-AUC / AP)

| Typology Group | Test Positives | Uniform-DP ($\varepsilon=4.0$)<br>AUC / AP | Individualized-DP ($\varepsilon \in [2,4]$)<br>AUC / AP | TA-DPFL Policy U ($\varepsilon=4.0$)<br>AUC / AP | TA-DPFL Policy D ($\varepsilon \in [2,4]$)<br>AUC / AP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`CYCLE`** | 109 | 0.8634 / 0.000385 | 0.8759 / 0.002774 | **0.9353** / 0.000973 | 0.9345 / 0.000952 |
| **`GATHER-SCATTER`**| 401 | 0.8744 / 0.001575 | 0.8318 / 0.003476 | **0.9414** / **0.004192** | 0.9365 / 0.003501 |
| **`BIPARTITE`** | 74 | 0.8653 / 0.000291 | 0.8793 / **0.001241** | **0.9249** / 0.000640 | 0.9188 / 0.000615 |
| **`FAN-OUT`** | 140 | 0.8644 / 0.000518 | 0.8791 / **0.003416** | **0.9322** / 0.001311 | 0.9160 / 0.001035 |
| **`SCATTER-GATHER`**| 252 | 0.8775 / 0.001001 | 0.8392 / 0.003434 | **0.9512** / **0.003631** | 0.9281 / 0.002257 |
| **`STACK`** | 140 | 0.8591 / 0.000504 | 0.8521 / 0.001324 | 0.9279 / **0.001360** | **0.9284** / 0.001319 |
| **`RANDOM`** | 86 | 0.8708 / 0.000345 | 0.8680 / **0.001568** | **0.9391** / 0.000831 | 0.9265 / 0.000937 |
| **`FAN-IN`** | 137 | 0.8691 / 0.000544 | 0.8560 / 0.000980 | 0.9198 / 0.001027 | **0.9312** / **0.001034** |
| **`UNCL`** | 459 | 0.7200 / 0.000936 | 0.7080 / 0.001571 | **0.7740** / **0.001905** | 0.7638 / 0.001563 |
| **OVERALL TEST** | **1,798** | **0.8319 / 0.005801** | **0.8147 / 0.015094** | **0.8955 / 0.014488** | **0.8873 / 0.012165** |

---

## Dataset Acquisition & Setup

The experiments use the synthetic **IBM Transactions for Anti-Money Laundering (AML)** benchmark dataset (`HI-Small`):

1. **Download the Dataset:**
   * Download `HI-Small_Trans.csv` (approx. 454 MB, 5,078,345 transactions) from [Kaggle: IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) or the official IBM Research repository.
2. **Place in the Project Directory:**
   ```bash
   mkdir -p data
   mv /path/to/downloaded/HI-Small_Trans.csv data/HI-Small_Trans.csv
   ```
3. **Typology Pattern File:**
   * Ensure `data/HI-Small_Patterns.txt` is present (included in the repository).
   * The typology mapping script will link transaction records to the 8 laundering typologies + unclassified positives (`UNCL`):
     ```bash
     python tools/build_typology_annotations.py
     ```
   * This generates `data/typology_annotations.csv`.

### Chronological Splits (Frozen to Prevent Lookahead)
* **Train split:** `timestamp < 2022-09-06T13:36:00 UTC` (3,046,861 records, 2,297 positives, $0.0753\%$ prevalence).
* **Validation split:** `2022-09-06T13:36:00 <= timestamp < 2022-09-08T16:12:00 UTC` (1,015,602 records, 1,082 positives).
* **Test split:** `timestamp >= 2022-09-08T16:12:00 UTC` (1,015,882 records, 1,798 positives, $0.1770\%$ prevalence).

---

## Repository Structure

```
FLDP/
├── configs/
│   ├── experiment.yaml          # Single master configuration (splits, roster, hyperparameters)
│   └── privacy_design.json      # Formal DP specifications, target eps, delta, typology weights
├── data/
│   ├── preprocessing.py         # 53-dim feature matrix extraction and chronological splitter
│   ├── HI-Small_Patterns.txt    # IBM laundering pattern annotations
│   └── typology_annotations.csv # Joined typology mapping per transaction
├── federated/
│   ├── training.py              # Federated training loop (FedAvg and server-side FedAdam)
│   ├── partition.py             # Client partitioner
│   └── virtual_partition.py     # BLAKE2b hash partitioner (30,470 source banks -> 256 virtual clients)
├── privacy/
│   ├── dp_sgd.py                # Per-sample autograd clipping and Gaussian mechanism
│   ├── rdp_accountant.py        # Poisson-subsampled Gaussian RDP accountant (Opacus-verified)
│   ├── scheduler.py             # Largest-remainder typology exposure scheduler
│   └── calibrate.py             # SLSQP noise calibration engine for Policy U and Policy D
├── models/
│   └── aml_classifier.py        # 53 -> 128 -> 64 -> 1 MLP with class-prior bias initialization
├── tools/
│   ├── verify_all_models.py     # Unified benchmark suite (evaluates baselines and all 4 DP policies)
│   ├── run_fixed_dp_model.py    # Standalone training script for fixed DP-FL model
│   ├── audit_dp_gradient_direction.py # Deterministic noise-free gradient sanity auditor
│   └── calibrate_validation_scores.py # Monotonic Platt scaling validation calibration
├── outputs/
│   └── final_benchmark/
│       ├── benchmark_summary.json # Machine-readable test split evaluation summary
│       └── checkpoints/         # Model checkpoints (uniform_dp, individualized_dp, policy_u, policy_d)
└── tests/                       # Full unit test suite (14 passing tests)
```

---

## Installation & Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Dhruv1249/ta-dpfl-aml.git
cd ta-dpfl-aml

# Create and activate virtual environment (Python 3.10 or 3.11 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Unit Test Suite

The test suite validates RDP accounting against Opacus to within $10^{-13}$, verifies per-example gradient clipping, hash-based client assignments, and typology schedulers:

```bash
pytest tests/ -v
```

### 3. Run the Unified Benchmark Verification Suite

To evaluate all baselines and train/verify all 4 Differentially Private policies across the 1,015,882-sample test split:

```bash
python tools/verify_all_models.py
```

This updates `outputs/final_benchmark/benchmark_summary.json` with exact ROC-AUC, Average Precision, and typology breakdown metrics.

### 4. Train the Fixed DP-FL Master Model Directly

```bash
python tools/run_fixed_dp_model.py
```

Checkpoints are saved to `outputs/fixed_dp_model.pt`.

---

## Privacy Model & Guarantees

* **Privacy Unit:** Single transaction record under standard **add/remove adjacency**.
* **Accounting Framework:** Poisson-subsampled Gaussian mechanism under Rényi Differential Privacy (RDP), converted to $(\varepsilon, \delta)$-DP at order $\alpha \in [2, 128]$ with $\delta = 10^{-8}$.
* **Independent Verification:** The accountant is independently checked against `opacus.accountants.RDPAccountant` at test time (max absolute difference $< 1.42 \times 10^{-14}$).
* **Scope Boundary:** Transaction-level DP protects against the reconstruction of individual transaction features. It does not provide node-level, edge-level, whole-graph, or multi-hop laundering-attempt privacy. Graph message passing and account identifiers are strictly excluded from classifier inputs.

---

For detailed mathematical derivations, diagnostic logs, and evidence audits, see [`FINAL_PROJECT_EVIDENCE_AND_FINDINGS_REPORT.md`](FINAL_PROJECT_EVIDENCE_AND_FINDINGS_REPORT.md).
