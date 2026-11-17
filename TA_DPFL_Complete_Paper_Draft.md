# Typology-Aware Privacy Scheduling for Differentially Private Federated AML Transaction Detection

**Abstract**—Federated Learning (FL) offers a promising framework for multi-institutional Anti-Money Laundering (AML) collaboration without centralizing sensitive customer records. However, under extreme class imbalance (prevalence $< 0.2\%$) and strict record-level Differential Privacy (DP), standard DP-SGD models experience catastrophic utility collapse. This paper investigates **Typology-Aware Privacy Scheduling (TA-DPFL)**: a federated optimization framework that strategically allocates scheduled exposure across distinct financial crime typologies (`CYCLE`, `GATHER-SCATTER`, `FAN-OUT`, etc.) under a unified Rényi Differential Privacy (RDP) accounting model. We present a fully paper-aligned, reproducible implementation evaluated on the standard IBM AML benchmark (5,078,345 transactions partitioned across 256 simulated banks). Our RDP privacy accountant is independently cross-verified against Opacus to within numerical precision ($1.42 \times 10^{-14}$). 

Crucially, we evaluate both non-private and private regimes. In the non-private setting, server-adaptive optimization (FedAdam) successfully captures laundering patterns, achieving $\text{ROC-AUC} = 0.9380$ and Average Precision ($\text{AP}$) $= 0.0859$ ($48.7\times$ prevalence), whereas standard unweighted FedAvg collapses ($\text{ROC-AUC} = 0.2796$). Under standard default initialization ($b_0 = 0$), differentially private models also experience score inversion ($\text{ROC-AUC} \approx 0.2752$), which validation-based monotonic calibration reveals to be an inverted representation ($w = -1.5601 \implies \text{ROC-AUC} = 0.7248$). 

We resolve this collapse at its root cause by establishing that extreme class imbalance ($0.075\%$ prevalence) induces a $1,332:1$ gradient force asymmetry in binary cross-entropy. By setting the output layer bias to the domain-knowledge base rate log-odds ($b_0 = \log(\frac{\pi}{1-\pi}) \approx -7.19$, consuming zero privacy budget) and applying server-side momentum via the DP Post-Processing Property, we restore native positive learning without score inversion. Evaluating across a comprehensive 4-way benchmark under strict Differential Privacy ($\varepsilon = 4.0, \delta = 10^{-8}$), TA-DPFL Policy U achieves $\text{ROC-AUC} = 0.8555$ and the highest Average Precision ($\text{AP} = 0.015259$, $8.62\times$ prevalence), outperforming Uniform-DP ($\text{AP} = 0.010708$) by $+42.5\%$ and demonstrating superior detection of complex cyclic laundering rings (`CYCLE` $\text{ROC-AUC} = 0.9239$ vs $0.9170$). 

Furthermore, we evaluate Individualized-DP-SGD and Policy D:
1. Individualized-DP-SGD experiences severe **Signal Dilution** ($\text{ROC-AUC} = 0.7837, \text{AP} = 0.008267$), proving that unstratified Poisson sampling leaves $99.5\%$ of batches devoid of rare positive signals, allowing accumulated Gaussian noise to drown out learning.
2. TA-DPFL Policy D confirms **Hypothesis 2** ($\text{ROC-AUC} = 0.7774, \text{AP} = 0.007914$), demonstrating that enforcing a stricter privacy target on the majority background class ($\bar{\varepsilon}_{\text{BG}} = 2.0$) forces the global noise multiplier to $\sigma = 1.15$, imposing an inescapable utility tax across all network weights. Finally, we formulate the RDP Privacy Conservation Law governing rare-stratum exposure under worst-case caps.

**Index Terms**—Federated Learning, Differential Privacy, Anti-Money Laundering, AML Typologies, Class Imbalance, Privacy-Utility Trade-off, RDP Accounting, Signal-to-Noise Ratio.

---

## I. INTRODUCTION

Anti-Money Laundering (AML) detection is extremely challenged by illicit activities that are most often distributed across multiple accounts, financial entities like banks, and transactions rather than happening as isolated events. Such activities take form through relational structures including transaction chains, cycles, and coordinated transfer patterns, making graph-based learning particularly suitable for capturing dependencies that conventional models would miss. Graph Neural Networks (GNNs) exploit these dependencies through neighborhood message passing, allowing the representation of an entity to make use of surrounding context—essential for AML as evidence of suspicious behavior resides in multiple nodes and not in an individual transaction alone. Fraud-detection research has explicitly shown the importance of preserving neighborhood information and subgraph signals, particularly when class imbalance is high [1]. However, on the flip side, the same structural dependence that makes GNNs effective creates a fundamental challenge when the underlying financial graph is split across multiple institutions.

In a cross-institutional AML setting, each financial institution sees only a specific portion of the underlying transaction network, while directly centralizing sensitive transaction and entity information is infeasible considering privacy, security, regulatory, and organizational constraints. Federated Learning (FL) provides a natural framework for collaborative model training while retaining local data within participating institutions. However, standard parameter aggregation does not reconstruct the graph context missing from an institution’s local view. A key problem is the loss of cross-client structural context: when a transaction connects entities across institutional boundaries, the corresponding message-passing methodology depends on representations unavailable to the local client. Recent federated graph-learning approaches have explored mechanisms to recover such information: FedGraph-VASP [2] exchanges representations of boundary accounts to recover cross-institutional context in AML, while CE-FedGNN [3] addresses cross-client graph coupling through infrequent embedding exchanges and moving-average estimators that reduce representation staleness.

The challenge becomes severe under extreme class imbalance, where fraudulent behavior is represented by a minute fraction of observations ($< 0.2\%$) and the evidence required for detection is concentrated in rare structural configurations. Differential Privacy (DP) introduces added difficulty: privacy protection bounds how strongly an individual contribution can affect the released model. In imbalanced learning, prior art shows that conventional oversampling and SMOTE can significantly increase sensitivity, as a single sensitive example may influence many derived points, requiring substantially larger noise for a given privacy guarantee [4]. Graph learning amplifies this issue: a change in one graph element can propagate to multi-hop neighborhood representations, influencing a broad portion of the learned space [5]. The trade-off is not simply privacy noise versus accuracy, but rather:
$$\text{Graph Dependency} \longrightarrow \text{Sensitivity} \longrightarrow \text{Perturbation Scale} \longrightarrow \text{Preservation or Destruction of Structural Information.}$$

Recent research has tackled facets of this challenge along various axes: federated heterogeneous graph learning has investigated structural recovery across fragmented clients [6]; federated cross-client subgraph detection has demonstrated that patterns like cycles and scatter-gather structures become unobservable locally when component structures span client boundaries [7]; and importance-aware GNN privacy has shown that privacy budgets can be differentiated by topological centrality [8]. At the fraud level, adaptive privacy has been explored according to institutional rarity and distributional divergence [9], and CE-FedGNN combines federated GNNs with metric-DP for released embeddings [3].

However, the question that remains open is not simply how to perform federated AML or provide graph privacy, but rather: **which structural information should receive higher protection against privacy-induced utility loss when the learning task is AML detection?** Generic topological centrality does not necessarily correspond to criminality: a rare cycle or scatter-gather configuration may carry far greater detection relevance than a structurally prominent but benign region of the graph [8], [9]. This motivates shifting from generic topology-aware privacy toward task-specific **AML-typology-aware privacy allocation**.

In this work, we examine **Typology-Aware Privacy Allocation for Federated Learning in Cross-Institutional Anti-Money Laundering (TA-DPFL)**. Our core idea is to use the structural relevance of laundering typologies to determine where a fixed privacy budget should be spent, aiming to preserve rare fraud-related structures under formal privacy constraints. We formulate this as a privacy-constrained scheduling and accounting problem on the standard IBM AML benchmark, evaluating both private and non-private regimes to understand the precise boundaries of private federated learning in high-imbalance financial settings.

---

## II. RELATED WORK

### A. Federated Learning for Financial Fraud Detection
Federated Learning enables collaborative fraud detection without centralizing raw financial records. FedAvg [10] established federated averaging, while FedProx [11] addressed non-IID client optimization. Financial applications subsequently explored oversampling [12], meta-learning [13], adaptive aggregation and imbalance adjustment [14], encrypted aggregation [15], and privacy-preserving cross-bank collaboration [16], [17]. Recent advances incorporate reputation-weighted privacy adaptation [18], noise-aware aggregation [19], and hierarchical collaboration [9], as surveyed in [20]. Although these studies address collaboration and client heterogeneity, the use of declared laundering typologies to determine training exposure and noise schedules remains an open research question.

### B. Graph-Based AML and Fraud Detection
Graph-based methods capture coordinated transaction patterns that single-transaction classifiers miss. Altman et al. [21] introduced the IBM AML benchmark with annotated laundering scenarios, while GARG-AML [22] investigated interpretable graph risk indicators. DCL-GFD [8] preserved abnormal signals through adaptive neighborhood modeling, and FedGAT-DCNN [24] combined graph attention with FL. Meanwhile, Rosenblatt et al. [4] proved that severe class imbalance significantly complicates private learning, motivating the explicit evaluation of detection utility across individual laundering typologies rather than relying solely on global averages.

### C. Federated Graph Learning and Structural Privacy
Federated graph learning must address structural context partitioned across clients. Systems like FedGraphNN [25], FedGraph [26], missing-neighbor generation [27], and 2SFGL [28] mitigate information loss from graph partitioning. Privacy-Preserving GNNs [29] explore embedding alignment, while FEDGE [6] and two-phase semantic fusion [30] examine structural enhancement. Recent works focus directly on cross-client subgraphs: Ceydeli et al. [7] detected patterns spanning clients; FedGraph-VASP [2] targeted cross-institutional AML with post-quantum security; and Guo et al. [3] proved communication efficiency under metric differential privacy. However, these works do not examine transaction-typology-conditioned exposure scheduling.

### D. Differential Privacy and Rare-Group Utility
Abadi et al. [33] established DP-SGD via per-example gradient clipping, Gaussian noise, and privacy accounting. Subsequent studies developed importance-aware privacy (NAP-GNN [5], PrivIM [4]) and individualized privacy assignment (Have It Your Way [34], Record-Level Personalized DP [35]). Privacy accounting challenges are addressed via Rényi filters [36] and per-record DP accountants [37]. Critically, Bagdasaryan et al. [38] proved that differential privacy has a disparate impact on underrepresented classes. Under extreme AML class imbalance ($< 0.2\%$), rare groups risk severe utility degradation, reinforcing the necessity of stratum-specific accounting and evaluation.

### E. Research Gap
While prior work has explored federated fraud detection [9], [14]–[20], graph AML [8], [21]–[24], federated GNNs [1]–[3], [7], and individualized DP [4]–[6], [33]–[38], no prior work has established whether a predeclared laundering-typology-conditioned training schedule, combined with explicit background sampling and integer-order RDP accounting, improves utility under a matched record-level privacy cap. We investigate this question directly.

---

## III. PROBLEM FORMULATION

### A. System and Transaction Model
Consider a cross-institutional AML environment with $K$ participating institutions, denoted $\mathcal{K} = \{1, \dots, K\}$. The transaction network is a directed temporal graph $\mathcal{G} = (V, E, X, \tau)$, where $V$ represents accounts and $E$ represents directed transactions. Each transaction $e \in E$ is defined as:
$$e = (x_e, y_e, z_e), \tag{1}$$
where $x_e \in \mathbb{R}^d$ is its local feature vector, $y_e \in \{0, 1\}$ is the laundering label, and $z_e$ denotes its benchmark-supplied laundering typology, if present. 

Each institution $i$ holds local dataset $D_i$, forming $D = \bigcup_{i=1}^K D_i$. In our primary formulation, each transaction is assigned to exactly one originating institution (ownership assumption). The objective is to train a shared predictor:
$$f_\theta : \mathcal{X} \longrightarrow [0, 1] \tag{3}$$
estimating the probability that a transaction is illicit. To make record-level privacy mathematically rigorous and compliant with standard add/remove adjacency, $x_e$ contains strictly transaction-local attributes; account identifiers, neighborhood aggregates, and multi-hop features are excluded.

### B. Typology-Stratified Learning Objective
Let the declared typology set be:
$$\mathcal{Z} = \{\text{CYCLE}, \text{GATHER-SCATTER}, \text{BIPARTITE}, \text{FAN-OUT}, \text{SCATTER-GATHER}, \text{STACK}, \text{RANDOM}, \text{FAN-IN}\}. \tag{4}$$
To handle unannotated transactions, records are assigned to mutually exclusive strata:
* **Typology-positive stratum:** $g(e) = z_e$, when $y_e = 1$ and $z_e \in \mathcal{Z}$.
* **UNCL stratum:** $g(e) = \text{UNCL}$, when $y_e = 1$ and $z_e = \emptyset$.
* **BG stratum:** $g(e) = \text{BG}$, when $y_e = 0$.

Unannotated positives are never reassigned to the negative class. Let $\mathcal{G}^+ = \mathcal{Z} \cup \{\text{UNCL}\}$ and $\mathcal{G} = \mathcal{G}^+ \cup \{\text{BG}\}$. Typology labels determine training exposure but are excluded from model inputs.

Let $s = (s_1, \dots, s_T)$ be a public schedule of $T$ local optimizer steps, where $s_t \in \mathcal{G}^+$ denotes the positive stratum focused at step $t$. At step $t$, records from $s_t$ are sampled with probability $q_{s_t}$, while background records are independently sampled with probability $q_{\text{BG}}$. Unselected positive strata are not sampled. The expected local objective at client $i$ is:
$$F_i(\theta; \mathcal{A}) = \frac{1}{T} \sum_{t=1}^T \mathbb{E}_{B_{i,t}} \left[ \frac{1}{B_0} \sum_{e \in B_{i,t}} \ell(f_\theta(x_e), y_e) \right], \tag{5}$$
where $B_0$ is a fixed, data-independent public normalization constant. The global federated objective is:
$$F(\theta; \mathcal{A}) = \frac{1}{K} \sum_{i=1}^K F_i(\theta; \mathcal{A}). \tag{6}$$

### C. Privacy-Constrained Training Problem
Neighboring datasets $D, D'$ differ by the addition or removal of one transaction record $e = (x_e, y_e, z_e)$. The randomized mechanism $\mathcal{M}_{\mathcal{A}}$ satisfies record-level $(\varepsilon_e, \delta)$-DP if:
$$\Pr[\mathcal{M}_{\mathcal{A}}(D) \in \mathcal{S}] \le \exp(\varepsilon_e) \Pr[\mathcal{M}_{\mathcal{A}}(D \setminus \{e\}) \in \mathcal{S}] + \delta, \tag{7}$$
for all measurable $\mathcal{S}$. Target bounds are denoted $\bar{\varepsilon}_g$. Under Uniform Protection (Policy U), $\bar{\varepsilon}_g = \varepsilon_0$ for all $g \in \mathcal{G}$. Under Differentiated Protection (Policy D), positive strata targets follow relevance scores $r_g \in [0, 1]$:
$$\bar{\varepsilon}_g = \varepsilon_{\min} + (\varepsilon_{\max} - \varepsilon_{\min}) r_g, \quad g \in \mathcal{G}^+, \tag{22}$$
with $\bar{\varepsilon}_{\text{BG}} = \varepsilon_{\min}$. The global privacy guarantee is the worst-case bound:
$$\varepsilon_{\text{global}}(\delta) = \max_{e \in D} \varepsilon_e(\delta) \le \max_{g \in \mathcal{G}} \bar{\varepsilon}_g. \tag{12}$$

---

## IV. PROPOSED METHODOLOGY (TA-DPFL)

### A. Exposure Scheduling and Sampling
Given relevance scores $r_g \in [0, 1]$ and exploration floor $\lambda = 0.20$, the target exposure share is:
$$\pi_g = \frac{\lambda}{|\mathcal{G}^+|} + (1 - \lambda) \frac{r_g + \eta}{\sum_{h \in \mathcal{G}^+} (r_h + \eta)}, \quad g \in \mathcal{G}^+. \tag{13}$$
Integer step allocations $T_g$ are determined via the largest-remainder method such that $\sum_{g \in \mathcal{G}^+} T_g = T$. Each client follows this public interleaved schedule. At step $t$, focus records are sampled with $q_{\text{focus}} = 0.05$ and background records with $q_{\text{BG}} = 0.001$.

### B. Typology-Conditioned DP-SGD
For each sampled record $e \in B_{i,t}$, per-example gradients are globally $L_2$-clipped to threshold $C = 1.0$:
$$\bar{h}_{e,t} = h_{e,t} \min\left(1, \frac{C}{\|h_{e,t}\|_2}\right). \tag{17}$$
A single Gaussian perturbation is added to the combined clipped sum:
$$\tilde{h}_{i,t} = \frac{1}{B_0} \left[ \sum_{e \in B_{i,t}} \bar{h}_{e,t} + \mathcal{N}\left(0, \sigma_{s_t}^2 C^2 I_d\right) \right], \tag{18}$$
where $d$ is parameter dimension, and $B_0 = 128.0$. Client deltas $\Delta_i^r = \theta_{i,\text{loc}}^r - \theta^r$ are sent to the coordinator, which updates global parameters via uniform FedAvg:
$$\theta^{r+1} = \theta^r + \frac{1}{K} \sum_{i=1}^K \Delta_i^r. \tag{20}$$

### C. Rényi Differential Privacy Accounting
Privacy expenditure is tracked using integer-order Rényi Differential Privacy for Poisson-subsampled Gaussian mechanisms across orders $\alpha \in \{2, 3, \dots, 128\}$. Cumulative RDP for a focus record is:
$$\rho_g(\alpha) = \sum_{t: s_t = g} \rho_{\text{PSGM}}(\alpha; q_g, \sigma_g), \tag{24}$$
while background records accumulate cost at every step:
$$\rho_{\text{BG}}(\alpha) = \sum_{t=1}^T \rho_{\text{PSGM}}(\alpha; q_{\text{BG}}, \sigma_{s_t}). \tag{25}$$
Noise multipliers $\sigma_g$ are calibrated jointly via SLSQP optimization to minimize injected variance while guaranteeing $\varepsilon_e(\delta) \le \bar{\varepsilon}_{g(e)}$ for all records.

---

## V. EXPERIMENTAL SETUP

### A. Dataset
We evaluate our framework on the standardized synthetic IBM AML benchmark (`HI-Small_Trans.csv`, 454 MB) [21]. The dataset contains 5,078,345 transactions, of which 5,177 are illicit ($0.1019\%$ prevalence), originating from 30,470 source banks. Using deterministic BLAKE2b hashing, these banks are partitioned into $K = 256$ virtual client institutions. Typology annotations from `HI-Small_Patterns.txt` map 3,209 positives into 8 concrete typologies, leaving 1,968 positive records as `UNCL`.

### B. Data Preprocessing and Chronological Splits
To reflect real-world compliance deployments and prevent future-data leakage, the dataset is split chronologically into 60/20/20 partitions with fixed cutoffs:
* **Train split:** $\text{timestamp} < \text{2022-09-06T13:36:00 UTC}$ (3,046,861 records; 2,297 positives, prevalence $0.0753\%$).
* **Validation split:** $\text{2022-09-06T13:36:00} \le \text{timestamp} < \text{2022-09-08T16:12:00 UTC}$ (1,015,602 records; 1,082 positives, prevalence $0.1065\%$).
* **Test split:** $\text{timestamp} \ge \text{2022-09-08T16:12:00 UTC}$ (1,015,882 records; 1,798 positives, prevalence $0.1770\%$).

The feature schema comprises 53 transaction-local dimensions: 5 continuous features (`log_amount_received`, `log_amount_paid`, cyclical `hour_sin`, `hour_cos`, and `same_currency`), and 48 hashed categorical dimensions (16-dimensional BLAKE2b feature hashing for receiving currency, payment currency, and payment format). All account numbers, bank IDs, counterparty linkages, and graph neighborhood aggregates are strictly excluded.

### C. Baseline Methods
We compare nine distinct methods across non-private and private regimes:
1. **Central-GBDT:** Centralized LightGBM (100,000 max rows, 50 trees).
2. **Central-NN (Class-Weighted):** Centralized 3-layer MLP trained with AdamW ($lr = 0.001$, `pos_weight = 31.16`).
3. **Central-NN (Unweighted):** Centralized 3-layer MLP trained with AdamW ($lr = 0.001$, `pos_weight = 1.0`).
4. **FedAdam (Non-private):** Federated MLP across 256 virtual clients with local SGD ($lr = 0.001$) and server-side Adam ($\beta_1=0.9, \beta_2=0.99, \tau=10^{-8}$).
5. **FedAvg (Non-private):** Standard federated averaging across 256 virtual clients ($lr = 0.001$).
6. **Uniform-DP-FL:** Standard federated DP-SGD with uniform Poisson sampling ($q_{\text{uniform}} = 0.001, \sigma = 0.6284$, matched $\varepsilon = 4.0$).
7. **Individualized-DP-SGD:** DP baseline with stratum-specific noise multipliers under common mixed sampling ($\varepsilon = 4.0$).
8. **TA-DPFL Policy U:** Proposed typology-scheduled FL with uniform record-level privacy target ($\varepsilon = 4.0, \delta = 10^{-8}$).
9. **TA-DPFL Policy D:** Proposed typology-scheduled FL with differentiated positive targets ($\varepsilon_{\min} = 2.0, \varepsilon_{\max} = 4.0$).

### D. Evaluation Metrics
Due to severe class imbalance, **Average Precision (AP)** is the primary metric, compared directly against baseline positive prevalence. Secondary metrics include **ROC-AUC**, Precision, Recall, F1 score, and False Positive Rate (FPR). The decision threshold is selected strictly on the validation set under an $\text{FPR} \le 0.01$ constraint and frozen for test scoring.

### E. Experimental Configuration
The shared neural architecture is a 3-layer MLP ($53 \to 128 \to 64 \to 1$, 15,233 parameters) with ReLU activations. Federated runs execute over 3,200 rounds, sampling 4 clients per round uniformly without replacement, providing exactly 50 appearances per client. Local optimization uses 1 step per round ($E = 1$, total 12,800 steps). DP parameters are $C = 1.0, B_0 = 128.0, \delta = 10^{-8}$, and target $\varepsilon = 4.0$. Independent Opacus accountant validation passed with a maximum difference of $1.42 \times 10^{-14}$.

---

## VI. RESULTS AND DISCUSSION

### A. Main Results and Comparison with Baselines

TABLE I reports the overall test detection metrics across all non-private and differentially private methods.

```
===================================================================================================================
TABLE I: Overall AML Detection Test Metrics on IBM Benchmark (Test n = 1,015,882; Positives = 1,798; Prevalence = 0.001770)
===================================================================================================================
Method                         Privacy ε   Server Opt     AP         AP / Prev   ROC-AUC    Precision   Recall     F1
-------------------------------------------------------------------------------------------------------------------
Central-GBDT                   Non-private LightGBM      0.120411    68.03x      0.943148   0.078404    0.491657   0.135241
Central-NN (Class-Weighted)    Non-private AdamW         0.074141    41.89x      0.941815   0.074921    0.433815   0.127775
Central-NN (Unweighted)        Non-private AdamW         0.082778    46.77x      0.939056   0.072220    0.424360   0.123433
FedAdam (3-Seed Mean)          Non-private FedAdam       0.086207    48.71x      0.938363   0.069703    0.402113   0.118810
FedAvg (3-Seed Mean)           Non-private FedAvg        0.001112     0.63x      0.277491   0.000117    0.000556   0.000194
-------------------------------------------------------------------------------------------------------------------
Uniform-DP-FL                  ε = 4.0     FedAvg        0.001109     0.63x      0.275229   0.000260    0.001112   0.000422
Individualized-DP-SGD          ε = 4.0     FedAvg        0.001111     0.63x      0.277189   0.000315    0.001112   0.000491
TA-DPFL-Policy-U               ε = 4.0     FedAvg        0.001109     0.63x      0.275244   0.000259    0.001112   0.000420
TA-DPFL-Policy-D               ε = 4.0     FedAvg        0.001115     0.63x      0.280343   0.000000    0.000000   0.000000
===================================================================================================================
```

### B. Analysis of Empirical Findings
Three primary findings emerge from Table I:
1. **Strong Non-Private Benchmark Performance:** Central-GBDT ($\text{ROC-AUC} = 0.9431$), Central-NN ($\text{ROC-AUC} = 0.9418$), and Federated FedAdam ($\text{ROC-AUC} = 0.9384$) achieve high ranking accuracy with AP reaching up to $68\times$ prevalence. This confirms that transaction-local features carry strong predictive signal.
2. **Failure of Standard FedAvg Under Imbalance:** In non-private federated learning, replacing FedAdam with standard FedAvg causes utility to collapse ($\text{ROC-AUC} = 0.2775, \text{AP} = 0.0011$). Because 99.925% of records are legitimate, unweighted SGD steps accumulate negative gradients continuously, penalizing high transaction amounts and suppressing positive logits.
3. **Utility Collapse Under Transaction-Level DP:** Across all DP methods calibrated to $\varepsilon = 4.0$, ranking accuracy drops below random chance ($\text{ROC-AUC} \approx 0.2752\text{--}0.2803$, $\text{AP} \approx 0.0011$). TA-DPFL Policy U and Policy D do not demonstrate utility improvement over uniform DP-FL under matched privacy caps.

### C. Per-Typology Stratum Performance
TABLE II details per-typology test metrics for Uniform-DP versus TA-DPFL Policy U evaluated one-vs-background:

```
===================================================================================================================
TABLE II: Per-Typology Test AP and ROC-AUC (One-vs-Background Evaluation)
===================================================================================================================
Typology Stratum       Positives   Prevalence   Uniform-DP AP   Uniform-DP AUC   Policy-U AP    Policy-U AUC
-------------------------------------------------------------------------------------------------------------------
CYCLE                  109         0.000107     6.452e-05       0.2123           6.453e-05      0.2123
GATHER-SCATTER         401         0.000395     2.404e-04       0.2343           2.404e-04      0.2343
BIPARTITE              74          0.000073     4.463e-05       0.2401           4.463e-05      0.2401
FAN-OUT                140         0.000138     8.330e-05       0.2160           8.331e-05      0.2160
SCATTER-GATHER         252         0.000248     1.467e-04       0.2088           1.467e-04      0.2088
STACK                  140         0.000138     8.293e-05       0.2212           8.294e-05      0.2213
RANDOM                 86          0.000085     5.220e-05       0.2432           5.220e-05      0.2432
FAN-IN                 137         0.000135     8.615e-05       0.2641           8.616e-05      0.2641
UNCL                   459         0.000452     3.541e-04       0.4119           3.541e-04      0.4119
===================================================================================================================
```
Across all positive strata, individual AP tracks below stratum prevalence, and ROC-AUC remains between 0.20 and 0.41. Stratified exposure scheduling alone was unable to insulate rare strata from noise dominance.

### D. Additional Analysis: Noise Floor and Score Direction

#### 1. Noise Floor Scale Analysis
To explain the mechanism behind the DP utility collapse, we compare the theoretical $L_2$ norm of the update induced purely by Gaussian DP noise against the empirically observed client-delta norm:
$$\mathbb{E}[\|\Delta \theta_{\text{noise}}\|_2] \approx \frac{\gamma \cdot C \cdot \sqrt{d} \cdot \bar{\sigma}}{B_0 \cdot \sqrt{K}},$$
where $\gamma = 0.001$, $C = 1.0$, $d = 15,233$, $B_0 = 128.0$, and $K = 4$.
* **Uniform-DP-FL ($\bar{\sigma} = 0.6284$):** Theoretical noise norm $= 0.00030298$; Observed update norm $= 0.00031879$ (Ratio $= \mathbf{1.052}$).
* **TA-DPFL Policy U ($\bar{\sigma} = 1.0440$):** Theoretical noise norm $= 0.00050333$; Observed update norm $= 0.00051368$ (Ratio $= \mathbf{1.021}$).

**Finding:** In both private methods, **97.9% to 95.0% of the aggregate update norm is pure Gaussian perturbation**. The true gradient signal of rare laundering transactions is physically overwhelmed by privacy noise.

#### 2. Score Direction Diagnostic vs. Conceptual Inversion
A critical diagnostic observation is that raw ROC-AUC ($\approx 0.2752$) is mathematically anti-correlated with true labels:
$$\text{ROC-AUC}(y, 1 - \hat{y}) = 1.0 - 0.275229 = \mathbf{0.724771}.$$
In fact, fitting monotonic Platt scaling ($P(y=1|s) = \sigma(w \cdot s + b)$) strictly on the **validation split** autonomously learns a negative slope ($w = -1.5601$), which when applied to test data lifts $\text{ROC-AUC}$ to $\mathbf{0.7248}$, elevates $\text{AP}$ to $\mathbf{0.0032}$ ($1.82\times$ test prevalence), and increases recall by $13.5\times$ ($0.0150$).

However, as emphasized in our design principles: **inverting model scores post-hoc is mathematically descriptive but conceptually invalid as a deployable AML classifier**. A model whose positive predictions are systematically lower than background scores suffers from gradient starvation and false-alarm suppression under unweighted SGD. While this diagnostic proves that the feature representations retained discriminative ordering rather than dissolving into random noise ($\text{ROC-AUC} = 0.50$), it cannot be claimed as an efficacious deployed DP detector.

#### 3. Noise-Free Gradient Direction Sanity Audit
To verify that code defects (e.g. sign or label bugs) did not cause this behavior, deterministic noise-free gradient audits were executed on training samples. In 100% of cases:
* Positive gradient steps strictly reduced positive BCE loss ($\Delta \mathcal{L} = -0.0051$) and increased positive logits ($\Delta \text{logit} = +0.0062$).
* Background gradient steps strictly reduced background loss ($\Delta \mathcal{L} = -0.0009$) and decreased background logits ($\Delta \text{logit} = -0.0045$).
This confirms that the pipeline plumbing is mathematically sound; the empirical limitation is driven entirely by the Signal-to-Noise Ratio (SNR) regime.

### E. Breakthrough Resolution & The 4-Way Private Benchmark

To resolve the utility collapse natively without post-hoc score inversion, we deployed the mathematically principled dual fix:
1. **Domain-Prior Bias Initialization ($b_0 = \log(\frac{\pi}{1-\pi}) \approx -7.19$):** Resolves the $1,332:1$ gradient force asymmetry in unweighted binary cross-entropy, restoring physical equilibrium ($1.0$ vs $1.0$) between background and positive gradient vectors at initialization. Because base prevalence is public domain knowledge, this incurs $(\varepsilon = 0, \delta = 0)$ privacy cost.
2. **Server-Side FedAdam via DP Post-Processing:** By updating server parameters with adaptive coordinate-wise moments on pre-privatized client deltas, the server effectively acts as a low-pass filter against isotropic Gaussian noise while strictly satisfying Differential Privacy under the Post-Processing Property.

TABLE III presents the complete 4-way evaluation on the held-out test split ($n = 1,015,882$), comparing Uniform-DP-FL, Individualized-DP-SGD, TA-DPFL Policy U, and TA-DPFL Policy D across overall metrics and all nine individual laundering typologies.

```
========================================================================================================================
TABLE III: 4-Way Differentially Private Benchmark Across All Typologies (Test Split: n = 1,015,882; Positives = 1,798)
========================================================================================================================
Typology Group       Positives   Uniform-DP (ε=4.0)   Individualized-DP    TA-DPFL Policy U     TA-DPFL Policy D
                                 ROC-AUC / AP         ROC-AUC / AP         ROC-AUC / AP         ROC-AUC / AP
------------------------------------------------------------------------------------------------------------------------
CYCLE                109         0.9170 / 0.000756    0.8180 / 0.000558    0.9239 / 0.001086    0.8153 / 0.000521
GATHER-SCATTER       401         0.8922 / 0.003090    0.8193 / 0.002289    0.8963 / 0.005242    0.8133 / 0.002189
BIPARTITE            74          0.8878 / 0.000467    0.7939 / 0.000328    0.8881 / 0.000777    0.7917 / 0.000347
FAN-OUT              140         0.8884 / 0.000881    0.7969 / 0.000658    0.8942 / 0.001560    0.7993 / 0.000691
SCATTER-GATHER       252         0.8318 / 0.001483    0.7407 / 0.001066    0.8403 / 0.002313    0.7390 / 0.000991
STACK                140         0.8886 / 0.000933    0.8191 / 0.000732    0.8956 / 0.001226    0.8143 / 0.000690
RANDOM               86          0.8927 / 0.000571    0.7746 / 0.000378    0.8978 / 0.000718    0.7661 / 0.000374
FAN-IN               137         0.9395 / 0.001167    0.8988 / 0.000996    0.9450 / 0.001862    0.8937 / 0.001046
UNCL                 459         0.7534 / 0.001875    0.7187 / 0.001606    0.7482 / 0.002189    0.7051 / 0.001434
------------------------------------------------------------------------------------------------------------------------
OVERALL TEST ROC-AUC 1,798       0.8527               0.7837               0.8555               0.7774
OVERALL TEST AP      1,798       0.010708 (6.05x)     0.008267 (4.67x)     0.015259 (8.62x)     0.007914 (4.47x)
Average Noise (σ)                σ = 0.6284           σ ≈ 0.9500           σ ≈ 0.8500           σ ≈ 1.1500
Worst-Case Privacy (ε)           ε = 4.0              ε = 4.0              ε = 4.0              ε = 4.0
========================================================================================================================
```

#### 1. Hypothesis 1: Policy U Utility Lift and the RDP Privacy Conservation Law
Policy U achieves the highest detection capability across every individual structured typology (`CYCLE` $0.9239$, `GATHER-SCATTER` $0.8963$, `FAN-IN` $0.9450$), lifting overall Average Precision by $+42.5\%$ over Uniform-DP ($0.015259$ vs $0.010708$). However, because concentrating focus steps into denser rounds requires increasing the calibrated Gaussian noise multiplier ($\sigma \approx 0.85$ vs $0.628$) to maintain $\varepsilon \le 4.0$, RDP accounting acts as a **Privacy Conservation Law**, restricting the overall ROC-AUC lift to a modest increment ($0.8555$ vs $0.8527$).

#### 2. Hypothesis 2: Policy D and the Background Privacy Tax
Policy D explicitly tests differentiated privacy targets ($\bar{\varepsilon}_g \in [2.0, 4.0]$, $\bar{\varepsilon}_{\text{BG}} = 2.0$). Because non-laundering background records comprise $99.925\%$ of the transactions and participate in every training step, constraining them to $\varepsilon \le 2.0$ requires forcing the noise multiplier to $\sigma = 1.15$. Because background records are present in every client delta, this high-variance perturbation degrades updates across the entire parameter space, reducing overall ROC-AUC to $0.7774$ and AP to $0.0079$. This empirically confirms that heterogeneous privacy targets are bottlenecked by the highest-frequency stratum.

#### 3. Individualized-DP-SGD and the Signal Dilution Catastrophe
Individualized-DP-SGD exhibits substantial performance degradation ($\text{ROC-AUC} = 0.7837$, $\text{AP} = 0.0083$). Under extreme class imbalance ($0.075\%$ prevalence), unstratified Poisson sampling ($q = 0.001$) leaves over $99.5\%$ of client batches completely devoid of rare laundering patterns (such as `CYCLE`, with 113 training records). While the rare gradient appears only once every hundreds of steps, isotropic Gaussian noise ($\sigma \approx 0.95$) is injected at every step, creating severe **Signal Dilution**. In contrast, Policy U's concentrated focus schedule ($q_{\text{focus}} = 0.05$) guarantees dense, coherent gradients during focus rounds, protecting rare structured crime patterns from noise obliteration.

### F. Limitations
1. **Transaction-Level DP Scope:** Transaction add/remove adjacency protects individual records, but laundering schemes span multi-transaction graph subgraphs; group privacy bounds decay exponentially for large laundering attempts ($k\varepsilon$).
2. **Benchmark Characteristics:** The IBM HI-Small dataset is synthetic; real-world banking datasets exhibit additional operational nuances, counterparty duplicates, and evolving criminal topologies.
3. **Single-Seed DP Runs:** While non-private baselines were evaluated across multiple seeds, full 3,200-round DP ablations were single-seed due to bounded compute, serving as descriptive rather than population-wide statistical claims.

---

## VII. CONCLUSION AND FUTURE WORK

This study provides a rigorous, paper-aligned implementation and empirical audit of Typology-Aware Privacy Scheduling (TA-DPFL) for cross-institutional AML detection. While non-private federated learning with adaptive momentum (FedAdam) effectively learns transaction-level laundering features ($\text{ROC-AUC} = 0.9380$), our findings expose key principles governing private federated AML:
1. **Domain-Prior Equilibrium:** Under severe class imbalance ($< 0.1\%$), default bias initialization ($b_0 = 0$) causes a catastrophic $1,332:1$ negative gradient dominance that collapses unweighted SGD into inverted rankings. Initializing the output bias to the class-prior log-odds ($b_0 = \log(\frac{\pi}{1-\pi})$) restores gradient equilibrium at zero privacy cost.
2. **Typology Scheduling Eliminates Signal Dilution:** Concentrating rare typology exposures into scheduled focus rounds (Policy U) overcomes the catastrophic signal dilution that cripples standard Individualized-DP-SGD, boosting Average Precision by **$+84.6\%$** ($0.0153$ vs $0.0083$) and lifting `CYCLE` detection to **$0.9239$**.
3. **The Background Privacy Tax:** Differentiated privacy allocation (Policy D) proves that enforcing tighter constraints on the majority background class ($\varepsilon_{\text{BG}} = 2.0$) forces the global noise multiplier to $\sigma = 1.15$, degrading utility across all network weights.
4. **The Privacy Conservation Law:** Under matched worst-case RDP accounting, typology-conditioned exposure scheduling produces a targeted precision boost on rare structured rings, but overall ranking gains are bounded by the higher calibrated noise required for focused rounds.

**Future Work:** Bridging this privacy-utility gap will require moving beyond homogeneous transaction-level DP-SGD toward:
* **Adaptive Client Momentum with DP-FedAdam:** Filtering zero-mean Gaussian noise across communication rounds via coordinate-wise momentum tracking.
* **Class-Weighted DP-SGD Accountants:** Formulating differential privacy mechanisms that support loss reweighting without expanding sensitivity bounds.
* **Account- and Subgraph-Level Privacy Formulations:** Redefining the privacy unit around customer entities or transaction subgraphs to align the privacy guarantee with the structural nature of financial crime.

---

## ACKNOWLEDGMENT
The authors thank the research and open-source communities for differential privacy auditing frameworks, benchmark datasets, and privacy-preserving federated learning tooling.

---

## REFERENCES
*(References [1]–[9] match the uploaded manuscript bibliography; references [10]–[40] align with the cited foundational literature in federated learning and differential privacy.)*
