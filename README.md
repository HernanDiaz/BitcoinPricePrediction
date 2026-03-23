# OnChain Direction: Bitcoin Price Direction Prediction via On-Chain Metrics and Dual Encoder Cross-Attention

> **Paper under review** — Information Fusion (Elsevier, Q1, IF~18)

## Overview

This repository contains the full reproducible implementation for the paper:

> *"Predicting Bitcoin Daily Price Direction Using On-Chain Blockchain Metrics and a Dual Encoder with Cross-Attention"*

We propose a novel **MLP Dual Encoder** architecture with bidirectional cross-attention that processes technical indicators and on-chain blockchain metrics through independent branches, enabling interpretable feature interaction learning across both domains.

### Key Results (Walk-Forward CV, 2019–2024, 6 folds)

| Model | Accuracy | MCC | AUC-ROC |
|---|---|---|---|
| XGBoost G3 + Optuna | **0.769** | **0.533** | **0.852** |
| XGBoost G3 (baseline) | 0.769 | 0.533 | 0.852 |
| RandomForest G3 | 0.668 | 0.360 | 0.769 |
| MLP Dual Encoder G3 | 0.645 | 0.302 | 0.721 |
| Baseline (OHLCV only) | 0.490 | 0.000 | 0.500 |

**Core scientific finding:** On-chain metrics are the dominant predictive signal — XGBoost with on-chain only (G2) achieves MCC=0.387 vs. MCC=0.036 with technical indicators only (G1), a 10× difference.

---

## Project Structure

```
onchain_direction/         # Main source package
├── config.yaml            # Central configuration (folds, features, hyperparameters)
├── requirements.txt       # Python dependencies
├── src/
│   ├── data/              # Dataset loading, feature groups, preprocessing
│   ├── models/            # RF, XGBoost, CNN-LSTM, MLP Dual Encoder
│   ├── validation/        # Walk-forward cross-validation
│   ├── evaluation/        # Metrics, bootstrap CI, statistical tests
│   └── visualization/     # Plots and LaTeX tables
├── experiments/
│   ├── run_ablation.py    # Full ablation (4 groups x 5 models)
│   ├── run_mlp_dual.py    # MLP Dual Encoder experiment
│   ├── run_shap.py        # SHAP interpretability analysis
│   ├── run_stats.py       # Statistical significance tests
│   ├── run_optuna_xgb.py  # XGBoost Bayesian optimisation (300 trials)
│   └── run_optuna_lgbm.py # LightGBM Bayesian optimisation (300 trials)
└── scripts/
    └── enrich_dataset.py  # Download additional on-chain features

data/
└── dataset_enriched.csv   # Full dataset (2013–2025, 33 features + target)

results/
├── metrics/               # Per-fold JSON metrics for all experiments
├── plots/                 # Publication-ready figures (PDF + PNG)
├── tables/                # LaTeX tables for the paper
├── shap/                  # SHAP importance values and plots
├── statistical_tests/     # McNemar, Wilcoxon, Bonferroni results
└── optuna/                # Hyperparameter optimisation results
```

---

## Experimental Design

### Feature Groups (Ablation Study)

| Group | Features | Description |
|---|---|---|
| G0 | OHLCV (5) | Raw market data only — baseline |
| G1 | OHLCV + Technical (15) | + RSI, Stochastic, Bollinger, OBV, etc. |
| G2 | OHLCV + On-Chain (23) | + MVRV, SOPR, aSOPR, Realized Price, etc. |
| G3 | All (33) | Full feature set |

### Walk-Forward Cross-Validation (7 folds, 2019–2025)

```
Train:  2013–2018  |  Test: 2019
Train:  2013–2019  |  Test: 2020
Train:  2013–2020  |  Test: 2021
Train:  2013–2021  |  Test: 2022
Train:  2013–2022  |  Test: 2023
Train:  2013–2023  |  Test: 2024
Train:  2013–2024  |  Test: 2025 (*)
```
(*) 2025 exhibits a market regime change (Bitcoin spot ETF institutionalisation); reported separately.

### Models

- **Random Forest** — `class_weight="balanced"`, 500 estimators
- **XGBoost** — `scale_pos_weight`, Optuna-optimised (300 trials, TPE+CMA-ES)
- **LightGBM** — `scale_pos_weight`, Optuna-optimised (300 trials, TPE+CMA-ES)
- **CNN-LSTM** — Two Conv1D + stacked LSTM, BCEWithLogitsLoss
- **MLP Dual Encoder** *(proposed)* — Two MLP branches + bidirectional cross-attention

---

## Reproducibility

### Requirements

- Python 3.13.3
- PyTorch 2.11.0+cu128 (CUDA 12.8, requires NVIDIA GPU)
- See `onchain_direction/requirements.txt` for all dependencies

### Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install PyTorch (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

# 3. Install remaining dependencies
pip install -r onchain_direction/requirements.txt
```

### Run Experiments

```bash
# Full ablation study (RF, XGBoost, CNN-LSTM — ~60 min)
python onchain_direction/experiments/run_ablation.py

# MLP Dual Encoder experiment (~15 min)
python onchain_direction/experiments/run_mlp_dual.py

# Hyperparameter optimisation — XGBoost (300 trials, ~90 min)
python onchain_direction/experiments/run_optuna_xgb.py

# Hyperparameter optimisation — LightGBM (300 trials, ~90 min)
python onchain_direction/experiments/run_optuna_lgbm.py

# SHAP interpretability analysis
python onchain_direction/experiments/run_shap.py

# Statistical significance tests (McNemar, Wilcoxon, Bonferroni)
python onchain_direction/experiments/run_stats.py
```

All results are saved to `results/`. Random seed is fixed at `42` across all experiments for full reproducibility.

---

## Data

The dataset covers **2013-01-03 to 2025-12-20** (4,735 daily observations).

| Category | Features |
|---|---|
| OHLCV | Open, High, Low, Close, Volume |
| Technical | RSI_14, Stoch_K/D, BB_PercentB, OBV, Dist_to_SMA200, ROI30d, Sharpe_30d, Drawdown_from_ATH, logret_1d |
| On-Chain | MVRV, SOPR, aSOPR, RealizedPrice, Supply_in_Profit/Loss, UTXOs_in_Profit/Loss, NRPL, NUL, CapMVRVCur, Price_to_Realized, MVRV_z_365, ... |

**Target:** `y = 1` if `Close(t+1) > Close(t)`, else `y = 0`

The enriched dataset (`data/dataset_enriched.csv`) includes additional features downloaded via free APIs (Fear & Greed Index, Hash Rate, NVT Ratio, Puell Multiple). The download script is provided in `onchain_direction/scripts/enrich_dataset.py`.

> **Full dataset and trained models** are available on Zenodo: [DOI — to be added upon acceptance]

---

## Citation

```bibtex
@article{onchain_direction_2026,
  title   = {Predicting Bitcoin Daily Price Direction Using On-Chain Blockchain Metrics
             and a Dual Encoder with Cross-Attention},
  journal = {Information Fusion},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
