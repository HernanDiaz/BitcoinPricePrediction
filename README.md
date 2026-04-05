# Bitcoin Price Direction Prediction via On-Chain Metrics and Dual-Encoder Cross-Attention (DECA)

> **Paper submitted to Engineering Applications of Artificial Intelligence** (Elsevier, Q1, IF~8)

## Overview

This repository contains the full reproducible implementation for the paper:

> *"Predicting Bitcoin Daily Price Direction Using On-Chain Blockchain Metrics and a Dual-Encoder Cross-Attention Architecture"*

We propose **DECA (Dual-Encoder Cross-Attention)**, a novel architecture that routes technical indicators and blockchain on-chain metrics through independent residual MLP encoders and learns their cross-domain interactions via cross-attention. DECA provides both interpretable attention weights and uncertainty-aware predictions through Monte Carlo Dropout.

All models are evaluated under a **six-fold expanding-window walk-forward cross-validation** protocol (2019–2024), which is significantly more rigorous than the simple 80/20 train-test splits used in prior work.

---

## Key Results (folds 1–6, 2019–2024)

| Model | Accuracy | MCC | AUC-ROC |
|---|---|---|---|
| **DECA (ours)** | 0.819 | **0.648** | 0.910 |
| SVM G3 + Optuna | **0.827** | 0.633 | **0.913** |
| LightGBM G3 + Optuna | 0.769 | 0.541 | 0.857 |
| XGBoost G3 + Optuna | 0.767 | 0.541 | 0.857 |
| MLP Dual (no attention) | 0.769 | 0.556 | 0.871 |
| MLP Simple | 0.716 | 0.450 | 0.807 |
| CNN-LSTM G3 | ~0.490 | ~0.010 | ~0.520 |

DECA and SVM are statistically equivalent in directional accuracy (McNemar p=1.0 after Bonferroni correction). DECA additionally provides cross-attention interpretability and calibrated epistemic uncertainty via Monte Carlo Dropout.

**Core ablation finding:** On-chain metrics provide approximately **10x more predictive signal** than technical indicators alone (XGBoost G2 MCC=0.387 vs G1 MCC=0.036).

### MC Dropout — Selective Prediction

| Coverage | Accuracy |
|---|---|
| 100% | 81.9% |
| 75% | 89.2% |
| 50% | **96.4%** |

### Backtesting (Omole 2025 conditions: long/short, 0.5% commission, 30% tax, $1,000 capital)

| Strategy | Annualised ROR |
|---|---|
| Buy & Hold | 70% |
| CNN-LSTM | -91% |
| XGBoost | 1,864% |
| LightGBM | 1,896% |
| SVM | 3,790% |
| DECA (deterministic) | 3,922% |
| **MC(k=1) dynamic threshold** | **4,013%** |
| Omole (2025) reference (*) | 4,970% |

(*) Omole uses a simple train-test split; results are not directly comparable under our walk-forward protocol.

---

## Project Structure

```
paper/EAAI/                        # LaTeX manuscript (submitted to EAAI)
    main.tex                       # Full paper source
    main.pdf                       # Compiled PDF (24 pages)
    references.bib
    figures/                       # All paper figures (PDF)

onchain_direction/
    requirements.txt               # Pinned dependencies (Python 3.13.3)
    src/
        data/                      # Dataset loading, feature groups, preprocessing
        models/                    # RF, XGBoost, LightGBM, SVM, CNN-LSTM, DECA
        validation/                # Walk-forward cross-validation
        evaluation/                # MCC, AUC, McNemar, Wilcoxon
        visualization/             # Elsevier-formatted plots
    experiments/
        run_optuna_mlp.py          # DECA Optuna tuning (50 trials)
        run_optuna_svm.py          # SVM Optuna tuning (50 trials)
        run_optuna_cnn_lstm.py     # CNN-LSTM Optuna tuning (50 trials)
        run_paper_plots.py         # Generate all paper figures
        run_paper_stats.py         # McNemar / Wilcoxon statistical tests
        run_paper_shap.py          # SHAP feature importance
        run_attention_analysis.py  # Extract cross-attention weights
        run_attention_plots.py     # Cross-attention interpretability figures
        run_mc_dropout_v2.py       # MC Dropout uncertainty analysis
        run_feature_selection.py   # Feature ranking and G3 overlap

run_backtesting_omole.py               # Long/short backtesting (Omole 2025 conditions)
run_backtesting_mc_omole.py            # MC Dropout backtesting
run_backtesting_dynamic_threshold.py   # Dynamic confidence threshold sweep (k=0..3)

results/
    optuna/            # Best params JSON + Optuna trial CSVs per model
    backtesting/       # Backtesting records and summaries
    mc_dropout/        # MC Dropout uncertainty records
    attention/         # Cross-attention weight summaries
    feature_selection/ # Feature ranking and G3 overlap analysis
    statistical_tests/ # McNemar / Wilcoxon results
    plots/             # All paper figures (PDF + PNG)
```

---

## Experimental Design

### Feature Groups (Ablation Study)

| Group | N features | Description |
|---|---|---|
| G0 | 5 | OHLCV only — baseline |
| G1 | 15 | OHLCV + Technical indicators (RSI, Stochastic, Bollinger, OBV, ...) |
| G2 | 23 | OHLCV + On-chain metrics (MVRV, SOPR, aSOPR, Realized Price, ...) |
| G3 | 33 | All features — used for all final model comparisons |

### Walk-Forward Cross-Validation

```
Train: 2013–2018  |  Test: 2019  (fold 1)
Train: 2013–2019  |  Test: 2020  (fold 2)
Train: 2013–2020  |  Test: 2021  (fold 3)
Train: 2013–2021  |  Test: 2022  (fold 4)
Train: 2013–2022  |  Test: 2023  (fold 5)
Train: 2013–2023  |  Test: 2024  (fold 6)  <- last reported fold
Train: 2013–2024  |  Test: 2025  (fold 7, out-of-distribution — reported separately)
```

Fold 7 (2025) exhibits near-random performance (DECA MCC=-0.09), consistent with a structural market regime change following Bitcoin spot ETF adoption.

---

## Reproducibility

### Requirements

- Python 3.13.3
- PyTorch 2.11.0+cu128 (CUDA 12.8, tested on NVIDIA RTX 5060 Ti)
- See `onchain_direction/requirements.txt` for all pinned dependencies

### Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r onchain_direction/requirements.txt
```

### Run Experiments

```bash
# Hyperparameter optimisation (results saved to results/optuna/)
python onchain_direction/experiments/run_optuna_mlp.py          # DECA (~2h on GPU)
python onchain_direction/experiments/run_optuna_svm.py          # SVM (~30 min)
python onchain_direction/experiments/run_optuna_cnn_lstm.py     # CNN-LSTM (~1h on GPU)

# Paper figures and statistical tests
python onchain_direction/experiments/run_paper_plots.py
python onchain_direction/experiments/run_paper_stats.py
python onchain_direction/experiments/run_paper_shap.py

# MC Dropout uncertainty and attention interpretability
python onchain_direction/experiments/run_mc_dropout_v2.py
python onchain_direction/experiments/run_attention_analysis.py
python onchain_direction/experiments/run_attention_plots.py

# Backtesting
python run_backtesting_omole.py
python run_backtesting_mc_omole.py
python run_backtesting_dynamic_threshold.py
```

All results are saved to `results/`. Random seed fixed at `42` for reproducibility.

---

## Data

The dataset covers **2013-01-03 to 2025-12-20** (4,735 daily observations, 33 features).

| Category | Features |
|---|---|
| OHLCV (5) | Open, High, Low, Close, Volume |
| Technical (10) | RSI_14, Stoch_K/D, BB_PercentB, OBV, Dist_SMA200, ROI30d, Sharpe_30d, Drawdown_ATH, logret_1d |
| On-Chain (18) | MVRV, SOPR, aSOPR, RealizedPrice, Supply_in_Profit/Loss, UTXOs_in_Profit/Loss, NRPL, NUL, CapMVRVCur, Price_to_Realized, MVRV_z_365, STH_SOPR, NVT, ... |

**Target:** `y = 1` if `Close(t+1) > Close(t)`, else `y = 0`

> The dataset CSV is not versioned in this repository (large file).
> Full dataset and trained model weights will be made available on Zenodo upon acceptance.

---

## Citation

```bibtex
@article{deca_bitcoin_2026,
  title   = {Predicting Bitcoin Daily Price Direction Using On-Chain Blockchain Metrics
             and a Dual-Encoder Cross-Attention Architecture},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
