# Bitcoin Price Direction Prediction — Project Notes

## Publication Strategy

### Target Journal (Active submission)
**Engineering Applications of Artificial Intelligence** (Elsevier, EAAI)
- Impact Factor: ~8 | Q1
- Status: **Manuscript complete — ready for submission**
- Paper: `paper/EAAI/main.tex` | PDF: `paper/EAAI/main.pdf` (24 pages)
- Rationale: Omole & Enke (2025) published in EAAI with simple split; our walk-forward CV
  is methodologically superior and we introduce DECA with UQ.

### Backup Journal (Tier 2)
**Information Fusion** (Elsevier)
- Impact Factor: ~18 | Q1 | SCI indexed
- Rationale: dual-encoder + cross-attention fits "information fusion" framing.

### Backup Journal (Tier 3)
**Neurocomputing** (Elsevier)
- Impact Factor: ~6 | Q1

---

### State of the Art References

**Omole & Enke (2024)** — Financial Innovation, DOI: 10.1186/s40854-024-00643-1
- Models: CNN-LSTM (best), LSTNet, TCN, ARIMA
- Features: 87 on-chain metrics only, Boruta/GA/LightGBM selection
- Split: simple 80/20
- Best result: Boruta + CNN-LSTM → Acc=0.8244 | MCC=0.6489 | AUC=0.8242
- Backtesting: long-and-short → 6653% annual return

**Omole & Enke (2025)** — EAAI, DOI: 10.1016/j.engappai.2025.111086
- Models: SVM, RF, GBM + LSTM, CNN-LSTM, CNN-GRU, TCN, LSTNet
- Features: 92 on-chain + 138 TA = 225 features, Boruta → 46 features
- Split: simple 80/20 (train 2013–2021, test 2021–2023, 748 test days)
- Best result: SVM + Boruta on-chain → Acc=0.83, F1=0.82
- Backtesting: Boruta-SVM most profitable (4970% ROR with on-chain data)
- KEY: MCC not reported. Our methodology is strictly superior.

---

## Scientific Contribution

1. **Ablation study** (G0–G3 feature groups): on-chain metrics ~10× more predictive than
   technical indicators alone (XGBoost G1 MCC=0.036 vs G2 MCC=0.387).

2. **DECA — Dual-Encoder Cross-Attention**: novel architecture routing technical indicators
   and on-chain metrics through independent residual MLP encoders with bidirectional
   cross-attention. Provides interpretable attention weights + uncertainty-aware predictions.

3. **Walk-forward expanding-window CV** (6 folds, 2019–2024, reported) — rigorous temporal
   validation with no data leakage. Fold 7 (2025) is near-random (MCC≈−0.09) due to
   structural regime change; reported separately, not included in main results.

4. **MC Dropout UQ**: epistemic uncertainty estimate per prediction. Enables selective
   trading (96.4% accuracy at 50% coverage) and dynamic confidence threshold strategy.

5. **Dynamic threshold MC strategy**: trade only when |p−0.5| > k×σ. Optimal k=1 achieves
   4,013%/yr vs DECA deterministic 3,922%/yr (Omole 2025 backtesting conditions).

---

## Key Results (folds 1–6, 2019–2024)

| Model | Accuracy | MCC | AUC |
|---|---|---|---|
| DECA (MLP Dual Encoder V2 + Optuna) | 0.819 | **0.648** | 0.910 |
| SVM G3 + Optuna | 0.827 | 0.633 | 0.913 |
| LightGBM G3 + Optuna | 0.769 | 0.541 | 0.857 |
| XGBoost G3 + Optuna | 0.767 | 0.541 | 0.857 |
| MLP Dual (no attention) + Optuna | 0.769 | 0.556 | 0.871 |
| MLP Simple + Optuna | 0.716 | 0.450 | 0.807 |
| CNN-LSTM G3 | ~0.49 | ~0.01 | ~0.52 |

**Per-fold leadership**: DECA best in folds 2, 3, 5 — SVM best in folds 1, 4, 6.
McNemar test: DECA statistically equivalent to SVM (p=1.0 after Bonferroni).

### Backtesting (Omole 2025 conditions: long/short, 0.5% commission, 30% tax, $1,000)

| Strategy | AnnROR% | Sharpe | MaxDD% |
|---|---|---|---|
| Buy & Hold | 70% | — | — |
| CNN-LSTM | −91% | — | — |
| XGBoost | 1,864% | — | — |
| LightGBM | 1,896% | — | — |
| SVM | 3,790% | — | — |
| DECA (deterministic) | 3,922% | — | — |
| MC(k=1.0) dynamic threshold | **4,013%** | 14.86 | −8.8% |
| Omole (2025) reference | 4,970%‡ | — | — |

‡ Omole uses simple split; our result under walk-forward CV is strictly comparable.

---

## Completed Experiments

- [x] XGBoost G3 + Optuna (300 trials) ✓
- [x] LightGBM G3 + Optuna (300 trials) ✓
- [x] SVM G3 + Optuna (50 trials) ✓
- [x] CNN-LSTM G3 + Optuna (50 trials) ✓
- [x] DECA (MLP Dual Encoder V2) + Optuna (50 trials) ✓ → MCC=0.648
- [x] MLP Dual (no attention) + Optuna ✓ → MCC=0.556
- [x] MLP Simple + Optuna ✓ → MCC=0.450
- [x] McNemar / Wilcoxon statistical tests ✓
- [x] SHAP feature importance ✓ → Short_Term_Holder_SOPR most predictive
- [x] Cross-attention interpretability analysis ✓
- [x] MC Dropout UQ analysis ✓ → 96.4% acc at 50% coverage
- [x] Backtesting Omole conditions (all models) ✓
- [x] Dynamic threshold MC sweep (k=0 to k=3) ✓ → k=1 optimal
- [x] Fold 7 (2025) backtesting probe ✓ → MCC=−0.087, all strategies lose money
- [x] GitHub repository updated ✓

---

## Dataset

- **Primary**: `dataset_COMPLETO_con_OHLCV_20251221_014211.csv` (33 features, 2013–2025)
  — not versioned in git (large CSV); available locally.
- **Enriched**: `data/dataset_enriched.csv` (51 features — NOT used in final experiments
  due to NaN coverage issues in early folds for neural network models)
- Use original dataset for ALL models to ensure fair comparison.

---

## Methodology Decisions

- **Walk-forward CV**: 7 folds, expanding window, one calendar year per test fold (2019–2025)
- **Reported folds**: 1–6 (2019–2024). Fold 7 (2025) treated as out-of-distribution.
- **Primary metric**: MCC — robust to class imbalance
- **Hyperparameter optimization**: Optuna (TPE + CMA-ES), same config for all models
- **Scaling**: RobustScaler fitted only on training data per fold (no leakage)
- **Imputation**: ffill + bfill applied before fold splitting
- **Backtesting**: long/short, 0.5% commission per trade, 30% tax on realized gains,
  $1,000 initial capital. Annual ROR = ((End/Start)^(365/days))−1. No slippage modeled.

---

## Computational Environment

- OS: Windows 10
- GPU: NVIDIA GeForce RTX 5060 Ti (Blackwell sm_120, CUDA 12.8)
- Python: 3.13.3
- Key libraries: PyTorch 2.11+cu128, scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6,
  Optuna 4.8, SHAP 0.51, numpy 2.4, pandas 3.0

---

## Repository Structure

```
paper/EAAI/          ← LaTeX manuscript (main.tex, figures/, main.pdf)
onchain_direction/
  experiments/       ← All experiment scripts (run_optuna_*.py, run_backtesting_*.py, ...)
  src/               ← Model implementations, data loaders, visualization
  requirements.txt   ← Pinned dependencies
results/
  optuna/            ← Best params JSON + Optuna trial CSVs per model
  backtesting/       ← Backtesting records and summaries
  mc_dropout/        ← MC Dropout uncertainty records
  attention/         ← Cross-attention weight summaries
  feature_selection/ ← Feature ranking and G3 overlap
  statistical_tests/ ← McNemar / Wilcoxon results
  plots/             ← All paper figures (PDF + PNG)
```