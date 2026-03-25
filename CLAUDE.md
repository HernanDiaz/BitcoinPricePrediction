# Bitcoin Price Direction Prediction — Project Notes

## Publication Strategy

### Target Journal (Tier 1)
**Information Fusion** (Elsevier)
- Impact Factor: ~18 | Q1 | SCI indexed
- Rationale: No paper on Bitcoin direction prediction currently published there → opportunity.
  The dual-encoder architecture with cross-attention between on-chain and technical feature
  branches fits the "information fusion" framing perfectly.
- Submission readiness: experiments + statistical validation complete.

### Backup Journal (Tier 2)
**Engineering Applications of Artificial Intelligence** (Elsevier, EAAI)
- Impact Factor: ~8 | Q1
- Rationale: Omole & Enke published similar work (83% acc, simple split). Our walk-forward CV
  is methodologically superior.
- Additional experiments needed IF rejected from IF:
  1. Backtesting / trading simulation (buy/sell strategy, Sharpe ratio, max drawdown)
  2. Re-implementation of Omole & Enke baseline evaluated with our walk-forward CV
  3. Analysis of temporal stability of SHAP feature importance across folds

### State of the Art References

**Omole & Enke (2024)** — Financial Innovation, DOI: 10.1186/s40854-024-00643-1
- Models: CNN-LSTM (best), LSTNet, TCN, ARIMA
- Features: 87 on-chain metrics only (NO technical indicators), Boruta/GA/LightGBM selection
- Split: simple 80/20 (no walk-forward CV)
- Best result: Boruta + CNN-LSTM → Acc=0.8244 (max), mean=0.7439±0.050 | MCC=0.6489 | AUC=0.8242
- Backtesting: long-and-short strategy → 6653% annual return
- Statistical tests: Wilcoxon signed-rank + Friedman ✓
- KEY: 82.44% is the MAXIMUM across experiments, not the typical result
- Our CNN-LSTM with walk-forward CV → MCC≈0.01 (near-random), showing simple split inflates results

**Omole & Enke (2025)** — EAAI, DOI: 10.1016/j.engappai.2025.111086
- Models: SVM, RF, GBM (ML) + LSTM, CNN-LSTM, CNN-GRU, TCN, LSTNet (DL)
- Features: 92 on-chain + 138 TA indicators = 225 features, Boruta selection → 46 features
- Split: simple 80/20 (train 2013-2021, test 2021-2023, 748 test days)
- Best result: SVM + Boruta on-chain data → Acc=0.83, F1=0.82 (ML beats DL!)
- TA indicators alone → ~50-53% accuracy (near-random) — confirms our G1 ablation finding
- On-chain data >> TA for classification — aligns with our G1 vs G2 ablation
- Backtesting: Boruta-SVM most profitable (4970% ROR with on-chain data)
- Hyperparameter tuning: random search (ours: Optuna TPE, more systematic)
- KEY: MCC not reported; uses only accuracy/F1. Our methodology is strictly superior.

### Backup Journal (Tier 3)
**Neurocomputing** (Elsevier)
- Impact Factor: ~6 | Q1

---

## Scientific Contribution

1. **Ablation study** (G0-G3 feature groups) demonstrating on-chain metrics are ~10x more
   predictive than technical indicators alone (XGBoost G1 MCC=0.036 vs G2 MCC=0.387).

2. **MLP Dual Encoder with bidirectional cross-attention** — novel architecture that processes
   technical and on-chain features through independent MLP branches and learns cross-domain
   interactions via stacked cross-attention layers.

3. **Walk-forward expanding-window CV** (7 folds, 2019–2025) — rigorous temporal validation
   with no data leakage. Most published papers use simple train/test splits.

4. **Temporal redundancy hypothesis** — the temporal information relevant for daily Bitcoin
   direction prediction is already captured by pre-computed technical indicators (RSI, MACD,
   Bollinger Bands) and on-chain metrics (MVRV, SOPR, NVT) that synthesize price history.
   Additional sequence modeling via CNN-LSTM does not provide incremental value because the
   input features are themselves temporal summaries — running a sequence model over them
   creates redundancy rather than new signal. This explains CNN-LSTM failure under walk-forward
   CV while static models (XGBoost, MLP) succeed.

5. **Market regime change 2025** — Fold 7 (2025) collapses to near-random (MCC≈0) across
   all models, consistent with structural break caused by institutional ETF adoption.

---

## Key Results (folds 2019–2024, stable period)

| Model | Accuracy | MCC | AUC |
|---|---|---|---|
| XGBoost G3 + Optuna | 0.767 | 0.541 | 0.857 |
| LightGBM G3 + Optuna | 0.769 | 0.541 | 0.857 |
| MLP Dual Encoder V2 + Optuna | 0.819 | 0.648 | 0.910 |
| MLP Dual (sin atención) + Optuna | 0.769 | 0.556 | 0.871 |
| MLP Simple + Optuna | 0.716 | 0.450 | 0.807 |
| XGBoost G3 baseline | 0.714 | 0.432 | 0.789 |
| Random Forest G3 | 0.668 | 0.360 | 0.769 |
| CNN-LSTM G3 | ~0.49 | ~0.01 | ~0.52 |

---

## Pending Experiments

- [x] XGBoost G3 + Optuna (50 trials, original dataset) ✓
- [x] LightGBM G3 + Optuna (50 trials, original dataset) ✓
- [x] MLP Dual Encoder V2 + Optuna (50 trials) ✓ → MCC=0.648 (folds 1-6)
- [x] MLP Dual (sin atención) + Optuna (50 trials) ✓ → MCC=0.556 (folds 1-6)
- [x] MLP Simple + Optuna (50 trials) ✓ → MCC=0.450 (folds 1-6)
- [x] Tests estadísticos (McNemar / Wilcoxon) ✓ → McNemar significativo para MLP_Dual_V2 vs todos
- [x] SHAP feature importance analysis ✓ → Short_Term_Holder_SOPR más predictivo
- [x] Final plots and tables ✓ → results/plots/
- [ ] Fix numpy.ptp() bug in plots.py (line ~291): x_col.ptp() → x_col.max()-x_col.min()
- [ ] Limpiar estructura de carpetas (bitcoin_paper/ vs onchain_direction/)
- [ ] Update GitHub repository with final results

---

## Dataset

- **Primary**: `dataset_COMPLETO_con_OHLCV_20251221_014211.csv` (33 features, 2013–2025)
- **Enriched**: `data/dataset_enriched.csv` (51 features — NOT used in final experiments
  due to NaN coverage issues in early folds for neural network models)
- Use original dataset for ALL models to ensure fair comparison.

---

## Methodology Decisions

- **Walk-forward CV**: 7 folds, expanding window, one calendar year per test fold (2019–2025)
- **Primary metric**: MCC (Matthews Correlation Coefficient) — robust to class imbalance
- **Hyperparameter optimization**: Optuna (TPE + CMA-ES), same config for all models
- **Scaling**: RobustScaler fitted only on training data per fold (no leakage)
- **Imputation**: ffill + bfill applied before fold splitting
- **Fold 2025**: reported separately — structural break / out-of-distribution regime
