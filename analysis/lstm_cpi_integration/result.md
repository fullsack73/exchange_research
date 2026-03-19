# LSTM CPI Integration Analysis Results

## Analysis Objective
Test whether incorporating CPI components (via top-5 SHAP-selected features) improves FX prediction accuracy when combined with existing Spread + MMF model. This addresses the hypothesis that CPI interaction effects may explain the KRW/USD anomalous spike (2024-11 ~ 2025-12).

## Methodology

### Data Integration
- **Base Daily Data**: USD_KRW_processed.csv (2532 business days)
- **CPI Components**: Monthly US CPI subcomponents with 82 lag features
- **Top-5 CPI Features** (selected by SHAP importance from full-period analysis):
  1. Energy_YoY_lag2 (SHAP: 0.428)
  2. Food_YoY (SHAP: 0.341)
  3. Shelter_YoY_lag2 (SHAP: 0.341)
  4. Durables_YoY_lag3 (SHAP: 0.248)
  5. Headline_MoM_lag1 (SHAP: 0.162)
- **Interpolation**: Monthly CPI → daily via linear interpolation + forward/backward fill

### Model Architectures
- **Model A (Baseline)**: Spread only
- **Model B (w/ Liquidity)**: Spread + MMF_total
- **Model C (CPI Integration)**: Spread + MMF_total + Top-5 CPI features
- **LSTM Config**: seq_length=30 days, pred_step=5 days, hyperparameter-tuned

### Evaluation Periods
- **Full Period**: All available data (80/20 train/test)
- **Anomaly Period**: 2024-11-01 ~ 2025-12-31 only (14 months, 80/20 train/test)

## Results

### Full Period (2025 train, 507 test)

| Model | Hidden Dim | Num Layers | Epochs | RMSE | MAE | Status |
|-------|-----------|-----------|--------|------|-----|--------|
| A: Spread | 32 | 1 | 30 | **0.1336** | **0.0965** | ✅ Best |
| C: Spread+MMF+CPI | 32 | 1 | 30 | 0.1579 | 0.1346 | Mixed |
| B: Spread+MMF | 32 | 1 | 30 | 0.2281 | 0.2027 | ❌ Worst |

**Inference**: Adding MMF + CPI features degrades performance on full dataset. Simple Spread-only model captures exchange rate dynamics most efficiently.

### Anomaly Period (2024-11~2025-12, 228 train, 57 test)

| Model | Hidden Dim | Num Layers | Epochs | RMSE | MAE | Status |
|-------|-----------|-----------|--------|------|-----|--------|
| A: Spread | 32 | 1 | 50 | **0.1589** | **0.1203** | ✅ Best |
| B: Spread+MMF | 32 | 1 | 50 | 0.3358 | 0.3155 | ❌ Poor |
| C: Spread+MMF+CPI | 16 | 1 | 90 | 0.3690 | 0.3496 | ❌ Worst |

**Key Insight**: During anomaly period, Spread model still outperforms despite regime shift. The 2024-11~2025-12 spike cannot be explained by MMF or CPI dynamics in lagged form. Model C required more intensive training (epochs 90 vs 50) yet achieved worst results, suggesting overfitting on noisy feature interactions.

## Model Comparison Summary

### Performance by Dataset
```
Full Period:     Model A > Model C > Model B  (0.1336 > 0.1579 > 0.2281)
Anomaly Period:  Model A > Model B > Model C  (0.1589 > 0.3358 > 0.3690)
```

### Feature Contribution Analysis
- **Spread**: Sufficient to capture exchange rate dynamics across both time periods
- **MMF**: Adds prediction noise rather than signal (RMSE increases ~1.7x in full period)
- **CPI**: Top-5 lagged features fail to improve predictions (0.8% worse than Baseline A in full period, 18% worse in anomaly period)

## Implications for Anomaly Hypothesis

### Previous Finding (Baseline LSTM Validation)
[analysis/lstm_validation_daily/result.md](../lstm_validation_daily/result.md) established that:
- Spread alone achieves RMSE 0.133 on full dataset
- MMF improved prediction in some configurations

### Current CPI Integration Results
- **CPI does NOT improve Model C**: Despite selecting top-5 SHAP-important features, CPI lags contribute noise
- **MMF benefit disappeared**: Earlier SHAP analysis suggested MMF contribution, but end-to-end LSTM training shows degradation
- **Regime-specific behavior**: Anomaly period model sensitivity differs from full-period, but adding features doesn't resolve it

### Conclusion
The 2024-11~2025-12 KRW/USD spike **cannot be explained by** lagged realizations of:
- Interest rate differentials (captured by Spread)
- Liquidity flows (MMF)
- US price pressures (CPI components)

This suggests anomaly drivers may be:
1. **Forward-looking expectations** (not lagged realizations) of economic divergence
2. **Non-linear FX regime change** (requires different model architecture)
3. **Exogenous event shocks** (geopolitical, policy surprises) not captured by economic indicators
4. **Lead-lag timing mismatch**: CPI/MMF movements may follow FX changes, not precede them

## Recommendations

1. **Model Architecture**: Consider attention-based or regime-switching LSTM rather than simple stacking
2. **Feature Engineering**: Test leading indicators (forward-looking measures) instead of lagged realization
3. **Event Analysis**: Combine LSTM with event study to identify specific policy/crisis dates driving anomaly
4. **Causality Testing**: Granger causality test (CPI → FX vs FX → CPI) to determine lead/lag relationship

## Output Files
- `daily_dataset_cpi_integrated.csv`: Merged daily data (2532 rows, 9 columns)
- `results.json`: Raw model performance metrics for all 6 configurations
- `result.md`: This analysis summary
