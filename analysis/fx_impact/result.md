# FX Impact Final Pipeline Result

## Purpose

This final pipeline estimates how domestic macro/financial variables respond with lags when USD/KRW follows an actual, predicted, or scenario shock path during anomaly-style conditions.

## Reproduction Commands

```bash
python analysis/fx_impact/run_final_fx_impact_pipeline.py
python analysis/fx_impact/lead_lag_causality_analysis.py
python analysis/fx_impact/predict_arimax.py
python analysis/fx_impact/predict_varx.py
```

## Target Selection

Targets are selected from source level columns only. Derived `_MoM`, `_YoY`, and `_lag` columns are excluded as target candidates to avoid duplicate targets and leakage.

- `CSI_CCSI`: lag=1 months, direction=negative, score=0.822, transform=diff
- `Imports`: lag=1 months, direction=negative, score=0.579, transform=log_diff
- `KOSPI`: lag=1 months, direction=negative, score=0.573, transform=log_diff
- `Industrial_Production`: lag=2 months, direction=negative, score=0.553, transform=log_diff
- `Import_Price_Index`: lag=2 months, direction=negative, score=0.535, transform=log_diff
- `Foreign_Bond_Investment`: lag=6 months, direction=negative, score=0.513, transform=diff
- `Foreign_Stock_Investment`: lag=2 months, direction=positive, score=0.398, transform=diff
- `Trade_Balance`: lag=4 months, direction=positive, score=0.393, transform=diff

## Selected FX Input

- Selected FX source: `hybrid_m2_model_a`
- Daily hybrid prediction outputs are resampled to month-end using monthly means. Months without an available hybrid prediction retain actual USD/KRW in the predicted path, so the comparison isolates error propagation where predictions exist.

## Final Impact Models

- `ARIMAX`: per-target SARIMAX on transformed target with target-specific selected USD/KRW lag.
- `RIDGE_DLM`: recursive regularized distributed-lag model using target lag 1 and USD/KRW lags 1-6.
- `TREE_DLM`: recursive ExtraTrees distributed-lag model using the same lag structure.
- `VARX`: multivariate VAR with USD/KRW lags 1-6 as exogenous inputs for the top selected targets.
- Calendar-time forecast tables are still saved for comparison, but final forecast plots now use the concatenated anomaly-month set.
- In the anomaly-set run, `Is_Abnormal_Period == 1` months are filtered first and then treated as one stitched sequence; lag distortion across gaps is intentionally ignored for this prototype pass.

- Scenario path: persistent `5.0%` upward shock to the selected predicted FX level from the first test month.
- Best predicted-FX model by average transformed NRMSE: `RIDGE_DLM`

## Model Comparison

- `VARX` with `actual` FX: avg transform RMSE=682.625994, avg transform NRMSE=1.0716, avg level RMSE=3227406.3751, targets=6
- `ARIMAX` with `actual` FX: avg transform RMSE=1526.114622, avg transform NRMSE=1.0174, avg level RMSE=471839.9350, targets=8
- `RIDGE_DLM` with `actual` FX: avg transform RMSE=1566.930119, avg transform NRMSE=1.0297, avg level RMSE=384143.7227, targets=8
- `TREE_DLM` with `actual` FX: avg transform RMSE=1579.559557, avg transform NRMSE=1.0339, avg level RMSE=343880.0313, targets=8
- `VARX` with `predicted` FX: avg transform RMSE=620.962049, avg transform NRMSE=1.0499, avg level RMSE=3130076.5418, targets=6
- `ARIMAX` with `predicted` FX: avg transform RMSE=1533.129494, avg transform NRMSE=1.0251, avg level RMSE=478648.2699, targets=8
- `RIDGE_DLM` with `predicted` FX: avg transform RMSE=1548.173451, avg transform NRMSE=1.0218, avg level RMSE=373274.4556, targets=8
- `TREE_DLM` with `predicted` FX: avg transform RMSE=1571.809883, avg transform NRMSE=1.0486, avg level RMSE=547266.9737, targets=8
- `VARX` with `scenario` FX: avg transform RMSE=620.962049, avg transform NRMSE=1.0499, avg level RMSE=3130076.5418, targets=6
- `ARIMAX` with `scenario` FX: avg transform RMSE=1533.811204, avg transform NRMSE=1.0247, avg level RMSE=588280.2117, targets=8
- `RIDGE_DLM` with `scenario` FX: avg transform RMSE=1558.868055, avg transform NRMSE=1.0306, avg level RMSE=623990.0903, targets=8
- `TREE_DLM` with `scenario` FX: avg transform RMSE=1597.044996, avg transform NRMSE=1.0502, avg level RMSE=673111.8130, targets=8

## Concatenated Anomaly-Set Model Comparison

- `TREE_DLM` with `actual` FX: avg transform NRMSE=1.0558, avg level NRMSE=2.3016, targets=7
- `ARIMAX` with `actual` FX: avg transform NRMSE=1.0585, avg level NRMSE=1.5410, targets=8
- `RIDGE_DLM` with `actual` FX: avg transform NRMSE=1.0913, avg level NRMSE=2.2646, targets=7
- `ARIMAX` with `predicted` FX: avg transform NRMSE=1.0835, avg level NRMSE=1.5613, targets=8
- `RIDGE_DLM` with `predicted` FX: avg transform NRMSE=1.0987, avg level NRMSE=2.3017, targets=7
- `TREE_DLM` with `predicted` FX: avg transform NRMSE=1.1185, avg level NRMSE=2.3368, targets=7
- `RIDGE_DLM` with `scenario` FX: avg transform NRMSE=1.1047, avg level NRMSE=1.9856, targets=7
- `ARIMAX` with `scenario` FX: avg transform NRMSE=1.1084, avg level NRMSE=1.6797, targets=8
- `TREE_DLM` with `scenario` FX: avg transform NRMSE=1.1317, avg level NRMSE=2.1723, targets=7

## Lag Effect Summary

- `CSI_CCSI`: lag=1, effect per +1% FX move=-0.455974 (unit_change), peak scenario delta=-2.8237
- `Imports`: lag=1, effect per +1% FX move=-0.003672 (log_change), peak scenario delta=-2789907.1192
- `KOSPI`: lag=1, effect per +1% FX move=-0.002613 (log_change), peak scenario delta=-59.8831
- `Industrial_Production`: lag=2, effect per +1% FX move=-0.000748 (log_change), peak scenario delta=-0.9897
- `Import_Price_Index`: lag=2, effect per +1% FX move=-0.002061 (log_change), peak scenario delta=0.0000
- `Foreign_Bond_Investment`: lag=6, effect per +1% FX move=-136.289444 (unit_change), peak scenario delta=-717.7911
- `Foreign_Stock_Investment`: lag=2, effect per +1% FX move=126.209182 (unit_change), peak scenario delta=1677.5495
- `Trade_Balance`: lag=4, effect per +1% FX move=59.186905 (unit_change), peak scenario delta=613.2317

## Concatenated Anomaly-Set Scenario Peaks

- `CSI_CCSI`: peak model=RIDGE_DLM, peak horizon=6, peak scenario level delta=-3.6749
- `Imports`: peak model=RIDGE_DLM, peak horizon=10, peak scenario level delta=-4660298.0134
- `KOSPI`: peak model=RIDGE_DLM, peak horizon=9, peak scenario level delta=-100.1216
- `Industrial_Production`: peak model=RIDGE_DLM, peak horizon=8, peak scenario level delta=-2.3396
- `Import_Price_Index`: peak model=ARIMAX, peak horizon=1, peak scenario level delta=0.0000
- `Foreign_Bond_Investment`: peak model=RIDGE_DLM, peak horizon=6, peak scenario level delta=-814.2646
- `Foreign_Stock_Investment`: peak model=TREE_DLM, peak horizon=12, peak scenario level delta=1267.1708
- `Trade_Balance`: peak model=RIDGE_DLM, peak horizon=10, peak scenario level delta=776.8233

## Outputs

- `analysis/fx_impact/reports/target_selection/target_ranking.csv`
- `analysis/fx_impact/reports/target_selection/target_ranking.json`
- `analysis/fx_impact/reports/fx_model_selection/fx_model_comparison.csv`
- `analysis/fx_impact/reports/fx_model_selection/selected_fx_predictions.csv`
- `analysis/fx_impact/reports/final/model_comparison.csv`
- `analysis/fx_impact/reports/final/impact_forecasts.csv`
- `analysis/fx_impact/reports/final/lag_effect_summary.csv`
- `analysis/fx_impact/reports/final/scenario_forecasts.csv`
- `analysis/fx_impact/reports/final/plot_model_selection.csv`
- `analysis/fx_impact/reports/final/anomaly_set/anomaly_model_panel.csv`
- `analysis/fx_impact/reports/final/anomaly_set/anomaly_impact_forecasts.csv`
- `analysis/fx_impact/reports/final/anomaly_set/anomaly_model_comparison.csv`
- `analysis/fx_impact/reports/final/anomaly_set/anomaly_scenario_forecasts.csv`
- `analysis/fx_impact/reports/final/anomaly_set/anomaly_lag_effect_summary.csv`

## Interpretation Notes

- `log_diff` forecasts are monthly log changes. Level forecasts are recursively inverse-transformed with `level_t = level_{t-1} * exp(log_change_t)`.
- `diff` forecasts are monthly unit changes. Level forecasts are recursively inverse-transformed with `level_t = level_{t-1} + change_t`.
- Anomaly-only target selection is reported but not over-weighted because overlap can be small for some targets.
