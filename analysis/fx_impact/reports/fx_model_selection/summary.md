# FX Model Selection

## Compared Inputs

- Existing `hybrid_mmf` and `hybrid_m2` result JSON files are compared for full-period and anomaly-block RMSE/MAE.
- Daily anomaly-block prediction CSV files are also compared by direct RMSE/MAE, direction accuracy, monthly direction accuracy, and downstream distributed-lag RMSE.
- The selected file `selected_fx_predictions.csv` keeps the daily prediction path; the final macro pipeline resamples it to month-end by monthly mean and fills non-predicted months with actual USD/KRW for controlled error-propagation tests.

## Selected FX Input

- Selected source: `hybrid_m2_model_a`
- Full-period RMSE: 6.8682
- Anomaly-block RMSE: 13.0838
- Downstream average RMSE: 1448.962740

## Ranking

- `hybrid_m2_model_a`: selected=True, daily RMSE=18.0948, downstream RMSE=1448.962740, score=7.0
- `hybrid_mmf_model_a`: selected=False, daily RMSE=18.1034, downstream RMSE=1449.526078, score=10.0
- `hybrid_m2_model_b`: selected=False, daily RMSE=18.1619, downstream RMSE=1448.431299, score=10.0
- `hybrid_mmf_model_b`: selected=False, daily RMSE=17.3021, downstream RMSE=1449.852507, score=13.0
- `hybrid_mmf_arima`: selected=False, daily RMSE=18.1058, downstream RMSE=1449.000938, score=nan
- `hybrid_m2_naive`: selected=False, daily RMSE=18.2496, downstream RMSE=1450.942541, score=nan
