import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def select_prediction_column(preds_df, requested=None):
    if requested:
        if requested not in preds_df.columns:
            raise ValueError(f"Prediction column '{requested}' not found in {list(preds_df.columns)}")
        return requested
    for col in ["pred_fx", "pred_model_a", "pred_model_b", "pred_arima", "pred_naive"]:
        if col in preds_df.columns:
            return col
    raise ValueError("No prediction column found. Expected one of pred_fx, pred_model_a, pred_model_b, pred_arima, pred_naive.")

def prepare_data(macro_filepath, preds_filepath, prediction_column=None):
    # 1. Load Macro Data
    df = pd.read_csv(macro_filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    targets = [
        'Import_Price_Index', 'CPI_KOR', 
        'Trade_Balance', 'Foreign_Stock_Investment', 
        'Foreign_Bond_Investment', 'KOSPI'
    ]
    
    # 2. Load Predictions
    preds_df = pd.read_csv(preds_filepath)
    preds_df['date'] = pd.to_datetime(preds_df['date'])
    preds_df = preds_df.set_index('date')
    pred_col = select_prediction_column(preds_df, prediction_column)
    
    monthly_preds = preds_df[pred_col].resample('ME').mean()
    monthly_preds.index = monthly_preds.index + pd.offsets.MonthEnd(0)
    
    # 3. Combine Levels
    levels_df = df[['USD_KRW'] + targets].copy().dropna()
    levels_df['USD_KRW_pred'] = levels_df['USD_KRW']
    
    for date in monthly_preds.index:
        if date in levels_df.index and not np.isnan(monthly_preds[date]):
            levels_df.loc[date, 'USD_KRW_pred'] = monthly_preds[date]
            
    # 4. Difference Data
    df_diff = pd.DataFrame(index=levels_df.index)
    df_diff['USD_KRW_actual_diff'] = np.log(levels_df['USD_KRW']).diff()
    df_diff['USD_KRW_pred_diff'] = np.log(levels_df['USD_KRW_pred']).diff()
    
    for col in targets:
        if col in ['Trade_Balance', 'Foreign_Stock_Investment', 'Foreign_Bond_Investment']:
            df_diff[col] = levels_df[col].diff()
        else:
            df_diff[col] = np.log(levels_df[col]).diff()
            
    df_diff = df_diff.dropna()
    return df_diff, pred_col

def run_arimax_for_target(target, lag, df_diff, test_obs=24, report_dir=''):
    print(f"\n--- Running ARIMAX for {target} (Lag: {lag}) ---")
    
    # Create shifted exogenous variables
    exog_actual = df_diff[['USD_KRW_actual_diff']].shift(lag).rename(columns={'USD_KRW_actual_diff': 'Exog_Actual'})
    exog_pred = df_diff[['USD_KRW_pred_diff']].shift(lag).rename(columns={'USD_KRW_pred_diff': 'Exog_Pred'})
    
    # Combine and drop NaN (due to shift)
    target_df = pd.concat([df_diff[[target]], exog_actual, exog_pred], axis=1).dropna()
    
    train_df = target_df.iloc[:-test_obs]
    test_df = target_df.iloc[-test_obs:]
    
    endog_train = train_df[target]
    exog_actual_train = train_df[['Exog_Actual']]
    
    endog_test = test_df[target]
    exog_actual_test = test_df[['Exog_Actual']]
    exog_pred_test = test_df[['Exog_Pred']]
    
    # Fit ARIMAX model (Since data is differenced, order=(1,0,1))
    model = ARIMA(endog_train, exog=exog_actual_train, order=(1,0,1))
    fitted = model.fit()
    
    # Forecast with actual FX
    forecast_actual = fitted.forecast(steps=test_obs, exog=exog_actual_test)
    
    # Forecast with predicted FX
    forecast_pred = fitted.forecast(steps=test_obs, exog=exog_pred_test)
    
    # Evaluation
    rmse_act = np.sqrt(mean_squared_error(endog_test, forecast_actual))
    rmse_pred = np.sqrt(mean_squared_error(endog_test, forecast_pred))
    diff_pct = ((rmse_pred - rmse_act) / rmse_act) * 100
    
    print(f"RMSE (Actual FX): {rmse_act:.6f}")
    print(f"RMSE (Pred FX)  : {rmse_pred:.6f} ({diff_pct:+.2f}%)")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, endog_test, label='Actual Target', marker='o', color='black')
    plt.plot(test_df.index, forecast_actual, label='ARIMAX (Actual FX)', marker='x', linestyle='--')
    plt.plot(test_df.index, forecast_pred, label='ARIMAX (Predicted FX)', marker='^', linestyle=':')
    plt.title(f'ARIMAX Error Propagation: {target} (Lag {lag})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(report_dir / f'arimax_{target}.png')
    plt.close()
    
    return rmse_act, rmse_pred, diff_pct

def main():
    parser = argparse.ArgumentParser(description="Run per-target ARIMAX error propagation check.")
    base_dir = Path(__file__).resolve().parents[2]
    default_selected = Path(__file__).resolve().parent / "reports" / "fx_model_selection" / "selected_fx_predictions.csv"
    default_legacy = base_dir / "analysis" / "LSTM" / "Hybrid" / "hybrid_m2" / "eval" / "predictions.csv"
    parser.add_argument("--macro-path", type=Path, default=base_dir / "data" / "integrated_macro_targets.csv")
    parser.add_argument("--predictions-path", type=Path, default=default_selected if default_selected.exists() else default_legacy)
    parser.add_argument("--prediction-column", default=None)
    args = parser.parse_args()

    report_dir = Path(__file__).resolve().parent / "reports" / "arimax"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    df_diff, pred_col = prepare_data(args.macro_path, args.predictions_path, args.prediction_column)
    
    target_lags = {
        'Import_Price_Index': 2,
        'KOSPI': 1,
        'Foreign_Bond_Investment': 1,
        'Foreign_Stock_Investment': 1,
        'Trade_Balance': 6,
        'CPI_KOR': 6
    }
    
    results = {}
    test_obs = 24
    
    for target, lag in target_lags.items():
        r_act, r_pred, diff = run_arimax_for_target(target, lag, df_diff, test_obs, report_dir)
        results[target] = (r_act, r_pred, diff, lag)
        
    with open(report_dir / 'arimax_metrics.txt', 'w') as f:
        f.write(f"ARIMAX Model Forecast Comparison (Test set: {test_obs} months)\n")
        f.write(f"Predictions path: {args.predictions_path}\n")
        f.write(f"Prediction column: {pred_col}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Variable':<25s} | {'Lag':<3s} | {'Actual FX RMSE':<15s} | {'Predicted FX RMSE':<15s} | {'% Diff':<10s}\n")
        f.write("-" * 80 + "\n")
        for var, (r_act, r_pred, diff, lag) in results.items():
            res_str = f"{var:<25s} | {lag:<3d} | {r_act:<15.6f} | {r_pred:<15.6f} | {diff:>8.2f}%"
            f.write(res_str + "\n")

if __name__ == '__main__':
    main()
