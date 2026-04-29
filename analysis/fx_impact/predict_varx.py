import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import matplotlib.pyplot as plt

def prepare_data(macro_filepath, preds_filepath):
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
    
    monthly_preds = preds_df['pred_model_b'].resample('ME').mean()
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
    
    # Create Lagged Exogenous Variables (Lags 1 to 6)
    exog_actual_cols = []
    exog_pred_cols = []
    
    for lag in range(1, 7):
        col_act = f'Exog_Actual_Lag{lag}'
        col_pred = f'Exog_Pred_Lag{lag}'
        df_diff[col_act] = df_diff['USD_KRW_actual_diff'].shift(lag)
        df_diff[col_pred] = df_diff['USD_KRW_pred_diff'].shift(lag)
        exog_actual_cols.append(col_act)
        exog_pred_cols.append(col_pred)
        
    df_diff = df_diff.dropna()
    return df_diff, targets, exog_actual_cols, exog_pred_cols

def evaluate_forecast(actual, forecast, variables):
    results = {}
    for i, var in enumerate(variables):
        rmse = np.sqrt(mean_squared_error(actual[:, i], forecast[:, i]))
        mae = mean_absolute_error(actual[:, i], forecast[:, i])
        results[var] = {'RMSE': rmse, 'MAE': mae}
    return results

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_dir = os.path.join(os.path.dirname(__file__), 'reports/varx')
    os.makedirs(report_dir, exist_ok=True)
    
    macro_filepath = os.path.join(base_dir, 'data/integrated_macro_targets.csv')
    preds_filepath = os.path.join(base_dir, 'analysis/LSTM/Hybrid/hybrid_m2/eval/predictions.csv')
    
    if not os.path.exists(macro_filepath) or not os.path.exists(preds_filepath):
        print("Required files not found.")
        return
        
    df_diff, targets, exog_actual_cols, exog_pred_cols = prepare_data(macro_filepath, preds_filepath)
    
    # Train-test split (Test on last 24 months)
    test_obs = 24
    train_df = df_diff.iloc[:-test_obs]
    test_df = df_diff.iloc[-test_obs:]
    
    endog_train = train_df[targets]
    exog_actual_train = train_df[exog_actual_cols]
    
    endog_test = test_df[targets]
    exog_actual_test = test_df[exog_actual_cols]
    exog_pred_test = test_df[exog_pred_cols]
    
    print(f"Training VARX model on {len(train_df)} observations. Exog lags: 1-6.")
    
    # Fit VAR with actual USD_KRW (multiple lags) as exogenous
    model = VAR(endog_train, exog=exog_actual_train)
    
    # Let VAR select the best endogenous lag (up to 3 to prevent overfitting with many exogs)
    lag_selection = model.select_order(maxlags=3)
    best_lag = lag_selection.aic
    print(f"Selected Endogenous Lag Order (AIC): {best_lag}")
    
    var_model = model.fit(best_lag)
    
    # Forecast with Actual FX Lags
    lagged_values = endog_train.values[-best_lag:] if best_lag > 0 else None
    forecast_actual = var_model.forecast(y=lagged_values, steps=test_obs, exog_future=exog_actual_test.values)
    
    # Forecast with Predicted FX Lags
    forecast_pred = var_model.forecast(y=lagged_values, steps=test_obs, exog_future=exog_pred_test.values)
    
    # Evaluate
    actual = endog_test.values
    metrics_actual = evaluate_forecast(actual, forecast_actual, targets)
    metrics_pred = evaluate_forecast(actual, forecast_pred, targets)
    
    # Save results
    with open(os.path.join(report_dir, 'varx_metrics.txt'), 'w') as f:
        f.write(f"VARX Model Forecast Comparison (Test set: {test_obs} months)\n")
        f.write(f"Endogenous Lag: {best_lag}, Exogenous Lags: 1-6\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Variable':<25s} | {'Actual FX RMSE':<15s} | {'Predicted FX RMSE':<15s} | {'% Diff':<10s}\n")
        f.write("-" * 75 + "\n")
        
        for var in targets:
            rmse_act = metrics_actual[var]['RMSE']
            rmse_pred = metrics_pred[var]['RMSE']
            diff_pct = ((rmse_pred - rmse_act) / rmse_act) * 100
            
            res_str = f"{var:<25s} | {rmse_act:<15.6f} | {rmse_pred:<15.6f} | {diff_pct:>8.2f}%"
            print(res_str)
            f.write(res_str + "\n")
            
    # Plot comparison
    for idx, var in enumerate(targets):
        plt.figure(figsize=(10, 5))
        plt.plot(test_df.index, actual[:, idx], label='Actual Target', marker='o', color='black')
        plt.plot(test_df.index, forecast_actual[:, idx], label='VARX (Actual FX)', marker='x', linestyle='--')
        plt.plot(test_df.index, forecast_pred[:, idx], label='VARX (Predicted FX)', marker='^', linestyle=':')
        plt.title(f'VARX Error Propagation: {var}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(report_dir, f'varx_{var}.png'))
        plt.close()
        
    print(f"\nVARX modeling complete. Results saved in '{report_dir}'.")

if __name__ == '__main__':
    main()
