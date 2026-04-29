import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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
    return df_diff

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
    plt.savefig(os.path.join(report_dir, f'arimax_{target}.png'))
    plt.close()
    
    return rmse_act, rmse_pred, diff_pct

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_dir = os.path.join(os.path.dirname(__file__), 'reports/arimax')
    os.makedirs(report_dir, exist_ok=True)
    
    macro_filepath = os.path.join(base_dir, 'data/integrated_macro_targets.csv')
    preds_filepath = os.path.join(base_dir, 'analysis/LSTM/Hybrid/hybrid_m2/eval/predictions.csv')
    
    df_diff = prepare_data(macro_filepath, preds_filepath)
    
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
        
    with open(os.path.join(report_dir, 'arimax_metrics.txt'), 'w') as f:
        f.write(f"ARIMAX Model Forecast Comparison (Test set: {test_obs} months)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Variable':<25s} | {'Lag':<3s} | {'Actual FX RMSE':<15s} | {'Predicted FX RMSE':<15s} | {'% Diff':<10s}\n")
        f.write("-" * 80 + "\n")
        for var, (r_act, r_pred, diff, lag) in results.items():
            res_str = f"{var:<25s} | {lag:<3d} | {r_act:<15.6f} | {r_pred:<15.6f} | {diff:>8.2f}%"
            f.write(res_str + "\n")

if __name__ == '__main__':
    main()
