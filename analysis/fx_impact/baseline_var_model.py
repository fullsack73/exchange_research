import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import matplotlib.pyplot as plt

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    targets = [
        'USD_KRW', 'Import_Price_Index', 'CPI_KOR', 
        'Trade_Balance', 'Foreign_Stock_Investment', 
        'Foreign_Bond_Investment', 'KOSPI'
    ]
    
    available_targets = [col for col in targets if col in df.columns]
    df = df[available_targets].dropna()
    
    # Store original levels for potential inverse transform later
    levels_df = df.copy()
    
    df_diff = pd.DataFrame(index=df.index)
    for col in available_targets:
        if col in ['Trade_Balance', 'Foreign_Stock_Investment', 'Foreign_Bond_Investment']:
            df_diff[col] = df[col].diff()
        else:
            df_diff[col] = np.log(df[col]).diff()
            
    df_diff = df_diff.dropna()
    return df_diff, levels_df.loc[df_diff.index]

def evaluate_forecast(actual, forecast, variables):
    results = {}
    for i, var in enumerate(variables):
        rmse = np.sqrt(mean_squared_error(actual[:, i], forecast[:, i]))
        mae = mean_absolute_error(actual[:, i], forecast[:, i])
        results[var] = {'RMSE': rmse, 'MAE': mae}
    return results

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_dir = os.path.join(os.path.dirname(__file__), 'reports/baseline')
    
    filepath = os.path.join(base_dir, 'data/integrated_macro_targets.csv')
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    df_diff, _ = prepare_data(filepath)
    
    # Train-test split (Test on last 24 months)
    test_obs = 24
    train_df = df_diff.iloc[:-test_obs]
    test_df = df_diff.iloc[-test_obs:]
    
    print(f"Training VAR model on {len(train_df)} observations. Testing on {len(test_df)} observations.")
    
    # Select best lag order based on AIC
    model = VAR(train_df)
    lag_selection = model.select_order(maxlags=15)
    best_lag = lag_selection.aic
    print(f"Selected lag order (AIC): {best_lag}")
    
    # Fit model
    var_model = model.fit(best_lag)
    
    # Forecast
    lagged_values = train_df.values[-best_lag:]
    forecast = var_model.forecast(y=lagged_values, steps=test_obs)
    
    # Evaluate
    actual = test_df.values
    variables = train_df.columns
    metrics = evaluate_forecast(actual, forecast, variables)
    
    # Save results
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, 'var_metrics.txt'), 'w') as f:
        f.write(f"VAR Model Baseline Evaluation (Test set: {test_obs} months)\n")
        f.write(f"Optimal Lag: {best_lag}\n")
        f.write("-" * 50 + "\n")
        for var, m in metrics.items():
            res_str = f"{var:25s} - RMSE: {m['RMSE']:.6f}, MAE: {m['MAE']:.6f}"
            print(res_str)
            f.write(res_str + "\n")
            
    # Plot forecast vs actual for KOSPI or others
    for idx, var in enumerate(variables):
        plt.figure(figsize=(10, 5))
        plt.plot(test_df.index, actual[:, idx], label='Actual', marker='o')
        plt.plot(test_df.index, forecast[:, idx], label='Forecast', marker='x', linestyle='--')
        plt.title(f'VAR Baseline Forecast vs Actual: {var}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(report_dir, f'var_forecast_{var}.png'))
        plt.close()
        
    print(f"\nBaseline VAR modeling complete. Results saved in '{report_dir}'.")

if __name__ == '__main__':
    main()
