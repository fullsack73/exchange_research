import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, ccf
import os

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    targets = [
        'USD_KRW', 'Import_Price_Index', 'CPI_KOR', 
        'Trade_Balance', 'Foreign_Stock_Investment', 
        'Foreign_Bond_Investment', 'KOSPI'
    ]
    
    # Keep only available targets
    available_targets = [col for col in targets if col in df.columns]
    df = df[available_targets].dropna()
    
    # Take first differences to ensure stationarity (log diffs for indices)
    df_diff = pd.DataFrame(index=df.index)
    for col in available_targets:
        if col in ['Trade_Balance', 'Foreign_Stock_Investment', 'Foreign_Bond_Investment']:
            # May contain negative values, so regular diff
            df_diff[col] = df[col].diff()
        else:
            # Log diffs for price indices and exchange rates
            df_diff[col] = np.log(df[col]).diff()
            
    df_diff = df_diff.dropna()
    return df_diff

def plot_ccf(df, x_col, y_col, max_lag=12, save_dir='reports/lead_lag'):
    # x_col is USD_KRW, y_col is target
    # ccf(x, y) where x is the predictor and y is the target.
    # Positive lag means x leads y. We want to see if USD_KRW leads the targets.
    
    # Calculate cross-correlation
    ccf_vals = ccf(df[x_col], df[y_col], adjusted=False)[:max_lag+1]
    
    # Create lag axis
    lags = np.arange(0, max_lag+1)
    
    plt.figure(figsize=(10, 5))
    plt.vlines(lags, [0], ccf_vals)
    plt.axhline(0, color='black', linewidth=1)
    
    # Approximate 95% confidence interval
    conf_interval = 1.96 / np.sqrt(len(df))
    plt.axhline(conf_interval, color='red', linestyle='--')
    plt.axhline(-conf_interval, color='red', linestyle='--')
    
    plt.title(f'Cross-Correlation: {x_col} leading {y_col}')
    plt.xlabel('Lag (Months)')
    plt.ylabel('CCF')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/ccf_{y_col}.png")
    plt.close()

def run_granger(df, x_col, y_col, max_lag=12, save_dir='reports/lead_lag'):
    print(f"\n--- Granger Causality: {x_col} -> {y_col} ---")
    # grangercausalitytests tests if the second column (x_col) Granger causes the first column (y_col)
    data = df[[y_col, x_col]]
    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # Save results
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/granger_{y_col}.txt", "w") as f:
            for lag, test_res in results.items():
                p_val = test_res[0]['ssr_ftest'][1]
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                res_str = f"Lag {lag}: F-test p-value = {p_val:.4f} {sig}"
                print(res_str)
                f.write(res_str + "\n")
    except Exception as e:
        print(f"Error computing Granger causality for {y_col}: {e}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_dir = os.path.join(os.path.dirname(__file__), 'reports/lead_lag')
    
    filepath = os.path.join(base_dir, 'data/integrated_macro_targets.csv')
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    df_diff = prepare_data(filepath)
    
    targets = [col for col in df_diff.columns if col != 'USD_KRW']
    
    for target in targets:
        plot_ccf(df_diff, 'USD_KRW', target, save_dir=report_dir)
        run_granger(df_diff, 'USD_KRW', target, save_dir=report_dir)
        
    print(f"\nLead-Lag and Causality Analysis complete. Results saved in '{report_dir}'.")

if __name__ == '__main__':
    main()
