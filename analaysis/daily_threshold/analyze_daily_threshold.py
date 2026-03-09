import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import os
import seaborn as sns

base_dir = '/Applications/dollar_price'

def analyze():
    import matplotlib
    import matplotlib.font_manager as fm
    
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False
        
    df = pd.read_csv(os.path.join(base_dir, 'm2/KOR/merged_daily_liquid.csv'))
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df.sort_values('observation_date', inplace=True)
    
    # Calculate daily differences (using log differences for rate, regular diffs for M2 assuming M2 proxy is an aggregate index)
    df['delta_ER'] = df['USD_KRW'].diff()
    df['delta_M2'] = df['M2_proxy'].diff()
    
    # Drop NAs
    df.dropna(inplace=True)
    
    print(f"Total trading days analyzed: {len(df)}")
    
    # Simple search for the optimal threshold (TAR model)
    # We will search thresholds from the 10th percentile to 90th percentile of positive liquidity shocks
    positive_shocks = df[df['delta_M2'] > 0]['delta_M2']
    if len(positive_shocks) < 10:
        print("Not enough positive shocks for threshold analysis.")
        return
        
    thresholds = np.percentile(positive_shocks, np.linspace(10, 95, 85))
    
    best_t = None
    best_rss = float('inf')
    best_model = None
    
    for t in thresholds:
        # Create dummy indicator for threshold breach
        df['D_tau'] = (df['delta_M2'] > t).astype(int)
        df['interaction'] = df['delta_M2'] * df['D_tau']
        
        # Fit OLS
        X = df[['delta_M2', 'D_tau', 'interaction']]
        X = sm.add_constant(X)
        y = df['delta_ER']
        
        model = sm.OLS(y, X).fit()
        if model.ssr < best_rss:
            best_rss = model.ssr
            best_t = t
            best_model = model
            
    print("\n--- OLS Threshold Regression Results ---")
    print(f"Optimal Threshold (M2 increment): {best_t:.2f}")
    if best_model is not None:
        print(best_model.summary())
    else:
        print("Model failed to converge.")
        return
    
    # Recalculate with best threshold map
    df['D_tau'] = (df['delta_M2'] > best_t).astype(int)
    
    print("\n--- How often did we breach this threshold? ---")
    print(f"Historical (2010-2025): {df['D_tau'].sum()} days out of {len(df)} ({df['D_tau'].mean()*100:.2f}%)")
    
    anomaly = df[(df['observation_date'] >= '2024-11-01') & (df['observation_date'] <= '2025-12-31')]
    if not anomaly.empty:
        print(f"Anomaly Period (Nov 24-Dec 25): {anomaly['D_tau'].sum()} days out of {len(anomaly)} ({anomaly['D_tau'].mean()*100:.2f}%)")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot normal regime
    normal = df[df['D_tau'] == 0]
    sns.regplot(x='delta_M2', y='delta_ER', data=normal, scatter_kws={'alpha':0.3, 'color':'blue'}, line_kws={'color':'blue'}, label='Normal Regime')
    
    # Plot jump regime
    jump = df[df['D_tau'] == 1]
    sns.regplot(x='delta_M2', y='delta_ER', data=jump, scatter_kws={'alpha':0.6, 'color':'red'}, line_kws={'color':'red'}, label='Jump Regime (Threshold breach)')
    
    plt.axvline(x=best_t, color='black', linestyle='--', label=f'Threshold ($\tau$ = {best_t:.2f})')
    plt.title('Non-linear Threshold Effect: Daily Liquidity (M2 Proxy) vs FX Jump')
    plt.xlabel('Daily Change in Short-Term Liquidity (\u0394 M2 Proxy)')
    plt.ylabel('Daily Change in USD/KRW (\u0394 Exchange Rate)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_graph = os.path.join(base_dir, 'threshold_regression_daily.png')
    plt.savefig(out_graph, dpi=150)
    plt.close()
    print(f"Saved visualization to {out_graph}")

if __name__ == "__main__":
    analyze()
