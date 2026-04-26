import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Load anomaly dataset
data_path = 'analysis/anomaly/anomaly_concatenated_dataset.csv'
df = pd.read_csv(data_path, parse_dates=['date', 'block_start', 'block_end'])
df.sort_values(by=['block_start', 'date'], inplace=True)

print("Total anomaly days:", len(df))

# Compute percentage change (returns) purely within each block to avoid invalid boundary jumps
df['FX_MoM'] = df.groupby('block_start')['FX_rate'].pct_change()
df['M2_MoM'] = df.groupby('block_start')['M2_KOR'].pct_change()

# Drop rows with NaNs (the first day of each block, plus any intrinsic NaNs)
df_clean = df.dropna(subset=['FX_MoM', 'M2_MoM']).copy()
print("Total anomaly days after differencing within blocks:", len(df_clean))

def adf_test(series, title=''):
    res = adfuller(series)
    is_stationary = res[1] <= 0.05
    print(f" ADF {title:<20} | p-val: {res[1]:.4f} | {'Stationary' if is_stationary else 'Non-stationary'}")
    return is_stationary

adf_test(df_clean['M2_MoM'], 'M2 Returns')
adf_test(df_clean['FX_MoM'], 'FX Returns')

# Drop infinites
df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]

# Granger Causality on the stacked "anomaly regime" days
print("\n=== Granger Causality Test: Anomaly Periods (Stacked Regime) ===")
maxlag = 5

try:
    print("\n --- Does FX_rate cause KOR M2? ---")
    res1 = grangercausalitytests(df_clean[['M2_MoM', 'FX_MoM']], maxlag=[1, 2, 3], verbose=False)
    for lag in [1, 2, 3]:
        p_val = res1[lag][0]['ssr_ftest'][1]
        f_val = res1[lag][0]['ssr_ftest'][0]
        print(f" Lag {lag} | p-value: {p_val:.4f} (F={f_val:.4f}) => {'Significant' if p_val < 0.05 else 'Not Significant'}")
        
    print("\n --- Does KOR M2 cause FX_rate? ---")
    res2 = grangercausalitytests(df_clean[['FX_MoM', 'M2_MoM']], maxlag=[1, 2, 3], verbose=False)
    for lag in [1, 2, 3]:
        p_val = res2[lag][0]['ssr_ftest'][1]
        f_val = res2[lag][0]['ssr_ftest'][0]
        print(f" Lag {lag} | p-value: {p_val:.4f} (F={f_val:.4f}) => {'Significant' if p_val < 0.05 else 'Not Significant'}")

except Exception as e:
    print("Granger test failed:", e)

# Let's save a quick summary table
res_summary = {
    'Hypothesis': [],
    'Lag': [],
    'F-Stat': [],
    'p-value': [],
    'Result': []
}

for lag in [1, 2, 3]:
    res_summary['Hypothesis'].append('FX -> M2')
    res_summary['Lag'].append(lag)
    res_summary['F-Stat'].append(res1[lag][0]['ssr_ftest'][0])
    res_summary['p-value'].append(res1[lag][0]['ssr_ftest'][1])
    res_summary['Result'].append('Significant' if res1[lag][0]['ssr_ftest'][1] < 0.05 else 'Not Significant')

    res_summary['Hypothesis'].append('M2 -> FX')
    res_summary['Lag'].append(lag)
    res_summary['F-Stat'].append(res2[lag][0]['ssr_ftest'][0])
    res_summary['p-value'].append(res2[lag][0]['ssr_ftest'][1])
    res_summary['Result'].append('Significant' if res2[lag][0]['ssr_ftest'][1] < 0.05 else 'Not Significant')

pd.DataFrame(res_summary).to_csv('analysis/m2_components/granger_results_anomaly.csv', index=False)


# Now analyzing Normal Periods using the macro dataset
macro_path = 'data/macro_dataset_processed.csv'
macro_df = pd.read_csv(macro_path, parse_dates=['Date'])
macro_df = macro_df.sort_values(by='Date')
print("\nTotal macro days:", len(macro_df))

# Isolate Normal Periods
# We need to find contiguous blocks to avoid jumping across anomalies
macro_df['Block_ID'] = (macro_df['Is_Abnormal_Period'].diff() != 0).cumsum()

# Keep only normal periods (assuming 0 is Normal, or where not abnormal)
# Wait, macro_dataset might have 0 or NaNs
normal_df = macro_df[macro_df['Is_Abnormal_Period'] == 0].copy()
print("Total Normal days:", len(normal_df))

# Compute returns strictly within normal blocks
normal_df['FX_MoM'] = normal_df.groupby('Block_ID')['USD_KRW'].pct_change()
normal_df['M2_MoM'] = normal_df.groupby('Block_ID')['M2'].pct_change()

normal_df_clean = normal_df.dropna(subset=['FX_MoM', 'M2_MoM']).copy()
normal_df_clean = normal_df_clean[~normal_df_clean.isin([np.inf, -np.inf]).any(axis=1)]

print("\n=== Granger Causality Test: Normal Periods (Stacked Regime) ===")
# Filter out tiny blocks smaller than maxlag if they exist, or just let python handle it.
# Actually, groupby pct_change already adds NaN at start of block
try:
    print("\n --- Does FX_rate cause KOR M2? (Normal) ---")
    res1_n = grangercausalitytests(normal_df_clean[['M2_MoM', 'FX_MoM']], maxlag=[1, 2, 3], verbose=False)
    for lag in [1, 2, 3]:
        p_val = res1_n[lag][0]['ssr_ftest'][1]
        f_val = res1_n[lag][0]['ssr_ftest'][0]
        print(f" Lag {lag} | p-value: {p_val:.4f} (F={f_val:.4f}) => {'Significant' if p_val < 0.05 else 'Not Significant'}")
        
    print("\n --- Does KOR M2 cause FX_rate? (Normal) ---")
    res2_n = grangercausalitytests(normal_df_clean[['FX_MoM', 'M2_MoM']], maxlag=[1, 2, 3], verbose=False)
    for lag in [1, 2, 3]:
        p_val = res2_n[lag][0]['ssr_ftest'][1]
        f_val = res2_n[lag][0]['ssr_ftest'][0]
        print(f" Lag {lag} | p-value: {p_val:.4f} (F={f_val:.4f}) => {'Significant' if p_val < 0.05 else 'Not Significant'}")

except Exception as e:
    print("Granger test failed for normal:", e)

# Save result combinations
import json
with open('analysis/m2_components/granger_findings.json', 'w') as f:
    json.dump({
        "anomaly": {
            "M2_causes_FX": {"p_val_lag2": res2[2][0]['ssr_ftest'][1]},
            "FX_causes_M2": {"p_val_lag2": res1[2][0]['ssr_ftest'][1]}
        },
        "normal": {
            "M2_causes_FX": {"p_val_lag2": res2_n[2][0]['ssr_ftest'][1]},
            "FX_causes_M2": {"p_val_lag2": res1_n[2][0]['ssr_ftest'][1]}
        }
    }, f, indent=4)

