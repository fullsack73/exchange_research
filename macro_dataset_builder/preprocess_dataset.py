import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv("dataset_raw_updated.csv", index_col="Date", parse_dates=True)

# Define groupings
financial_vars = ["VIX", "DXY", "Policy_Rate_KOR", "Policy_Rate_USA", "Rate_Spread_KOR_USA", "USD_KRW"]
diff_vars = ["Trade_Balance", "Current_Account", "Policy_Rate_KOR", "Policy_Rate_USA", "Rate_Spread_KOR_USA", "Unemployment_KOR"] # Use diff rather than pct_change
pct_vars = [c for c in df.columns if c not in diff_vars]

log_messages = []
log_messages.append("=== Preprocessing Log ===")
log_messages.append("1. Missing Value Imputation (Max 1-Month ffill allowed for Non-Financial Only)")

# Imputation
for col in df.columns:
    if col not in financial_vars:
        gaps_before = df[col].isna().sum()
        df[col] = df[col].ffill(limit=1)
        gaps_after = df[col].isna().sum()
        if gaps_before > gaps_after:
            log_messages.append(f" - Filled {gaps_before - gaps_after} gaps with ffill=1 for {col}")
            
log_messages.append(" - No Interpolation Policy Applied for Financial variables (Remaining gaps are NA).")

feature_desc = []
df_final = pd.DataFrame(index=df.index)

# Transformation loops
for col in df.columns:
    df_final[f"{col}"] = df[col]
    feature_desc.append({"Variable": f"{col}", "Type": "Raw", "Description": f"{col} original measurement"})
    
    # Generate MoM and YoY
    if col in diff_vars:
        df_final[f"{col}_MoM"] = df[col].diff(1)
        df_final[f"{col}_YoY"] = df[col].diff(12)
        feature_desc.append({"Variable": f"{col}_MoM", "Type": "MoM_Diff", "Description": "1-month base difference"})
        feature_desc.append({"Variable": f"{col}_YoY", "Type": "YoY_Diff", "Description": "12-month base difference"})
    else:
        df_final[f"{col}_MoM"] = df[col].pct_change(1)
        df_final[f"{col}_YoY"] = df[col].pct_change(12)
        feature_desc.append({"Variable": f"{col}_MoM", "Type": "MoM_Pct", "Description": "1-month percentage change"})
        feature_desc.append({"Variable": f"{col}_YoY", "Type": "YoY_Pct", "Description": "12-month percentage change"})
        
    # Generate lags for the derivatives
    for transform in ["MoM", "YoY"]:
        col_name = f"{col}_{transform}"
        for lag in [1, 3, 6]:
            lag_name = f"{col_name}_lag{lag}"
            df_final[lag_name] = df_final[col_name].shift(lag)
            feature_desc.append({"Variable": lag_name, "Type": "Lag", "Description": f"{lag}-month lag of {col_name}"})

# FX Surge Threshold Labeling (Top 10% MoM)
fx_mom = df_final["USD_KRW_MoM"]
threshold_90 = fx_mom.quantile(0.9)
# Compute boolean, but where FX_mom is NA, keep it as NA
df_final["FX_Surge"] = (fx_mom >= threshold_90).astype(float)
df_final.loc[fx_mom.isna(), "FX_Surge"] = np.nan
feature_desc.append({"Variable": "FX_Surge", "Type": "Target", "Description": f"1 if USD_KRW_MoM >= {threshold_90:.4f} (Top 10%), else 0"})

# Abnormal Periods Mapping
def is_abnormal(dt):
    if pd.Timestamp('1997-01-01') <= dt <= pd.Timestamp('1998-12-31'): return 1
    if pd.Timestamp('2008-01-01') <= dt <= pd.Timestamp('2009-12-31'): return 1
    if pd.Timestamp('2020-01-01') <= dt <= pd.Timestamp('2021-06-30'): return 1
    if pd.Timestamp('2024-01-01') <= dt <= pd.Timestamp('2026-03-31'): return 1
    return 0

df_final["Is_Abnormal_Period"] = df_final.index.map(is_abnormal)
df_final["Period_Type"] = df_final["Is_Abnormal_Period"].map({1: "Abnormal", 0: "Normal"})
feature_desc.append({"Variable": "Is_Abnormal_Period", "Type": "Label", "Description": "1 if identified as abnormal sequence, else 0"})
feature_desc.append({"Variable": "Period_Type", "Type": "Label", "Description": "String classification (Abnormal vs Normal)"})

# Export
for_drop = df_final.isna().sum()
log_messages.append("\n2. Data Dimensionality & Information Drops:")
log_messages.append(f" - Original shape: {df.shape}")
log_messages.append(f" - Final processed shape: {df_final.shape} (Due to 12-month YoY + 6-month Lag, mathematically 18 initial rows will strictly have NAs on lagging components)")

log_messages.append("\n3. Missing / Invalid Value Presence in Key Targets:")
log_messages.append(f" - FX_Surge NaN Count: {df_final['FX_Surge'].isna().sum()}")

pd.DataFrame(feature_desc).to_csv("feature_description.csv", index=False, encoding='utf-8-sig')
df_final.to_csv("dataset_processed.csv")

with open("preprocess_log.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(log_messages))

print("\n".join(log_messages))
print(f"\nProcessing successful. Written files to dataset_processed.csv and feature_description.csv")
