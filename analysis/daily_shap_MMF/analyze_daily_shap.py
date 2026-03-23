import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os
import platform
from matplotlib import font_manager, rc

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

base_dir = '/Applications/dollar_price'

def analyze_daily_shap():
    # Load dataset
    df = pd.read_csv(os.path.join(base_dir, 'data/m2/KOR/merged_daily_liquid.csv'))
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df.sort_values('observation_date', inplace=True)
    
    # Calculate daily differences
    df['delta_ER'] = df['USD_KRW'].diff()
    df['delta_CMA'] = df['CMA_total'].diff()
    df['delta_MMF'] = df['MMF_total'].diff()
    df.dropna(inplace=True)
    
    # Feature columns
    feature_cols = ['delta_CMA', 'delta_MMF']
    target_col = 'delta_ER'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Training RandomForest on {len(X)} daily samples...")
    rf = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
    rf.fit(X, y)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.title("Daily Liquidity SHAP Summary Plot")
    plt.tight_layout()
    out_summary = os.path.join(base_dir, 'analysis/daily_shap/daily_shap_summary.png')
    plt.savefig(out_summary)
    plt.close()
    
    # 2. SHAP Dependence Plot for delta_MMF
    plt.figure(figsize=(8, 6))
    shap.dependence_plot('delta_MMF', shap_values, X, feature_names=feature_cols, show=False, interaction_index=None)
    plt.title("SHAP Dependence: Daily MMF \u0394 vs Ex.Rate Impact (\u0394 KRW)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_dep_mmf = os.path.join(base_dir, 'analysis/daily_shap/daily_shap_dependence_mmf.png')
    plt.savefig(out_dep_mmf)
    plt.close()
    
    # 3. SHAP Dependence Plot for delta_CMA
    plt.figure(figsize=(8, 6))
    shap.dependence_plot('delta_CMA', shap_values, X, feature_names=feature_cols, show=False, interaction_index=None)
    plt.title("SHAP Dependence: Daily CMA \u0394 vs Ex.Rate Impact (\u0394 KRW)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_dep_cma = os.path.join(base_dir, 'analysis/daily_shap/daily_shap_dependence_cma.png')
    plt.savefig(out_dep_cma)
    plt.close()

    print("SHAP analysis completed. Graphs saved.")

if __name__ == "__main__":
    os.makedirs(os.path.join(base_dir, 'analysis/daily_shap'), exist_ok=True)
    analyze_daily_shap()
