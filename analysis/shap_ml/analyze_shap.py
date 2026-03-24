import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from functools import reduce
from matplotlib import font_manager, rc
import platform
import json
from pathlib import Path

# 운영체제에 따른 한글 폰트 설정
if platform.system() == 'Darwin': # Mac 환경
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    base_path = '/Applications/dollar_price'
    files = {
        'USD_KRW': 'data/exchange_rate/exchange_rate_processed.csv',
        'SPREAD_POLICY': 'data/policy_rate/spread_KOR_USA_processed.csv',
        'BOND_KOR': 'data/10y_bond/KOR/10y_bond_KOR_processed.csv',
        'BOND_USA': 'data/10y_bond/USA/GS10.csv',
        'M2_KOR': 'data/m2/KOR/M2_KOR_processed.csv',
        'M2_USA': 'data/m2/USA/M2SL.csv',
        'CPI_KOR': 'data/CPI/KOR/CPI_KOR_processed.csv',
        'CPI_USA': 'data/CPI/USA/CPIAUCSL.csv',
        'IPI_KOR': 'data/production_index/KOR/IPI_KOR_processed.csv',
        'IPI_USA': 'data/production_index/USA/INDPRO.csv'
    }

    dfs = []
    for name, path in files.items():
        try:
            full_path = f"{base_path}/{path}"
            df = pd.read_csv(full_path)
            
            if 'observation_date' not in df.columns:
                continue
            df['observation_date'] = pd.to_datetime(df['observation_date'])
            
            val_col = [c for c in df.columns if c != 'observation_date'][0]
            df = df.rename(columns={val_col: name})
            
            dfs.append(df[['observation_date', name]])
        except Exception as e:
            print(f"Warning: Failed to load {path} - {e}")

    # 데이터 병합 (Inner Join)
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='observation_date', how='inner'), dfs)

    # 10Y Spread 계산
    if 'BOND_KOR' in df_merged.columns and 'BOND_USA' in df_merged.columns:
        df_merged['SPREAD_10Y'] = df_merged['BOND_KOR'] - df_merged['BOND_USA']

    return df_merged.sort_values('observation_date')

def analyze_shap_drivers(df):
    target_col = 'USD_KRW'
    feature_cols = [
        'SPREAD_POLICY', 'SPREAD_10Y',
        'M2_KOR', 'M2_USA',
        'CPI_KOR', 'CPI_USA',
        'IPI_KOR', 'IPI_USA'
    ]
    
    period_path = Path('/Applications/dollar_price/analysis/anomaly/dynamic_periods.json')
    with open(period_path, 'r', encoding='utf-8') as f:
        period_info = json.load(f)

    cutoff_date = pd.to_datetime(period_info['primary_anomaly_period']['start'])
    anomaly_end = pd.to_datetime(period_info['primary_anomaly_period']['end'])
    
    df_train = df[df['observation_date'] < cutoff_date].dropna()
    df_test = df[(df['observation_date'] >= cutoff_date) & (df['observation_date'] <= anomaly_end)].dropna()
    
    print(f"\n[데이터셋]")
    print(f"학습용(Train): ~ {df_train['observation_date'].max().date()} (n={len(df_train)})")
    print(f"분석용(Test): {df_test['observation_date'].min().date()} ~ {df_test['observation_date'].max().date()} (n={len(df_test)})")
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    
    X_test = df_test[feature_cols]
    
    # 2. Random Forest 학습
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    # 2.5 Random Forest Feature Importance (Anomaly 기간) bar chart
    rf_importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=rf_importances, hue='Feature', palette='viridis', legend=False)
    for i, row in enumerate(rf_importances.itertuples()):
        plt.text(row.Importance, i, f' {row.Importance:.3f}', va='center')
    plt.title(f'Random Forest Feature Importance (Anomaly Period: {df_test["observation_date"].min().date()} ~ {df_test["observation_date"].max().date()})')
    plt.xlabel('Importance')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('rf_importance_anomaly.png')
    print(f"\n[알림] RF Feature Importance 차트가 'rf_importance_anomaly.png'로 저장되었습니다.")
    print("\n[Random Forest 변수 중요도 (Anomaly)]")
    print(rf_importances.to_string(index=False))

    # 3. SHAP 값 계산 (Anomaly 기간에 대해서)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # 4. 시각화: 어떤 변수가 환율을 밀어올렸나?
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title(f"SHAP Summary Plot (Anomaly Period: {df_test['observation_date'].min().date()} ~ {df_test['observation_date'].max().date()})")
    plt.tight_layout()
    plt.savefig('shap_summary_anomaly.png')
    print(f"\n[알림] SHAP Summary Plot이 'shap_summary_anomaly.png'로 저장되었습니다.")
    
    # 5. 시각화: 기여도 막대 그래프 (평균 절대 기여도)
    # Anomaly 기간 동안 각 변수가 환율 변동에 기여한 평균 크기(원)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df_shap = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_SHAP_Impact': mean_abs_shap
    }).sort_values(by='Mean_SHAP_Impact', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Mean_SHAP_Impact', y='Feature', data=df_shap, palette='viridis')
    plt.title('Anomaly 기간 환율 변동 기여도 (평균 |SHAP|)')
    plt.xlabel('평균 영향력 (원)')
    plt.tight_layout()
    plt.savefig('shap_impact_bar.png')
    print(f"\n[알림] SHAP 기여도 차트가 'shap_impact_bar.png'로 저장되었습니다.")
    
    print("\n[Anomaly 기간 주요 환율 상승/하락 요인]")
    print(df_shap)

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        analyze_shap_drivers(df)
