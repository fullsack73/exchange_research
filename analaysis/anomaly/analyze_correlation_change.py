import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from functools import reduce
from matplotlib import font_manager, rc
import platform

# 운영체제에 따른 한글 폰트 설정
if platform.system() == 'Darwin': # Mac 환경
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    base_path = '/Applications/dollar_price'
    files = {
        'USD_KRW': 'exchange_rate/exchange_rate_processed.csv',
        'SPREAD_POLICY': 'policy_rate/spread_KOR_USA_processed.csv',
        'BOND_KOR': '10y_bond/KOR/10y_bond_KOR_processed.csv',
        'BOND_USA': '10y_bond/USA/GS10.csv',
        'M2_KOR': 'm2/KOR/M2_KOR_processed.csv',
        'M2_USA': 'm2/USA/M2SL.csv',
        'CPI_KOR': 'CPI/KOR/CPI_KOR_processed.csv',
        'CPI_USA': 'CPI/USA/CPIAUCSL.csv',
        'IPI_KOR': 'production_index/KOR/IPI_KOR_processed.csv',
        'IPI_USA': 'production_index/USA/INDPRO.csv'
    }

    dfs = []
    for name, path in files.items():
        try:
            full_path = f"{base_path}/{path}"
            df = pd.read_csv(full_path)
            
            # 날짜 컬럼 형식 통일
            if 'observation_date' in df.columns:
                df['observation_date'] = pd.to_datetime(df['observation_date'])
            
            # 값 컬럼 이름 변경 (날짜 제외)
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

def analyze_correlation_change(df):
    feature_cols = [
        'SPREAD_POLICY', 'SPREAD_10Y',
        'M2_KOR', 'M2_USA',
        'CPI_KOR', 'CPI_USA',
        'IPI_KOR', 'IPI_USA'
    ]
    target_col = 'USD_KRW'
    
    # 구간 분리
    # Normal: ~ 2024-10-31
    # Anomaly: 2024-11-01 ~ 2026-01-31
    cutoff_start = pd.Timestamp('2024-11-01')
    cutoff_end_excl = pd.Timestamp('2026-02-01')
    
    df_normal = df[df['observation_date'] < cutoff_start]
    df_anomaly = df[(df['observation_date'] >= cutoff_start) & (df['observation_date'] < cutoff_end_excl)]
    
    print(f"\n[구간 정보]")
    print(f"Normal Period: {df_normal['observation_date'].min().date()} ~ {df_normal['observation_date'].max().date()} (n={len(df_normal)})")
    print(f"Anomaly Period: {df_anomaly['observation_date'].min().date()} ~ {df_anomaly['observation_date'].max().date()} (n={len(df_anomaly)})")
    
    # 상관관계 계산
    corr_normal = df_normal[[target_col] + feature_cols].corr()
    corr_anomaly = df_anomaly[[target_col] + feature_cols].corr()
    
    # 시각화 (두 개의 히트맵을 나란히)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    sns.heatmap(corr_normal, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Normal Period Correlation (~24.10)')
    
    sns.heatmap(corr_anomaly, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('Anomaly Period Correlation (24.11~26.01)')
    
    plt.tight_layout()
    plt.savefig('correlation_comparison.png')
    print(f"\n[알림] 상관관계 비교 히트맵이 'correlation_comparison.png'로 저장되었습니다.")
    
    # 이상 기간(Anomaly Period) 단독 히트맵 저장
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_anomaly, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    plt.title('Anomaly Period Correlation (24.11~26.01)')
    plt.tight_layout()
    plt.savefig('correlation_anomaly_only.png')
    print(f"\n[알림] 이상 기간 단독 히트맵이 'correlation_anomaly_only.png'로 저장되었습니다.")
    
    # 변화량 출력
    diff = corr_anomaly[target_col] - corr_normal[target_col]
    print("\n[상관관계 변화 (Anomaly - Normal)]")
    print(diff.drop(target_col).sort_values(ascending=False))

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        analyze_correlation_change(df)
