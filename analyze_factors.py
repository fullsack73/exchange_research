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
        'IPI_USA': 'production_index/USA/INDPRO.csv',
        'THEORETICAL_FWD': 'exchange_rate/theoretical_fwd_rate_processed.csv'
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
    return df_merged.sort_values('observation_date')

def analyze(df):
    # 1. 파생 변수 생성: 시장 금리차 (10년물 국채 스프레드)
    # 한국 금리가 높으면 양수, 낮으면 음수
    df['SPREAD_10Y'] = df['BOND_KOR'] - df['BOND_USA']
    
    # 분석 대상 변수 (Features)
    feature_cols = [
        'SPREAD_POLICY', # 정책금리차
        'SPREAD_10Y',    # 시장금리차
        'M2_KOR', 'M2_USA',
        'CPI_KOR', 'CPI_USA',
        'IPI_KOR', 'IPI_USA'
    ]
    
    target_col = 'USD_KRW'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. Random Forest 분석
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    
    # 중요도 추출
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # 3. 상관관계 분석
    corr_matrix = df[[target_col] + feature_cols].corr()
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    
    return df, importances, target_corr, corr_matrix

def visualize_results(importances, target_corr, corr_matrix):
    print("\n========== [분석 결과] 환율(USD/KRW) 영향력 분석 ==========")
    print("\n1. [Random Forest 변수 중요도 (Top 5)]")
    print(importances.head(5).to_string(index=False))
    
    print("\n2. [상관관계 (Pearson Coefficient)]")
    # 양의 상관관계: 이 변수가 오르면 환율도 오름
    # 음의 상관관계: 이 변수가 오르면 환율은 내림
    print(target_corr.to_string())

    # Heatmap 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    plt.title('환율 및 주요 변수 상관관계 히트맵')
    plt.tight_layout()
    
    save_path = 'exchange_rate_heatmap.png'
    plt.savefig(save_path)
    print(f"\n[알림] 히트맵 이미지가 '{save_path}'로 저장되었습니다.")
    plt.show()

def check_theoretical_fwd(df):
    # 이론가와 실제 환율 비교
    # 건덕지 분석: 실제 환율과 이론가의 차이(Spread)가 어떤 의미를 가지는지
    # 혹은 이론가가 환율을 얼마나 잘 설명하는지(R^2)
    
    from sklearn.metrics import r2_score
    
    r2 = r2_score(df['USD_KRW'], df['THEORETICAL_FWD'])
    correlation = df['USD_KRW'].corr(df['THEORETICAL_FWD'])
    
    print("\n========== [이론적 선물환율(Theoretical Fwd) 분석] ==========")
    print(f"설명력(R-squared): {r2:.4f} (1에 가까울수록 실제 환율과 유사함)")
    print(f"상관관계: {correlation:.4f}")
    
    # 괴리율 계산
    df['Deviation'] = df['USD_KRW'] - df['THEORETICAL_FWD']
    print(f"평균 괴리(Actual - Theoretical): {df['Deviation'].mean():.2f} 원")

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df, importances, correlations, corr_matrix = analyze(df)
        visualize_results(importances, correlations, corr_matrix)
        
        if 'THEORETICAL_FWD' in df.columns:
            check_theoretical_fwd(df)
    else:
        print("데이터 로드 실패: 공통 날짜를 가진 데이터가 없습니다.")
