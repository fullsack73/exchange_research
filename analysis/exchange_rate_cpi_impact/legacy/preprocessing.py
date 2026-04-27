import pandas as pd
import numpy as np

# 데이터 로드
cpi_df = pd.read_csv('../../data/us_cpi_monthly.csv', index_col=0, parse_dates=True)
fx_df = pd.read_csv('../../data/krwusd_daily.csv', index_col=0, parse_dates=True)

# 1. 환율 전처리 (월간 변화율 및 급등 구간 정의)
# 월평균 환율 계산
fx_monthly = fx_df.resample('MS').mean()
# 월간 수익률 (Log Return or Pct Change)
fx_monthly['FX_Ret'] = fx_monthly['KRWUSD'].pct_change()

# 환율 급등 구간 정의 (평균 + 1.5 표준편차 초과인 달을 'Spike'로 정의)
threshold = fx_monthly['FX_Ret'].mean() + 1.5 * fx_monthly['FX_Ret'].std()
fx_monthly['Is_Spike'] = (fx_monthly['FX_Ret'] > threshold).astype(int)


# 2. CPI 전처리 (YoY, MoM)
cpi_features = pd.DataFrame(index=cpi_df.index)
for col in cpi_df.columns:
    # MoM (Month-over-Month)
    cpi_features[f'{col}_MoM'] = cpi_df[col].pct_change()
    # YoY (Year-over-Year)
    cpi_features[f'{col}_YoY'] = cpi_df[col].pct_change(12)

# 3. Lag 변수 생성 (1~3개월 시차)
# CPI 발표는 1개월 지연되므로, 실제 환율에 영향을 주는 것은 과거 CPI 데이터임
lagged_features = []
for col in cpi_features.columns:
    for lag in [1, 2, 3]:
        cpi_features[f'{col}_lag{lag}'] = cpi_features[col].shift(lag)

# 4. 데이터 병합
# 환율 타겟을 t시점, CPI 피쳐를 t시점(이미 lag 적용됨)으로 결합
# 사실 CPI는 발표일 기준 이벤트 스터디가 필요하므로, 월간 분석에서는 t시점 환율 수익률을 예측하는 t-1, t-2, t-3 CPI를 사용함.
final_df = pd.merge(fx_monthly[['FX_Ret', 'Is_Spike']], cpi_features, left_index=True, right_index=True, how='inner')

# 결측치 제거
final_df = final_df.dropna()

# 저장
final_df.to_csv('../../data/final_processed_data.csv')

print("Preprocessing completed.")
print(f"Final dataset shape: {final_df.shape}")
print(f"Number of spikes identified: {final_df['Is_Spike'].sum()}")
