import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 데이터 로드
fx_daily = pd.read_csv('../../data/krwusd_daily.csv', index_col=0, parse_dates=True)
release_dates = pd.read_csv('../../data/cpi_release_dates.csv', parse_dates=['release_date'])
cpi_monthly = pd.read_csv('../../data/us_cpi_monthly.csv', index_col=0, parse_dates=True)

# 일별 수익률 계산
fx_daily['Return'] = fx_daily['KRWUSD'].pct_change()

# 이벤트 분석 결과 저장할 리스트
event_results = []

for idx, row in release_dates.iterrows():
    r_date = row['release_date']
    ref_month = row['reference_month']
    
    # 발표일 환율 반응 (당일 수익률)
    if r_date in fx_daily.index:
        fx_reaction = fx_daily.loc[r_date, 'Return']
        fx_price = fx_daily.loc[r_date, 'KRWUSD']
        
        # 이전 5일 평균 변동성 대비 당일 변동성 비교를 위해
        prev_5d = fx_daily.loc[:r_date].iloc[-6:-1]
        vol_ratio = abs(fx_reaction) / prev_5d['Return'].abs().mean() if not prev_5d.empty else 1.0
        
        # 해당 월의 CPI 데이터 (Surprise 대용으로 Core MoM 사용)
        # reference_month는 '2025-01' 형식이므로 해당 월 첫날로 맞춰서 찾음
        ref_dt = pd.to_datetime(ref_month + "-01")
        if ref_dt in cpi_monthly.index:
            core_mom = cpi_monthly.loc[ref_dt, 'Core'] / cpi_monthly.shift(1).loc[ref_dt, 'Core'] - 1
            headline_mom = cpi_monthly.loc[ref_dt, 'Headline'] / cpi_monthly.shift(1).loc[ref_dt, 'Headline'] - 1
            
            event_results.append({
                'release_date': r_date,
                'fx_reaction': fx_reaction,
                'vol_ratio': vol_ratio,
                'core_mom': core_mom,
                'headline_mom': headline_mom
            })

event_df = pd.DataFrame(event_results)
event_df.to_csv('full/event_study_results.csv', index=False)

# 시각화: Core MoM vs FX Reaction
plt.figure(figsize=(10, 6))
plt.scatter(event_df['core_mom'], event_df['fx_reaction'], alpha=0.6)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.title("US Core CPI MoM vs KRW/USD Reaction on Release Day")
plt.xlabel("Core CPI MoM")
plt.ylabel("KRW/USD Daily Return")
plt.savefig('full/event_study_scatter.png', bbox_inches='
plt.close()

print("Event study completed.")
print(f"Analyzed {len(event_df)} release events.")
