import pandas as pd
import numpy as np
import os

def prep_daily_data():
    out_dir = 'analysis/lstm_validation_daily'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load M2 and Exchange Rate (already merged daily)
    daily_liq = pd.read_csv('data/m2/KOR/merged_daily_liquid.csv')
    daily_liq['observation_date'] = pd.to_datetime(daily_liq['observation_date'])
    
    # Load and interpolate spread
    spread_df = pd.read_csv('data/policy_rate/spread_KOR_USA_processed.csv')
    spread_df['observation_date'] = pd.to_datetime(spread_df['observation_date'])
    
    # Merge on observation_date
    # Since spread is monthly (e.g., 2010-12-01), we can merge and forward-fill or interpolate
    df = pd.merge(daily_liq, spread_df, on='observation_date', how='left')
    df.sort_values('observation_date', inplace=True)
    df['RATE_SPREAD_KOR_USA'] = df['RATE_SPREAD_KOR_USA'].interpolate(method='linear').ffill().bfill()
    
    # Drop NAs
    df.dropna(subset=['USD_KRW', 'MMF_total', 'M2_proxy', 'RATE_SPREAD_KOR_USA'], inplace=True)
    
    df.to_csv(f'{out_dir}/daily_dataset.csv', index=False)
    print(f"Daily dataset created: {len(df)} rows.")

if __name__ == '__main__':
    prep_daily_data()
