import pandas as pd
import numpy as np
import os

def prep_monthly_data():
    out_dir = 'analysis/lstm_validation_monthly'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load all monthly processing outputs
    ex_rate = pd.read_csv('data/exchange_rate/exchange_rate_processed.csv')
    spread = pd.read_csv('data/policy_rate/spread_KOR_USA_processed.csv')
    m2_total = pd.read_csv('data/m2/KOR/M2_KOR_processed.csv')
    m2_details = pd.read_csv('data/m2/KOR/M2_details_processed.csv')
    
    # We will use M2_MMF_지분 from M2 details
    # Let's standardize date column
    for df in [ex_rate, spread, m2_total, m2_details]:
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        
    df = ex_rate.merge(spread, on='observation_date', how='inner')
    df = df.merge(m2_total, on='observation_date', how='inner')
    df = df.merge(m2_details[['observation_date', 'M2_MMF_지분']], on='observation_date', how='inner')
    
    # Sort and clean
    df.sort_values('observation_date', inplace=True)
    df.dropna(subset=['USD_KRW', 'RATE_SPREAD_KOR_USA', 'M2_KOR', 'M2_MMF_지분'], inplace=True)
    
    df.to_csv(f'{out_dir}/monthly_dataset.csv', index=False)
    print(f"Monthly dataset created: {len(df)} rows.")

if __name__ == '__main__':
    prep_monthly_data()
