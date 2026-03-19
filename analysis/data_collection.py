import pandas as pd
import requests
import io
import os

os.makedirs('data', exist_ok=True)

series_dict = {
    'Headline': 'CPIAUCSL',
    'Core': 'CPILFESL',
    'Shelter': 'CUSR0000SAH1',
    'Food': 'CPIFABSL',
    'Energy': 'CPIENGSL',
    'MedicalCare': 'CPIMEDSL',
    'Transportation': 'CPITRNSL',
    'Apparel': 'CPIAPPSL',
    'Durables': 'CUSR0000SAD',
    'Services': 'CUSR0000SAS',
    'KRWUSD': 'DEXKOUS' 
}

def download_fred_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"Downloading {series_id}...")
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True)
    else:
        print(f"Failed to download {series_id}: Status {response.status_code}")
        return None

all_dfs = []
for name, series_id in series_dict.items():
    df = download_fred_csv(series_id)
    if df is not None:
        df.columns = [name]
        all_dfs.append(df)

if all_dfs:
    final_df = pd.concat(all_dfs, axis=1)
    
    # Save daily KRWUSD
    krwusd_daily = final_df[['KRWUSD']].copy().dropna()
    krwusd_daily.to_csv('data/krwusd_daily.csv')
    
    # Save monthly CPI (resampled to Month Start)
    cpi_cols = [c for c in final_df.columns if c != 'KRWUSD']
    cpi_monthly = final_df[cpi_cols].resample('MS').first().dropna(how='all')
    cpi_monthly.to_csv('data/us_cpi_monthly.csv')
    
    print("Data collection completed successfully via direct download.")
    print(f"Files saved in data/ directory. CPI rows: {len(cpi_monthly)}")
else:
    print("No data was downloaded.")
