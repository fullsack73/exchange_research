import os
import io
import requests
import pandas as pd
import yfinance as yf
import subprocess

ECOS_API_KEY = "N1R89SK6XDA2XE9XLGVY"
START_DATE = "2026-02-01"
END_DATE = "2026-03-31"

dataset_frames = []
missing_log = []

def fetch_yahoo(ticker, var_name):
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            missing_log.append(f"{var_name}: API Connection/No Data for Ticker {ticker}")
            return
        
        df_m = df['Close'].resample('ME').last().to_frame(name=var_name)
        df_m.index = df_m.index.to_period('M').to_timestamp('M')
        dataset_frames.append(df_m)
    except Exception as e:
        missing_log.append(f"{var_name}: API ERROR - {str(e)}")

def fetch_fred(series_id, var_name):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        result = subprocess.run(["curl.exe", "-s", url], capture_output=True, text=True, check=True)
        temp_df = pd.read_csv(io.StringIO(result.stdout), index_col='observation_date', parse_dates=True)
        temp_df[series_id] = pd.to_numeric(temp_df[series_id], errors='coerce')
        temp_df.rename(columns={series_id: var_name}, inplace=True)
        
        df_m = temp_df.resample('ME').last()
        df_m.index = df_m.index.to_period('M').to_timestamp('M')
        df_m = df_m[(df_m.index >= START_DATE) & (df_m.index <= END_DATE)]
        
        if df_m.empty:
            missing_log.append(f"{var_name}: NO DATA RETURNED FROM FRED (Delayed or Not Found)")
        dataset_frames.append(df_m)
    except Exception as e:
        missing_log.append(f"{var_name}: FRED ERROR - {str(e)}")

def fetch_ecos(stat_code, item_code, var_name, cycle='M'):
    start_str = "202602"
    end_str = "202603"
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}/json/kr/1/1000/{stat_code}/{cycle}/{start_str}/{end_str}/{item_code}/"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'StatisticSearch' in data:
            rows = data['StatisticSearch']['row']
            df = pd.DataFrame(rows)
            df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m')
            df['DATA_VALUE'] = pd.to_numeric(df['DATA_VALUE'], errors='coerce')
            df = df.set_index('TIME')[['DATA_VALUE']].rename(columns={'DATA_VALUE': var_name})
            
            df = df.resample('ME').last()
            df.index = df.index.to_period('M').to_timestamp('M')
            if df.empty:
                missing_log.append(f"{var_name}: NO DATA (ECOS frame empty)")
            dataset_frames.append(df)
        else:
            missing_log.append(f"{var_name}: ECOS Response Missing Data (Not yet published?): {data}")
    except Exception as e:
        missing_log.append(f"{var_name}: ECOS ERROR - {str(e)}")


def update_dataset():
    print("Collecting 2026-02 ~ 2026-03 data...")
    # Yahoo
    fetch_yahoo("KRW=X", "USD_KRW")
    fetch_yahoo("DX-Y.NYB", "DXY")
    fetch_yahoo("^VIX", "VIX")
    fetch_yahoo("CL=F", "WTI_Oil")
    
    # ECOS
    fetch_ecos("901Y118", "T002", "Exports")
    fetch_ecos("901Y118", "T004", "Imports")
    fetch_ecos("301Y017", "SA100", "Trade_Balance")
    fetch_ecos("301Y017", "SA000", "Current_Account")
    fetch_ecos("901Y009", "0", "CPI_KOR")
    fetch_ecos("401Y015", "*AA", "Import_Price_Index")
    fetch_ecos("901Y033", "A00", "Industrial_Production")
    fetch_ecos("101Y004", "BBHA00", "M2")
    fetch_ecos("101Y004", "BBHA04", "MMF")
    fetch_ecos("101Y004", "BBHA02", "Demand_Deposits")
    fetch_ecos("722Y001", "0101000", "Policy_Rate_KOR")
    
    # FRED
    fetch_fred("LRUN64TTKRM156S", "Unemployment_KOR")
    fetch_fred("FEDFUNDS", "Policy_Rate_USA")

    if dataset_frames:
        new_df = pd.concat(dataset_frames, axis=1)
        
        # Calculate rate spread
        if "Policy_Rate_KOR" in new_df.columns and "Policy_Rate_USA" in new_df.columns:
            new_df["Rate_Spread_KOR_USA"] = new_df["Policy_Rate_KOR"] - new_df["Policy_Rate_USA"]

        # Base index 2026-02 ~ 2026-03
        base_index = pd.date_range(start="2026-02-28", end="2026-03-31", freq='ME')
        new_df = new_df.reindex(base_index)
        new_df.index.name = "Date"

        # Check existing data shape to align columns strictly
        if not os.path.exists("dataset_raw.csv"):
            print("dataset_raw.csv not found! Aborting.")
            return
            
        old_df = pd.read_csv("dataset_raw.csv", index_col="Date", parse_dates=True)
        # Ensure columns match strict structure constraint precisely
        new_df = new_df.reindex(columns=old_df.columns)
        
        # Register NAs to log
        missing_counts = new_df.isna().sum()
        for col, count in missing_counts.items():
            if count > 0:
                missing_log.append(f"COLUMN '{col}' contains {count} NA(s) during Feb-Mar 2026 (Likely delayed publication)")

        # Append! (Old on top, New on bottom)
        updated_df = pd.concat([old_df, new_df])
        
        # Output save
        updated_df.to_csv("dataset_raw_updated.csv")
        
        # Write Log
        with open("update_log.txt", "w", encoding='utf-8') as f:
            f.write("=== Dataset Update Log (2026-02 to 2026-03) ===\n")
            f.write(f"Appended Rows: {len(base_index)}\n")
            f.write(f"Index extent added: {base_index[0].date()} to {base_index[-1].date()}\n\n")
            f.write("Failure / Missing variable records (Blank/NA Handling):\n")
            if not missing_log:
                f.write("- All variables fully collected without any NA delays.\n")
            else:
                for log in missing_log:
                    f.write(f"- {log}\n")
                    
        print(f"Successfully generated dataset_raw_updated.csv ({updated_df.shape}) and update_log.txt")
    else:
        print("No new data fetched.")

if __name__ == "__main__":
    update_dataset()
