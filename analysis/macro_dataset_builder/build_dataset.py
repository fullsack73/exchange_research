import os
import io
import time
import requests
import pandas as pd
import yfinance as yf
import subprocess

ECOS_API_KEY = "N1R89SK6XDA2XE9XLGVY"
START_DATE = "1995-01-01"
END_DATE = "2026-01-31"

data_dict = []
dataset_frames = []

def add_to_dict(var_name, description, source, unit, api_info):
    data_dict.append({
        "Variable": var_name,
        "Description": description,
        "Source": source,
        "Unit": unit,
        "API_Info": api_info
    })

def fetch_yahoo(ticker, var_name, desc, unit):
    print(f"Fetching {ticker} from Yahoo Finance...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            print(f"Warning: No data returned for {ticker}")
            return
        
        # Monthly resample (End of Month)
        df_m = df['Close'].resample('ME').last().to_frame(name=var_name)
        df_m.index = df_m.index.to_period('M').to_timestamp('M') # Align exactly to Month end
        dataset_frames.append(df_m)
        add_to_dict(var_name, desc, "Yahoo Finance", unit, f"Ticker: {ticker}")
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")

def fetch_fred(series_id, var_name, desc, unit):
    print(f"Fetching {series_id} from FRED...")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        result = subprocess.run(["curl.exe", "-s", url], capture_output=True, text=True, check=True)
        temp_df = pd.read_csv(io.StringIO(result.stdout), index_col='observation_date', parse_dates=True)
        temp_df[series_id] = pd.to_numeric(temp_df[series_id], errors='coerce')
        temp_df.rename(columns={series_id: var_name}, inplace=True)
        
        # Monthly resample
        df_m = temp_df.resample('ME').last()
        df_m.index = df_m.index.to_period('M').to_timestamp('M')
        df_m = df_m[(df_m.index >= START_DATE) & (df_m.index <= END_DATE)]
        dataset_frames.append(df_m)
        add_to_dict(var_name, desc, "FRED", unit, f"Series: {series_id}")
    except Exception as e:
        print(f"Failed to fetch {series_id}: {e}")

def fetch_ecos(stat_code, item_code, var_name, desc, unit, cycle='M'):
    print(f"Fetching {stat_code}/{item_code} from ECOS...")
    start_str = START_DATE[:7].replace('-', '') # YYYYMM
    end_str = "202601"
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
            
            # Align end of month
            df = df.resample('ME').last()
            df.index = df.index.to_period('M').to_timestamp('M')
            dataset_frames.append(df)
            add_to_dict(var_name, desc, "ECOS API", unit, f"STAT: {stat_code}, ITEM: {item_code}")
        else:
            print(f"Failed to fetch {var_name}: ECOS Response Error: {data}")
            # Fallback to FRED or other sources if fails? 
            # We'll just leave it and record failure in the console log.
    except Exception as e:
        print(f"Failed to fetch {var_name}: {e}")


def build_dataset():
    fetch_yahoo("KRW=X", "USD_KRW", "원/달러 환율", "KRW")
    
    # 2. 대외부문
    fetch_ecos("901Y118", "T002", "Exports", "수출(통관기준)", "천달러", "M")
    fetch_ecos("901Y118", "T004", "Imports", "수입(통관기준)", "천달러", "M")
    fetch_ecos("301Y017", "SA100", "Trade_Balance", "상품수지(계절조정)", "백만달러", "M")
    fetch_ecos("301Y017", "SA000", "Current_Account", "경상수지(계절조정)", "백만달러", "M")

    # 3. 물가
    fetch_ecos("901Y009", "0", "CPI_KOR", "소비자물가지수(한국)", "2020=100", "M")
    fetch_ecos("401Y015", "*AA", "Import_Price_Index", "수입물가지수", "2020=100", "M")

    # 4. 실물경제
    fetch_ecos("901Y033", "A00", "Industrial_Production", "전산업생산지수", "2020=100", "M")
    # 실업률은 FRED 활용이 깔끔함 (코드 일치성 확보 불량시 대체)
    fetch_fred("LRUN64TTKRM156S", "Unemployment_KOR", "한국 실업률(계절조정)", "%")

    # 5. 금융/통화
    fetch_ecos("101Y004", "BBHA00", "M2", "M2 (평잔, 원계열)", "십억원", "M")
    fetch_ecos("101Y004", "BBHA04", "MMF", "MMF (평잔, 원계열)", "십억원", "M")
    fetch_ecos("101Y004", "BBHA02", "Demand_Deposits", "요구불예금 (평잔, 원계열)", "십억원", "M")
    
    fetch_ecos("722Y001", "0101000", "Policy_Rate_KOR", "한국은행 기준금리", "%", "M")
    fetch_fred("FEDFUNDS", "Policy_Rate_USA", "미국 연준 기준금리", "%")

    # 6. 대외요인
    fetch_yahoo("DX-Y.NYB", "DXY", "미국 달러 인덱스", "Index")
    fetch_yahoo("^VIX", "VIX", "VIX 지수 (변동성 지수)", "Index")
    fetch_yahoo("CL=F", "WTI_Oil", "WTI 국제유가", "USD/bbl")

    # 데이터 병합
    print("\nMerging all data...")
    if dataset_frames:
        final_df = pd.concat(dataset_frames, axis=1)
        final_df = final_df.sort_index()
        
        # 한국-미국 금리차
        if "Policy_Rate_KOR" in final_df.columns and "Policy_Rate_USA" in final_df.columns:
            final_df["Rate_Spread_KOR_USA"] = final_df["Policy_Rate_KOR"] - final_df["Policy_Rate_USA"]
            add_to_dict("Rate_Spread_KOR_USA", "한국-미국 금리차", "Calculated", "%p", "KOR - USA")

        # Base index 1995-01 ~ 2026-01
        base_index = pd.date_range(start="1995-01-31", end="2026-01-31", freq='ME')
        final_df = final_df.reindex(base_index)
        final_df.index.name = "Date"
        
        # Save Raw Dataset
        final_df.to_csv("dataset_raw.csv")
        
        # Save Dictionary
        dict_df = pd.DataFrame(data_dict)
        dict_df.to_csv("data_dictionary.csv", index=False, encoding='utf-8-sig')
        
        print(f"Successfully created dataset_raw.csv with shape {final_df.shape}")
        missing_vars = final_df.isna().sum()
        print("\nMissing values per variable:")
        print(missing_vars[missing_vars > 0])
    else:
        print("No data collected!")

if __name__ == "__main__":
    build_dataset()
