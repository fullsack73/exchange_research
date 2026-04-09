import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime

# .env 파일에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("ECOS_API_KEY")

def fetch_ecos_data(stat_code, item_code, start_date, end_date, cycle='M'):
    """
    한국은행 ECOS API를 통해 통계 데이터를 가져옵니다.
    """
    # URL Format: /StatisticSearch/Key/Type/Language/Start/End/StatCode/Cycle/StartDay/EndDay/ItemCode1/ItemCode2/ItemCode3/
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/{stat_code}/{cycle}/{start_date}/{end_date}/{item_code}/"
    
    print(f"Fetching: {stat_code} ({item_code}) from {start_date} to {end_date}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "StatisticSearch" in data:
            rows = data["StatisticSearch"]["row"]
            df = pd.DataFrame(rows)
            # 필요한 컬럼만 선택 및 데이터 정제
            df = df[['TIME', 'DATA_VALUE']]
            df.columns = ['Date', item_code]
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m' if cycle == 'M' else '%Y%m%d')
            df[item_code] = pd.to_numeric(df[item_code])
            return df
        else:
            msg = data.get('RESULT', {}).get('MESSAGE', 'Unknown error')
            print(f"No data or API Error for {item_code}: {msg}")
            return None
    else:
        print(f"HTTP Error: {response.status_code}")
        return None

def main():
    if not API_KEY:
        print("Error: ECOS_API_KEY not found in .env file.")
        return

    # 1. 외국인 증권투자 (BOP 기준)
    # 통계표: 301Y013 (국제수지 월)
    # 항목코드:
    # 1.1 지분증권(부채): BOPF22100000 (외국인 주식 투자)
    # 1.2 부채성증권(부채): BOPF22200000 (외국인 채권 투자)
    
    stat_code = "301Y013"
    items = {
        "BOPF22100000": "Foreign_Stock_Investment",
        "BOPF22200000": "Foreign_Bond_Investment"
    }
    
    start_date = "199501"
    end_date = datetime.now().strftime("%Y%m")
    
    dfs = []
    for item_code, col_name in items.items():
        df = fetch_ecos_data(stat_code, item_code, start_date, end_date)
        if df is not None:
            df = df.rename(columns={item_code: col_name})
            df = df.set_index('Date')
            dfs.append(df)
            
    if dfs:
        final_df = pd.concat(dfs, axis=1)
        
        # 합계 계산 (총 증권투자)
        final_df['Total_Foreign_Investment'] = final_df.sum(axis=1)
        
        # data 폴더 생성
        os.makedirs('data', exist_ok=True)
        
        # CSV 저장
        output_path = 'data/foreign_investment_monthly.csv'
        final_df.to_csv(output_path)
        print(f"\nSuccessfully saved data to {output_path}")
        print(f"Data range: {final_df.index.min().date()} to {final_df.index.max().date()}")
        print(f"Total entries: {len(final_df)}")
        print("\nFirst 5 rows:")
        print(final_df.head())
    else:
        print("Failed to fetch any data.")

if __name__ == "__main__":
    main()
