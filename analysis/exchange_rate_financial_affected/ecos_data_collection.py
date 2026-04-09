import os
import pandas as pd
import requests
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("ECOS_API_KEY")

def fetch_ecos_data(stat_code, item_code1, start_date, end_date):
    """
    한국은행 ECOS API를 통해 통계 데이터를 가져옵니다.
    """
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/{stat_code}/M/{start_date}/{end_date}/{item_code1}/"
    
    print(f"Fetching data from ECOS: {stat_code} ({item_code1}) from {start_date} to {end_date}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "StatisticSearch" in data:
            rows = data["StatisticSearch"]["row"]
            df = pd.DataFrame(rows)
            # 필요한 컬럼만 선택 및 이름 변경
            df = df[['TIME', 'DATA_VALUE', 'ITEM_NAME1']]
            df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m')
            df['DATA_VALUE'] = pd.to_numeric(df['DATA_VALUE'])
            return df
        else:
            print(f"No data found or API Error: {data.get('RESULT', {}).get('MESSAGE', 'Unknown error')}")
            return None
    else:
        print(f"HTTP Error: {response.status_code}")
        return None

def main():
    if not API_KEY:
        print("Error: ECOS_API_KEY not found in .env file.")
        return

    # 국제수지 (BOP) 테이블 코드: 301Y013
    # 경상수지 항목 코드: 000000
    stat_code = "301Y013"
    items = {
        "000000": "CurrentAccount", # 경상수지
        "100000": "GoodsAccount", # 상품수지
        "200000": "ServicesAccount", # 서비스수지
        "300000": "PrimaryIncome", # 본원소득수지
        "400000": "SecondaryIncome" # 이전소득수지
    }
    
    start_date = "196001"
    end_date = "202612"  # 현재 시점까지 최대한 가져오기 위해 넉넉히 설정
    
    all_series = []
    
    for item_code, item_name in items.items():
        df = fetch_ecos_data(stat_code, item_code, start_date, end_date)
        if df is not None:
            df = df.rename(columns={'DATA_VALUE': item_name})
            df = df.set_index('TIME')[[item_name]]
            all_series.append(df)
    
    if all_series:
        final_df = pd.concat(all_series, axis=1)
        
        # data 폴더 생성
        os.makedirs('data', exist_ok=True)
        
        # CSV 저장
        output_path = 'data/ecos_bop_monthly.csv'
        final_df.to_csv(output_path)
        print(f"Successfully saved BOP data to {output_path}")
        print(f"Data range: {final_df.index.min()} to {final_df.index.max()}")
        print(f"Total rows: {len(final_df)}")
    else:
        print("Failed to collect any data.")

if __name__ == "__main__":
    main()
