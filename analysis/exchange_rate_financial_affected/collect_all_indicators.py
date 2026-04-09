import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime

# .env 파일에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("ECOS_API_KEY")

def fetch_ecos_data(stat_code, start_date, end_date, cycle, item_code1, item_code2=None, item_code3=None):
    """
    한국은행 ECOS API를 통해 통계 데이터를 가져옵니다.
    """
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/50000/{stat_code}/{cycle}/{start_date}/{end_date}/{item_code1}/"
    if item_code2:
        url += f"{item_code2}/"
    if item_code3:
        url += f"{item_code3}/"
        
    print(f"Fetching: {stat_code} ({item_code1}) {cycle} from {start_date} to {end_date}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "StatisticSearch" in data:
            rows = data["StatisticSearch"]["row"]
            df = pd.DataFrame(rows)
            # 필요한 컬럼만 선택 및 정제
            df = df[['TIME', 'DATA_VALUE']]
            # TIME 형식 변환
            if cycle == 'D':
                df['Date'] = pd.to_datetime(df['TIME'], format='%Y%m%d')
            elif cycle == 'M':
                df['Date'] = pd.to_datetime(df['TIME'], format='%Y%m')
            else:
                df['Date'] = df['TIME'] # Fallback
                
            df['DATA_VALUE'] = pd.to_numeric(df['DATA_VALUE'])
            return df[['Date', 'DATA_VALUE']]
        else:
            msg = data.get('RESULT', {}).get('MESSAGE', 'Unknown error')
            print(f"No data or API Error for {stat_code}/{item_code1}: {msg}")
            return None
    else:
        print(f"HTTP Error: {response.status_code}")
        return None

def main():
    if not API_KEY:
        print("Error: ECOS_API_KEY not found in .env file.")
        return

    start_date_m = "199501"
    start_date_d = "19950101"
    end_date_m = datetime.now().strftime("%Y%m")
    end_date_d = datetime.now().strftime("%Y%m%d")
    
    results = {}

    # 1. 외국인 증권투자 (BOP) - 월간
    print("\n--- 1. 외국인 증권투자 (BOP) ---")
    stock_df = fetch_ecos_data("301Y013", start_date_m, end_date_m, "M", "BOPF22100000")
    bond_df = fetch_ecos_data("301Y013", start_date_m, end_date_m, "M", "BOPF22200000")
    if stock_df is not None:
        results['Foreign_Stock_Investment'] = stock_df.set_index('Date')['DATA_VALUE']
    if bond_df is not None:
        results['Foreign_Bond_Investment'] = bond_df.set_index('Date')['DATA_VALUE']

    # 2. KOSPI 주가지수 - 일간 가져와서 월평균 계산
    print("\n--- 2. KOSPI 주가지수 ---")
    kospi_df = fetch_ecos_data("802Y001", start_date_d, end_date_d, "D", "0001000")
    if kospi_df is not None:
        kospi_monthly = kospi_df.set_index('Date').resample('MS')['DATA_VALUE'].mean()
        results['KOSPI'] = kospi_monthly

    # 3. 기업경기실사지수 (BSI) - 월간 (전산업 업황실적)
    print("\n--- 3. 기업경기실사지수 (BSI) ---")
    bsi_df = fetch_ecos_data("512Y001", start_date_m, end_date_m, "M", "AA", "99988")
    if bsi_df is not None:
        results['BSI_All_Industry'] = bsi_df.set_index('Date')['DATA_VALUE']

    # 4. 소비자동향지수 (CSI) - 월간 (소비자심리지수 CCSI)
    print("\n--- 4. 소비자동향지수 (CSI) ---")
    csi_df = fetch_ecos_data("511Y002", start_date_m, end_date_m, "M", "FME", "99988")
    if csi_df is not None:
        results['CSI_CCSI'] = csi_df.set_index('Date')['DATA_VALUE']

    # 모든 결과 합치기
    if results:
        final_df = pd.DataFrame(results)
        final_df.index.name = 'Date'
        
        # data 폴더 생성
        os.makedirs('data', exist_ok=True)
        output_path = 'data/financial_indicators_monthly_1995_2026.csv'
        final_df.to_csv(output_path)
        
        print(f"\nSuccessfully saved all data to {output_path}")
        print(f"Data range: {final_df.index.min().date()} to {final_df.index.max().date()}")
        print(f"Columns: {list(final_df.columns)}")
        print("\nLast 5 rows:")
        print(final_df.tail())
    else:
        print("Failed to collect any data.")

if __name__ == "__main__":
    main()
