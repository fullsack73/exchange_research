import pandas as pd # 데이터처리
import requests # 웹에서 데이터 가져오기
import io # 메모리에서 데이터를 파일처럼 다루기
import os # 운영체제 기능 사용

os.makedirs('../../data', exist_ok=True) # data 폴더 생성, 이미 있으면 무시 (exist_ok=True)

series_dict = {
    'Headline': 'CPIAUCSL', # 미국 전체 CPI
    'Core': 'CPILFESL', # Core CPI (식품/에너지 제외)
    'Shelter': 'CUSR0000SAH1', # 주거비
    'Food': 'CPIFABSL', # 식료품
    'Energy': 'CPIENGSL', # 에너지
    'MedicalCare': 'CPIMEDSL', # 의료비
    'Transportation': 'CPITRNSL', # 교통비
    'Apparel': 'CPIAPPSL', # 의류비
    'Durables': 'CUSR0000SAD', # 내구재
    'Services': 'CUSR0000SAS', # 서비스
    'KRWUSD': 'DEXKOUS' # 원/달러 환율
}

def download_fred_csv(series_id): # FRED에서 데이터 다운로드
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"Downloading {series_id}...") # 다운로드 URL 생성
    response = requests.get(url) 
    if response.status_code == 200: # 성공 여부 확인 (200 = 성공)
        return pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True) # CSV 데이터를 pandas DataFrame(테이블형)으로 변환
    else:
        print(f"Failed to download {series_id}: Status {response.status_code}")
        return None

all_dfs = [] # 다운로드된 데이터들을 담을 리스트
for name, series_id in series_dict.items(): # series_dict의 각 항목을 순회
    df = download_fred_csv(series_id) # FRED에서 데이터 다운로드
    if df is not None: # 데이터가 성공적으로 다운로드되었는지 확인
        df.columns = [name] # 데이터프레임의 컬럼 이름을 시리즈 이름으로 변경
        all_dfs.append(df) # 다운로드된 데이터프레임을 all_dfs 리스트에 추가

if all_dfs: # all_dfs 리스트에 데이터가 하나라도 있으면
    final_df = pd.concat(all_dfs, axis=1) # all_dfs 리스트의 데이터프레임들을 가로로 합침
    
    # Save daily KRWUSD
    krwusd_daily = final_df[['KRWUSD']].copy().dropna() # KRWUSD 컬럼만 선택하고 결측치 제거
    krwusd_daily.to_csv('../../data/krwusd_daily.csv') # CSV 파일로 저장
    
    # Save monthly CPI (resampled to Month Start) # 월별 CPI 저장 (월초 기준으로 리샘플링)
    cpi_cols = [c for c in final_df.columns if c != 'KRWUSD'] # KRWUSD를 제외한 나머지 컬럼 선택
    cpi_monthly = final_df[cpi_cols].resample('MS').first().dropna(how='all') # 월초 기준으로 리샘플링하고 결측치 제거
    cpi_monthly.to_csv('../../data/us_cpi_monthly.csv') # CSV 파일로 저장
    
    print("Data collection completed successfully via direct download.")
    print(f"Files saved in data/ directory. CPI rows: {len(cpi_monthly)}")
else:
    print("No data was downloaded.")
