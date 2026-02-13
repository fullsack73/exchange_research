import pandas as pd
import os

# 파일 경로 설정
base_path = '/Applications/dollar_price'
file_paths = {
    'exchange': os.path.join(base_path, 'exchange_rate/exchange_rate_processed.csv'),
    'base_rate_kor': os.path.join(base_path, 'policy_rate/KOR/base_rate_KOR_processed.csv'),
    'fed_funds': os.path.join(base_path, 'policy_rate/USA/FEDFUNDS.csv')
}

# 데이터 로드 함수
def load_data(path):
    print(f"Reading {path}...")
    try:
        df = pd.read_csv(path)
        # format='%Y-%m-%d' ensures we parse the YYYY-MM-DD format correctly
        df['observation_date'] = pd.to_datetime(df['observation_date'], format='%Y-%m-%d')
        df.set_index('observation_date', inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        return None
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def process_data():
    # 1. 데이터 불러오기
    df_exch = load_data(file_paths['exchange'])
    df_kor = load_data(file_paths['base_rate_kor'])
    df_usa = load_data(file_paths['fed_funds'])

    if df_exch is None or df_kor is None or df_usa is None:
        print("Required files are missing. Aborting.")
        return

    # 데이터 병합 (교집합 날짜 기준 - inner join)
    # df_exch: USD_KRW
    # df_kor: BASE_RATE_KOR
    # df_usa: FEDFUNDS
    df_merged = df_exch.join([df_kor, df_usa], how='inner')
    
    # 병합된 데이터가 비어있는지 확인
    if df_merged.empty:
        print("Merged result is empty. Check if dates match across files.")
        return

    print(f"Merged data has {len(df_merged)} rows.")

    # 2. 한미 금리차 계산 (Spread)
    # 공식: 한국 기준금리 - 미국 기준금리
    df_spread = pd.DataFrame(index=df_merged.index)
    df_spread['observation_date'] = df_merged.index
    df_spread['RATE_SPREAD_KOR_USA'] = df_merged['BASE_RATE_KOR'] - df_merged['FEDFUNDS']
    
    # 3. 이론적 1년 선물환율 계산 (Theoretical Forward Rate based on IRP)
    # 공식: F = S * (1 + r_kor) / (1 + r_usa)
    # * 가정: 1년 만기
    # * r은 % 단위이므로 100으로 나누어 소수점으로 변환
    spot_rate = df_merged['USD_KRW']
    r_kor = df_merged['BASE_RATE_KOR'] / 100
    r_usa = df_merged['FEDFUNDS'] / 100
    
    df_fwd = pd.DataFrame(index=df_merged.index)
    df_fwd['observation_date'] = df_merged.index
    df_fwd['THEORETICAL_FWD_RATE'] = spot_rate * (1 + r_kor) / (1 + r_usa)

    # 4. 파일 저장
    # 저장 경로 설정
    output_spread_path = os.path.join(base_path, 'policy_rate/spread_KOR_USA_processed.csv')
    output_fwd_path = os.path.join(base_path, 'exchange_rate/theoretical_fwd_rate_processed.csv')
    
    # observation_date 컬럼을 포함하여 저장 (index=False로 저장하기 위해 데이터프레임 조작했음)
    # 하지만 더 깔끔하게 인덱스 리셋을 이용하는게 나을 수도 있음. 
    # 위에서 이미 컬럼을 할당했으므로 to_csv시 index=False 하면 됨.
    
    # CSV 저장 (날짜 포맷 유지)
    df_spread.to_csv(output_spread_path, index=False)
    df_fwd.to_csv(output_fwd_path, index=False)
    
    print("Files created successfully:")
    print(f"1. {output_spread_path}")
    print(df_spread.head())
    print("-" * 30)
    print(f"2. {output_fwd_path}")
    print(df_fwd.head())

if __name__ == "__main__":
    process_data()
