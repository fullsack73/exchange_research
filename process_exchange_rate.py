import pandas as pd
import os

# 파일 경로 설정
base_dir = '/Applications/dollar_price'
source_file = os.path.join(base_dir, 'exchange_rate/주요국 통화의 대원화환율_13193422.csv')
output_file = os.path.join(base_dir, 'exchange_rate/exchange_rate_processed.csv')

def process_exchange_rate():
    # CSV 파일 읽기
    # 데이터가 '통계표', '계정항목' 등으로 시작하는 헤더를 가지고 있음
    # 첫 번째 줄이 헤더가 됨
    try:
        df = pd.read_csv(source_file)
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다: {source_file}")
        return

    # '원/미국달러(매매기준율)' 데이터만 필터링 (필요한 경우)
    # 현재 파일에는 하나의 데이터 행만 보이나, 확실히 하기 위해 필터링
    target_row = df[df['계정항목'] == '원/미국달러(매매기준율)']
    
    if target_row.empty:
        # 혹시 필터링이 안되면 첫번째 데이터 행을 사용
        target_row = df.iloc[[0]]

    # 고정된 컬럼(메타데이터)을 제외한 날짜 컬럼들만 선택
    # 보통 앞의 5개 컬럼이 메타데이터임 (통계표, 계정항목, 측정항목, 단위, 변환)
    metadata_cols = ['통계표', '계정항목', '측정항목', '단위', '변환']
    date_cols = [col for col in df.columns if col not in metadata_cols]

    # Melt를 사용하여 Wide format -> Long format 변환
    melted_df = target_row.melt(id_vars=metadata_cols, 
                                value_vars=date_cols, 
                                var_name='original_date', 
                                value_name='USD_KRW')

    # 날짜 처리: '2010/12' -> '2010-12-01'
    melted_df['observation_date'] = pd.to_datetime(melted_df['original_date'], format='%Y/%m').dt.strftime('%Y-%m-01')

    # 값 처리: 쉼표 제거 및 float 변환
    # 데이터에 쉼표가 포함되어 있을 수 있으므로 제거 후 변환
    melted_df['USD_KRW'] = melted_df['USD_KRW'].astype(str).str.replace(',', '').astype(float)

    # 필요한 컬럼만 선택 및 정렬
    final_df = melted_df[['observation_date', 'USD_KRW']].sort_values('observation_date')

    # 결과 저장
    final_df.to_csv(output_file, index=False)
    print(f"Processing complete. Saved to {output_file}")
    print(final_df.head())

if __name__ == "__main__":
    process_exchange_rate()
