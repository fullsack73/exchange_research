import pandas as pd
import os

base_dir = '/Applications/dollar_price'

# 처리할 작업 목록 정의
tasks = [
    {
        "source": "10y_bond/KOR/시장금리(월,분기,년)_13205444.csv",
        "output": "10y_bond/KOR/10y_bond_KOR_processed.csv",
        "filters": {"계정항목": "국고채(10년)"},
        "value_name": "10Y_BOND_KOR"
    },
    {
        "source": "CPI/KOR/소비자물가지수_13204503.csv",
        "output": "CPI/KOR/CPI_KOR_processed.csv",
        "filters": {"계정항목": "총지수"},
        "value_name": "CPI_KOR"
    },
    {
        "source": "policy_rate/KOR/한국은행 기준금리 및 여수신금리_13200252.csv",
        "output": "policy_rate/KOR/base_rate_KOR_processed.csv",
        "filters": {"계정항목": "한국은행 기준금리"},
        "value_name": "BASE_RATE_KOR"
    },
    {
        "source": "production_index/KOR/산업별 생산_출하_재고 지수_13204837.csv",
        "output": "production_index/KOR/IPI_KOR_processed.csv",
        "filters": {"계정항목": "총지수", "구분코드": "생산지수(계절조정)"},
        "value_name": "IPI_KOR"
    }
]

def process_file(task):
    source_path = os.path.join(base_dir, task['source'])
    output_path = os.path.join(base_dir, task['output'])
    
    print(f"Processing {source_path}...")
    
    if not os.path.exists(source_path):
        print(f"Error: File not found - {source_path}")
        return

    try:
        df = pd.read_csv(source_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 필터링 적용
    # filters 딕셔너리에 있는 모든 조건(AND 조건)을 만족하는 행을 찾음
    target_row = df.copy()
    for col, val in task['filters'].items():
        if col in target_row.columns:
            target_row = target_row[target_row[col] == val]
        else:
            print(f"Warning: Column '{col}' not found in {task['source']}")
    
    if target_row.empty:
        print(f"Error: No matching row found in {task['source']} with filters {task['filters']}")
        # 디버깅을 위해 가능한 유니크 값 출력 (첫번째 필터 기준)
        first_filter_col = list(task['filters'].keys())[0]
        if first_filter_col in df.columns:
            print(f"Available values in '{first_filter_col}': {df[first_filter_col].unique()}")
        return
    
    # 첫 번째 매칭되는 행만 사용
    target_row = target_row.iloc[[0]]

    # 메타데이터 컬럼 식별 (날짜가 아닌 컬럼들)
    # 날짜 컬럼은 보통 'YYYY/MM' 형태이거나 특정 포맷을 따름
    # 여기서는 고정된 메타데이터 컬럼 목록 + '통계표', '계정항목' 등 일반적인 메타데이터 컬럼명을 제외하는 방식을 사용
    potential_metadata = ['통계표', '계정항목', '단위', '변환', '구분코드', '가중치', '측정항목']
    metadata_cols = [c for c in target_row.columns if c in potential_metadata]
    
    date_cols = [c for c in target_row.columns if c not in metadata_cols]
    
    # Melt 수행
    melted_df = target_row.melt(id_vars=metadata_cols, 
                                value_vars=date_cols, 
                                var_name='original_date', 
                                value_name=task['value_name'])

    # 날짜 처리 ('2010/12' -> '2010-12-01')
    try:
        melted_df['observation_date'] = pd.to_datetime(melted_df['original_date'], format='%Y/%m').dt.strftime('%Y-%m-01')
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return

    # 값 처리 (쉼표 제거 및 숫자 변환)
    val_col = task['value_name']
    melted_df[val_col] = melted_df[val_col].astype(str).str.replace(',', '').astype(float)

    # 결과 정리
    final_df = melted_df[['observation_date', val_col]].sort_values('observation_date')
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(final_df.head(3))
    print("-" * 30)

if __name__ == "__main__":
    for task in tasks:
        process_file(task)
