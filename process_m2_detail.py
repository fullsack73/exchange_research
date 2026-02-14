import pandas as pd

def process_m2_file():
    # 파일명에 공백과 한글이 포함되어 있어 그대로 사용
    input_path = 'm2/KOR/M2 상품별 구성내역(평잔, 계절조정계열)_14184322.csv'
    output_path = 'm2/KOR/M2_details_processed.csv'
    
    try:
        # 데이터 구조: 
        # 1행: 날짜 헤더 (2010/12 ~ 2025/12) - 컬럼 인덱스 4부터
        # 2행~: 데이터 (컬럼 인덱스 1: 항목명, 컬럼 인덱스 4~: 값)
        
        # 헤더 없이 불러와서 직접 처리
        df_raw = pd.read_csv(input_path, header=None, encoding='utf-8')
        
        # 1. 날짜 리스트 생성 (첫 번째 행, 5번째 컬럼부터 끝까지)
        raw_dates = df_raw.iloc[0, 4:].values
        dates = []
        for d in raw_dates:
            if isinstance(d, str):
                # 2010/12 -> 2010-12-01
                dates.append(d.replace('/', '-') + '-01')
            else:
                dates.append(None) # 날짜 없는 경우
                
        # 결과를 담을 딕셔너리 초기화
        data = {'observation_date': dates}
        
        # 2. 각 항목별 데이터 추출
        # 첫 번째 행은 날짜 헤더이므로 제외
        for idx, row in df_raw.iloc[1:].iterrows():
            category_name = row[1]
            if pd.isna(category_name):
                continue
                
            # 항목명 정제
            clean_name = "M2_" + category_name.strip().replace(' ', '_').replace('1)', '').replace('(', '').replace(')', '')
            
            # 값 추출 (5번째 컬럼부터)
            values = []
            for val in row[4:]:
                if pd.isna(val) or val == '':
                    values.append(0.0)
                else:
                    # 쉼표 제거 후 float 변환
                    try:
                        val_str = str(val).replace(',', '')
                        values.append(float(val_str))
                    except:
                        values.append(0.0)
            
            # 길이 맞추기 (혹시 모를 오류 방지)
            if len(values) == len(dates):
                data[clean_name] = values
            else:
                # 길이가 안 맞으면 날짜 길이만큼 자르거나 0으로 채움
                if len(values) > len(dates):
                    data[clean_name] = values[:len(dates)]
                else:
                    values += [0.0] * (len(dates) - len(values))
                    data[clean_name] = values

        # DataFrame 생성
        df_result = pd.DataFrame(data)
        
        # 날짜 형식 변환
        df_result['observation_date'] = pd.to_datetime(df_result['observation_date'])
        
        # 저장
        df_result.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        print(f"Total rows: {len(df_result)}")
        print("Columns:", df_result.columns.tolist())
        
        return df_result
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    process_m2_file()
