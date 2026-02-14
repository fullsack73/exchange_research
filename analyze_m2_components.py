import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import platform

# 운영체제에 따른 한글 폰트 설정
if platform.system() == 'Darwin': # Mac 환경
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

def analyze_contribution():
    file_path = 'm2/KOR/M2_details_processed.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    df['observation_date'] = pd.to_datetime(df['observation_date'])
    
    # 분석 구간 설정 (Anomaly Period)
    start_date = '2024-11-01'
    end_date = '2025-12-01' # 데이터상 마지막 시점 활용
    
    # 해당 날짜의 데이터 추출
    row_start = df[df['observation_date'] == start_date]
    row_end = df[df['observation_date'] == end_date]
    
    if row_start.empty or row_end.empty:
        print(f"Data for {start_date} or {end_date} not found.")
        return

    # M2 총계 및 불필요한 컬럼 제외
    # 'M2_M2평잔,계절조정계열'은 총합이므로 제외
    # '원자료' 등이 포함될 수 있으므로 실제 상품명만 필터링 필요
    # 여기서는 데이터 구조상 'M2_'로 시작하는 것 중 총합 제외하고 변화량 계산
    
    exclude_keywords = ['M2_M2평잔,계절조정계열', 'M2_상품별_구성내역평잔,_계절조정계열', 'observation_date']
    target_cols = [c for c in df.columns if c not in exclude_keywords and c.startswith('M2_')]
    
    changes = []
    for col in target_cols:
        val_start = row_start[col].values[0]
        val_end = row_end[col].values[0]
        diff = val_end - val_start
        
        # 상품명 가독성 좋게 변경
        display_name = col.replace('M2_', '').replace('_', ' ')
        
        changes.append({
            'Component': display_name,
            'Increase_Amount': diff,
            'Start_Val': val_start,
            'End_Val': val_end
        })

    # DataFrame 변환
    df_change = pd.DataFrame(changes)
    
    # 내림차순 정렬 (증가액 기준)
    df_change = df_change.sort_values(by='Increase_Amount', ascending=False)
    
    # 총 증가분 (Top 5 합계가 아니라 전체 합계)
    total_increase = df_change['Increase_Amount'].sum()
    
    # 비중 계산
    df_change['Contribution (%)'] = (df_change['Increase_Amount'] / total_increase) * 100
    
    print(f"\n========== [M2 증가 원인 정밀 분석] ({start_date} ~ {end_date}) ==========")
    print(f"기간 내 M2 구성항목 총 증가분: {total_increase:,.1f} 십억원")
    print(f"(참고: 약 {total_increase/1000:,.1f} 조 원 증가)")
    
    print("\n[증가 기여도 Top 5 상품]")
    print(df_change[['Component', 'Increase_Amount', 'Contribution (%)']].head(5).to_string(index=False))
    
    # 시각화 (Top 10)
    plt.figure(figsize=(12, 8))
    top_10 = df_change.head(10)
    
    sns.barplot(x='Increase_Amount', y='Component', data=top_10, palette='viridis')
    
    # 수치 표시
    for index, value in enumerate(top_10['Increase_Amount']):
        plt.text(value, index, f' {value:,.0f} ({top_10.iloc[index]["Contribution (%)"]:.1f}%)', va='center')

    plt.title(f'대한민국 M2 증가 주도 상품 Top 10 ({start_date} ~ {end_date})')
    plt.xlabel('증가액 (십억원)')
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = 'm2_components_analysis.png'
    plt.savefig(save_path)
    print(f"\n[알림] 시각화 결과가 '{save_path}'로 저장되었습니다.")

if __name__ == "__main__":
    analyze_contribution()
