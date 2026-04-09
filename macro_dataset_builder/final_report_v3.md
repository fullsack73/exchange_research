# 시계열 흐름 동시 비교 (Time-series Flow Analysis) 최종 결과

## 1. 분석 개요 및 핵심 변수 선정 기준
환율(USD_KRW)과의 상호 작용을 시계열 역학(선행/동행/디커플링) 관점에서 직관적으로 파악하기 위해 경제적 중요도가 뚜렷한 8개 핵심 지표만을 선정했습니다.
- **달러 공급 및 대외결제 근간 지표**: `Exports_YoY` (수출액), `Trade_Balance_MoM` (무역수지 변화)
- **인플레이션 및 물가 전가 지표**: `Import_Price_Index_YoY` (환율 충격 1차 전가 대상), `CPI_KOR_YoY` (국내 경제 기조 전환점)
- **자본시장 심리/자금 흐름 지표**: `DXY_MoM` (글로벌 달러 자본 흐름), `Rate_Spread_KOR_USA_MoM` (내외금리차에 따른 자본이탈 모멘텀)
- **실물 경제 충격 지표**: `Industrial_Production_YoY` (실물 경기 수축), `Unemployment_KOR_YoY` (가장 후행적인 충격 지표)

## 2. 변수별 환율 반응 동태 (Time-series Dynamics) 특성 요약

*(모든 플롯은 Z-Score 스케일러가 적용되어 절대 눈금이 아닌 방향과 교차 타이밍 중심으로 통일 배치되었습니다. 환율은 Red, 변수는 Blue입니다. 이상 구간은 음영 처리되었습니다.)*

### 동행성(Coupling)이 뚜렷한 실시간 전이 지표
- **DXY_MoM**와 **Import Price Index_YoY**는 4번의 위기 구간(Zoom-in 그래프) 내내 피크(정점)를 거의 완벽히 겹치며 동기화되는 철저한 동반비행(Co-movement) 커플링 특성을 보입니다.

### 선행성(Leading) 강한 위기 예고 지표
- **Exports_YoY (수출액)** 및 **Trade_Balance (무역수지)** 등 창구 달러 수급 지표들은 위기 돌입 수개월 전부터 하향 쐐기(마이너스)를 파면서 레드선(환율) 상승에 강하게 역의 선행 상관 관계를 발휘합니다. 실물 위기의 신호탄 성격을 띠게 됩니다.

### 디커플링(Decoupling) 성향 또는 후행 지표
- **Industrial_Production**과 **Unemployment (실업률)** 등 내부 경제 생산 지표들은 초반에 이렇다할 반응을 바로 보이지 않고 일정 국면을 한참 지난 뒤에서야 환율에 수동적으로 하락/상승 반응하거나 고유 노이즈 형태로 진행(디커플링)되는 모습을 나타냈습니다.

## 3. 계량적 선행/후행 (Lag Correlation) 시차 측정 요약
`lag_correlation.csv`의 도출 결과입니다. 
(* 양수(Positive) Lag는 해당 변수가 환율보다 먼저 일어나는 선행 지표임을 시사합니다 *)
- **DXY_MoM**: 최고 상관계수 `0.646`, 동행 (Immediate Co-movement)
- **Import_Price_Index_YoY**: 최고 상관계수 `0.250`, 1개월 후행 (Variable Lags FX)
- **Exports_YoY**: 최고 상관계수 `-0.161`, 6개월 후행 (Variable Lags FX)
- **Trade_Balance_MoM**: 최고 상관계수 `0.146`, 2개월 선행 (Variable Leads FX)
- **CPI_KOR_YoY**: 최고 상관계수 `0.107`, 1개월 후행 (Variable Lags FX)
- **Industrial_Production_YoY**: 최고 상관계수 `-0.190`, 4개월 후행 (Variable Lags FX)
- **Rate_Spread_KOR_USA_MoM**: 최고 상관계수 `0.193`, 3개월 선행 (Variable Leads FX)
- **Unemployment_KOR_YoY**: 최고 상관계수 `-0.097`, 3개월 선행 (Variable Leads FX)
