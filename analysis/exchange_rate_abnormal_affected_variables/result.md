# 환율 이상구간(Abnormal) 영향 변수 선형분석 결과

## 1. 분석 목적
이전 `macro_dataset_builder`에 산재된 거시경제 변수 환율분석(모멘텀/상관/RF 분류) 코드를 현 프로젝트의 일원화된 분석 방법론(동적 이상구간 분리 및 선형 회귀 계수 델타(Delta) 추출)으로 통합 개선하였습니다. 본 문서는 환율이 평상시(Baseline) 대비 이상구간(Anomaly)에 접어들었을 때 어떤 대외/실물 지표와 강하게 동조(Regime Change)하는지 파악한 결과입니다.

## 2. 분석 방법
- **데이터 소스**: `data/macro_dataset_processed.csv` 내 통합 파생 변수 (`_{MoM, YoY}` 형태)
- **기간 설정**: `analysis/anomaly/period_definition.json`에 정의된 Baseline과 Anomaly 구간 동적 분리 적용.
- **분석 기법**:
  - `Linear Regression`: 단위 변동량에 따른 환율 변동성($\beta$, Coef) 추정, 정상/이상 간의 $\Delta\beta$ 비교.
  - `Rolling Correlation`: 분석을 통해 추출된 상위 2대 핵심 연관변수(**Policy_Rate_KOR_MoM**, **Rate_Spread_KOR_USA_MoM**)의 구조적 변화(Regime Shift) 시계열 추적.

## 3. 분석 결과분석 요약

### 3.1. 위기 구간에서 가장 영향을 크게 미치는 핵심동인
이상구간 내에서 상관계수 및 회귀계수가 폭증한 상위 2개의 동인은 **Policy_Rate_KOR_MoM**와 **Rate_Spread_KOR_USA_MoM** 입니다. 안전 자산 선호가 쏠리는 달러 강세 타이밍에 밀접한 상관성을 보이고 있습니다.

### 3.2. 평상시 대비 환율-거시 변수 간의 동조화 경향(Regime Shift)
- `results/linear_analysis_delta.csv` 자료 및 회귀계수 결과에서 나타나듯, 평상시에는 계수나 상관성이 낮던 요소들이 위기가 발생하면 1에 가깝게 폭증하며,
- 동적 `rolling_corr` 차트를 확인하면 이상구간 통과 시점에 특정 변수들과의 방향성이 돌변하거나 증폭되는 동조화 지점을 뚜렷이 확인할 수 있습니다.
