# Exchange Rate Analysis Project (USD/KRW)

## 프로젝트 개요 (Overview)
본 프로젝트는 장기적인 관점에서 USD/KRW 환율의 움직임을 규명하고, 금리차 등 전통적 거시경제 이론이 작동하지 않는 '이상 구간(Anomaly Period)' 동안 환율 변동의 움직임과 그 여파를 **'예측(Predict)'** 하는 모델을 구축하는 것을 핵심 목표로 합니다. 위기나 이벤트 상황에서 단기 유동성(M2, MMF, 요구불예금 등)이 환율에 미치는 영향을 중심으로 새로운 이론과 '임계점 가설(Threshold Hypothesis)'을 검증합니다.

**이상 구간(Anomaly Period)의 정의:** 환율(Exchange Rate)과 한-미 정책금리차(Interest Rate Spread)가 전통적인 **'양(+)의 상관관계'를 보이지 않고 디커플링(Decoupling)되는 모든 시점**을 이상 구간으로 정의하여 집중 분석합니다.

**분석 기간 (Analysis Period):** 1995-01-01 ~ 2026-01-01

### 주요 분석 결과 및 단계 (Key Findings & Hypotheses)
1. **장기/이상 구간 비교 (Baseline Deviation):** 선형 관계 분석을 통해 환율과 주요 거시 지표 간의 장기적 관계(Baseline)와 최근 이상 구간에서의 확실한 괴리 포착.
2. **M2 영향력 및 임계점 가설 (Threshold Hypothesis):** Random Forest 및 SHAP 분석을 통해 이상 구간 내 M2의 압도적인 영향력을 확인하고, 선형 관계를 벗어난 '임계점 가설' 수립.
3. **M2 구성요소 분석 (Component Analysis):** 전체 M2가 아닌 구성 요소별 기여도를 분석하여 쏠림 현상 규명.
4. **MMF & CMA 집중 분석:** MMF와 CMA 데이터를 대상으로 한 일 단위 Threshold 및 SHAP 분석을 통해 임계점 가설 추가 증명.
5. **LSTM 예측 1 (MMF & CPI):** MMF 단일 변수 및 MMF+CPI 변수를 활용한 딥러닝(LSTM) 환율 예측 모델 구축 및 현상 설명력 한계 확인.
6. **LSTM 예측 2 (요구불예금):** 단기 요구불예금(Demand Deposits)으로 대체한 딥러닝 모델(및 Hybrid LSTM)을 통해 환율 예측 성능의 유의미한 개선(Counterfactual Simulation) 확인.

---

## 프로젝트 구조 (Repository Structure)
- `data/`: 거시경제 지표 및 환율 원본 데이터 관리
  - `process_scripts/`: 원본 데이터를 병합하고 파생 변수(예: 이자율 차이, Forward Rate)를 계산하는 전처리 스크립트 모음
  - 하위 폴더: `10y_bond`, `CPI`, `exchange_rate`, `m2`, `policy_rate`, `production_index` 등 각 지표별 데이터
- `analysis/`: 핵심 분석 스크립트, 머신러닝/딥러닝 모델링 및 평가지표 결과(`result.md`)
  - `baseline/` & `anomaly/`: 장기(Baseline) 및 이상구간(Anomaly) 설정 및 기본 선형 분석
  - `shap_ml/` & `m2_components/`: 머신러닝(RF, XGB)과 SHAP을 활용한 변수 중요도 및 M2 세부 요소 분석
  - `daily_threshold_MMF/` & `daily_shap_MMF/`: MMF 중심의 일별 스파이크(Threshold) 효과 정밀 검증
  - `LSTM/`: 환율 방향성 및 변동 예측을 위한 딥러닝 모델 (순수 LSTM 및 ARIMA-CNN-LSTM 결합 Hybrid 모델 실험 코드 포함)
- `reports/`: 연구진 공유용 종합 보고서, 검증 결과 및 프레젠테이션 문서(Markdown)

---

## 환경 설정 및 실행 가이드 (Setup & Reproduction)

본 프로젝트의 코드를 실행하고 실험 결과를 재현하기 위한 가이드입니다.

### 1. 환경 설정 (Environment Setup)
프로젝트 루트 경로에 포함된 `requirements.txt`를 사용하여 가상 환경에 의존성 패키지를 설치합니다. 딥러닝 학습(LSTM) 속도 향상을 위해 GPU 사용이 가능한 경우, 환경에 맞는 [PyTorch](https://pytorch.org/get-started/locally/)를 별도로 설치할 것을 권장합니다.

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비 및 전처리 (Data Preparation)
한국은행(BOK)이나 FRED 등에서 추출한 원본 CSV 데이터가 `data/` 내부의 각 하위 폴더에 존재해야 합니다.
(현재 원본 데이터를 자동으로 다운로드하는 스크립트는 포함되어 있지 않으므로 수동 배치가 필요합니다.)

데이터가 준비되면 아래 순서대로 전처리 파이프라인을 실행하여 훈련 가능한 일단위/월단위 데이터를 생성합니다.

```bash
# 1. 각 지표별 개별 데이터 병합 및 정제
python data/process_scripts/process_all_indicators.py

# 2. 통합 파이프라인 구동 (스프레드 및 이론 환율 계산)
python data/process_scripts/rebuild_daily_pipeline.py
```

### 3. 분석 및 모델링 실행 순서 (Execution Pipeline)
모델 학습과 결과 도출은 폴더별로 다음 세 가지 Phase로 나뉘어 진행됩니다. 반드시 프로젝트 루트(`BASE_DIR`)에서 스크립트를 실행해 주십시오.

#### Phase 1: Baseline 및 이상 구간 정의
```bash
python analysis/baseline/analyze_factors.py
python analysis/anomaly/detect_anomaly_period.py
```

#### Phase 2: Feature Importance 및 SHAP 분석
```bash
python analysis/shap_ml/analyze_shap.py
python analysis/m2_components/analyze_m2_components.py
python analysis/daily_threshold_MMF/analyze_daily_threshold.py
python analysis/daily_shap_MMF/analyze_daily_shap.py
```

#### Phase 3: LSTM 및 Hybrid 모델 예측 (가설 검증)
```bash
# 데이터 분리 및 텐서 변환 (MMF 및 요구불예금 데이터셋 준비)
python analysis/LSTM/lstm_mmf/prep_daily.py
python analysis/LSTM/lstm_m2_demand_deposit/prep_m2_demand_deposit.py

# 단일 변수/복합 변수 LSTM 훈련 및 평가
python analysis/LSTM/lstm_mmf/train_eval_extended.py
python analysis/LSTM/lstm_m2_demand_deposit/train_eval_extended.py

# Hybrid 모델(ARIMA + LSTM/CNN) 훈련 
python analysis/LSTM/run_hybrid_periods.py
python analysis/LSTM/run_hybrid_log_multistep.py
```

---

## ⚠️ 실행 시 주의 사항 (Important Warnings)

1. **하드코딩된 경로 (Hardcoded Paths):** 다수의 파이썬 스크립트 내부에 `BASE_DIR = Path("/Applications/dollar_price")`와 같이 절대 경로가 지정되어 있습니다. 자신의 로컬 환경에 맞게 코드를 수정하거나, 상대 경로로 리팩토링한 뒤 실행해야 에러가 발생하지 않습니다.
2. **시각화 폰트 오류 (Korean Fonts rendering):** 차트 및 플롯(Plot) 생성 스크립트는 시각화를 위해 macOS의 경우 `AppleGothic`, Windows의 경우 `Malgun Gothic`(맑은고딕)을 사용하도록 하드코딩 되어 있습니다. 폰트가 설치되어 있지 않으면 그래프 내 한글이 깨져(▯▯▯) 보일 수 있습니다.
3. **결과 확인:** 주요 실행 결과 및 그래프 이미지는 각 `analysis/*` 내 폴더의 하위 경로 혹은 프로젝트 루트의 `reports/` 내 `.md` 파일에 기록됩니다.