# LSTM 분석 결과: 금리차 + 수시입출식저축성예금

## 분석 목적
이전 실험과 동일한 구조로, 아래 두 모델의 환율 예측 성능을 비교했습니다.

- Model A: USD_KRW + RATE_SPREAD_KOR_USA
- Model B: USD_KRW + RATE_SPREAD_KOR_USA + M2_수시입출식저축성예금

## 데이터/설정
- 데이터셋: `daily_dataset_m2_demand_deposit.csv`
- 데이터 기간: 2010-12-01 ~ 2026-03-16
- 시퀀스 길이: 30일
- 예측 시점: 5영업일 ahead (`pred_step=5`)
- 학습/평가 분할: 각 기간별 시계열 80/20
- 평가 지표: RMSE, MAE

## 결과 요약

### 1) 전체 구간 (full_2010_12_to_2026_03)
- 표본 수: 3774 (train=3019, test=755)
- Model A: RMSE 44.62, MAE 40.98
- Model B: RMSE 64.00, MAE 59.98
- 우수 모델: Model A

### 2) 이상 구간 (anomaly_2024_11_to_2026_03)
- 표본 수: 333 (train=266, test=67)
- Model A: RMSE 21.21, MAE 18.13
- Model B: RMSE 18.98, MAE 14.45
- 우수 모델: Model B

## 해석
- 전체 구간에서는 Model A가 더 우수했습니다. 즉 장기 전체 구간에서는 금리차 기반 단순 모델이 더 안정적으로 동작했습니다.
- 이상 구간에서는 Model B가 더 우수했습니다. 특히 MAE가 18.13 -> 14.45로 개선되어, 스트레스 국면에서는 `수시입출식저축성예금` 정보가 보조 설명력을 제공할 가능성을 확인했습니다.
- 정리하면, 해당 변수의 효과는 "전체 구간 일관 개선"보다는 "이상 구간 조건부 개선"에 가깝습니다.

## 산출물
- `analysis/LSTM/lstm_m2_demand_deposit/results.json`
- `analysis/LSTM/lstm_m2_demand_deposit/results.txt`
- `analysis/LSTM/lstm_m2_demand_deposit/full/full_2010_12_to_2026_03_lstm_plot_full.png`
- `analysis/LSTM/lstm_m2_demand_deposit/eval/full_2010_12_to_2026_03_lstm_plot_eval.png`
- `analysis/LSTM/lstm_m2_demand_deposit/full/anomaly_2024_11_to_2026_03_lstm_plot_full.png`
- `analysis/LSTM/lstm_m2_demand_deposit/eval/anomaly_2024_11_to_2026_03_lstm_plot_eval.png`
