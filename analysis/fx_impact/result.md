# FX Impact 최종 파이프라인 결과

## 목적

이 최종 파이프라인은 이상구간 월(`event_date`)에 USD/KRW 환율이 움직였을 때, 국내 거시/금융 변수들이 이후 `h=1..6`개월 동안 어떤 방향과 크기로 반응하는지 추정한다. 핵심 질문은 “이상구간의 이벤트 월 `t`에서 환율 충격이 발생하면, `t+h` 시점의 target response는 얼마인가?”이다.

## 기존 Anomaly-Concat Forecast 방식의 문제

- 월별 연속 forecast는 이상구간 조건부 파급효과라기보다 일반적인 calendar-time 예측 성능을 주로 측정한다.
- anomaly month만 필터링한 뒤 하나의 시계열처럼 이어붙이면 실제 calendar gap이 사라진다. 이 경우 `shift(1)`이 서로 떨어진 이상구간 사이를 바로 연결해 lag가 왜곡될 수 있다.
- 기존 level-path forecast plot은 예측선이 실제 level 경로를 얼마나 따라가는지에 초점이 있어, “환율 충격 후 h개월 뒤 target response”라는 핵심 질문을 직관적으로 보여주기 어렵다.

## Event-Time Local Projection 방식

- 새 최종 단계는 `(event_date, target, horizon)` 단위의 event-time panel을 만든다.
- `event_date` row는 `Is_Abnormal_Period == 1`인 월로만 제한한다.
- 단, lag와 response는 반드시 원래 월별 calendar index에서 가져온다. 즉 `t-1`, `t-6`, `t+h`는 anomaly-only로 이어붙인 시계열에서 계산하지 않는다.
- response는 event month `t`에서 `t+h`까지의 누적 변화로 정의한다.
- log-level target은 `log(target_{t+h}) - log(target_t)`를 사용한다.
- diff target은 `target_{t+h} - target_t`를 사용한다.

## 데이터 구성

- 패널에 사용된 anomaly event month 수: `68`
- horizon과 결측값 제거 후 event-time panel row 수: `2937`
- horizon: `1, 2, 3, 4, 5, 6`개월
- selected targets: `CSI_CCSI`, `Imports`, `KOSPI`, `Industrial_Production`, `Import_Price_Index`, `Foreign_Bond_Investment`, `Foreign_Stock_Investment`, `Trade_Balance`
- FX input source: `hybrid_m2_model_a`
- 선택된 daily FX prediction은 월말 기준 월평균으로 resample해 macro panel에 결합한다. hybrid prediction이 없는 월은 이전 final pipeline과 동일하게 actual USD/KRW를 유지한다.
- scenario shock은 각 event month의 predicted USD/KRW에 `1.050`을 곱해 만든다. scenario의 current-month log shock은 shock이 들어가지 않은 predicted `t-1` 대비로 계산하므로, event month의 +5% 충격이 `scenario_fx_change_t`에 반영된다.

## 모델

- `LocalProjection_OLS`: target별, horizon별 OLS local projection. p-value 계산을 위해 HAC covariance를 사용한다.
- `LocalProjection_Ridge`: target별, horizon별 Ridge local projection. scaler는 train split에만 fit한다.
- `LocalProjection_ElasticNet`: event sample이 충분한 경우 time-series CV를 사용하는 ElasticNet/ElasticNetCV local projection.
- 기본 feature set은 `fx_shock_t`, `fx_lag1..fx_lag6`, `target_lag1`이다.
- 외부 controls는 이번 최종 모델에서 제외했다. target/horizon별 event sample이 작아 controls를 넣으면 표본 대비 feature 수가 과해질 수 있기 때문이다.
- split은 target/horizon별 `event_date` 기준 time split이며, 유효 anomaly event date의 마지막 약 25%를 test set으로 사용한다.
- leakage 방지를 위해 scaler는 train에만 fit하고, future target 값은 `actual_response` 평가에만 사용한다. lag와 response 계산은 모두 원래 monthly calendar index에서 수행한다.

## 성능 요약

Predicted FX 기준 전체 test 성능:

- `LocalProjection_ElasticNet`: avg RMSE=1770.129739, avg NRMSE=1.2328, avg level-delta RMSE=427221.7958, targets=8
- `LocalProjection_Ridge`: avg RMSE=1779.840873, avg NRMSE=1.2707, avg level-delta RMSE=561025.4634, targets=8
- `LocalProjection_OLS`: avg RMSE=1780.422375, avg NRMSE=1.2762, avg level-delta RMSE=570948.2033, targets=8

Target별 plot용 best model:

- `CSI_CCSI`: `LocalProjection_ElasticNet`, avg NRMSE=0.9910, avg RMSE=7.264194, peak scenario-baseline response=-2.391069 at h=2
- `Foreign_Bond_Investment`: `LocalProjection_ElasticNet`, avg NRMSE=1.0301, avg RMSE=5008.097973, peak scenario-baseline response=-276.915493 at h=1
- `Foreign_Stock_Investment`: `LocalProjection_OLS`, avg NRMSE=0.9890, avg RMSE=6233.877249, peak scenario-baseline response=1023.551512 at h=6
- `Import_Price_Index`: `LocalProjection_OLS`, avg NRMSE=1.7859, avg RMSE=0.061496, peak scenario-baseline response=-0.016686 at h=3
- `Imports`: `LocalProjection_ElasticNet`, avg NRMSE=1.0302, avg RMSE=0.063879, peak scenario-baseline response=-0.021083 at h=3
- `Industrial_Production`: `LocalProjection_ElasticNet`, avg NRMSE=1.0979, avg RMSE=0.011537, peak scenario-baseline response=0.000000 at h=1
- `KOSPI`: `LocalProjection_ElasticNet`, avg NRMSE=1.2365, avg RMSE=0.174893, peak scenario-baseline response=-0.000124 at h=1
- `Trade_Balance`: `LocalProjection_ElasticNet`, avg NRMSE=1.1408, avg RMSE=2858.799051, peak scenario-baseline response=386.788263 at h=1

필수 target별 코멘트:

- `KOSPI`: best model은 `LocalProjection_ElasticNet`이며 avg NRMSE=1.2365이다. scenario effect는 음의 방향이고 h=1 부근에서 가장 크게 나타난다. 안정성은 낮은 편이므로 해석에 주의가 필요하다.
- `Import_Price_Index`: best model은 `LocalProjection_OLS`이며 avg NRMSE=1.7859이다. scenario effect는 음의 방향이고 h=3 부근에서 peak가 나타난다. NRMSE가 높아 가장 불안정한 target 중 하나다.
- `Industrial_Production`: best model은 `LocalProjection_ElasticNet`이며 avg NRMSE=1.0979이다. scenario effect는 거의 0에 가까워, +5% FX shock에 대한 추가 반응이 뚜렷하지 않다.
- `Trade_Balance`: best model은 `LocalProjection_ElasticNet`이며 avg NRMSE=1.1408이다. scenario effect는 양의 방향이고 h=1 부근에서 가장 크게 나타난다. 다만 안정성은 중간 이하로 보아야 한다.

## 주요 해석

Scenario baseline 대비 절대 response delta가 큰 변수는 다음과 같다.

- `Foreign_Stock_Investment`: h=6에서 가장 큰 양의 반응. response delta=1023.551512, level-delta effect=1023.5515
- `Trade_Balance`: h=1에서 양의 반응. response delta=386.788263, level-delta effect=386.7883
- `Foreign_Bond_Investment`: h=1에서 음의 반응. response delta=-276.915493, level-delta effect=-276.9155
- `CSI_CCSI`: h=2에서 음의 반응. response delta=-2.391069, level-delta effect=-2.3911
- `Imports`: h=3에서 음의 반응. response delta=-0.021083, level-delta effect=-1121521.2044

상대적으로 안정적인 target은 predicted-FX NRMSE 기준 `CSI_CCSI`, `Foreign_Bond_Investment`, `Foreign_Stock_Investment`, `Imports`이다.

해석에 더 주의가 필요한 target은 `Import_Price_Index`, `Industrial_Production`, `KOSPI`, `Trade_Balance`이다.

## 산출물

- `analysis/fx_impact/reports/final/event_panel/anomaly_event_panel.csv`
- `analysis/fx_impact/reports/final/event_panel/local_projection_coefficients.csv`
- `analysis/fx_impact/reports/final/event_panel/event_response_forecasts.csv`
- `analysis/fx_impact/reports/final/event_panel/local_projection_metrics.csv`
- `analysis/fx_impact/reports/final/event_panel/scenario_response_forecasts.csv`
- `analysis/fx_impact/reports/final/event_panel/event_panel_model_selection.csv`
- `analysis/fx_impact/reports/final/event_panel/plots/response_*.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_summary_top_targets.png`
- `analysis/fx_impact/reports/final/plots/response_*.png`에는 바로 확인할 수 있도록 최신 event-time response plot을 복사해 두었다. 기존 calendar-time output table은 삭제하지 않고 유지한다.

## 재현 커맨드

```bash
python analysis/fx_impact/run_final_fx_impact_pipeline.py
```
