# Target Selection Summary

## Method

- Target candidates are source level columns from `data/integrated_macro_targets.csv`; derived `_MoM`, `_YoY`, and `_lag` columns are excluded from the target universe.
- USD/KRW is transformed as monthly log change. Positive index/flow targets are log-differenced; balances, rates, spreads, survey indexes, and capital-flow variables are first-differenced.
- Evidence combines lagged cross-correlation, Granger causality, distributed-lag OLS/ARDL, ElasticNet lag selection, ExtraTrees permutation importance, and Tree SHAP where available.
- Full-period and anomaly-month results are compared, but anomaly-only estimates are treated cautiously because several targets have limited overlap.

## Selected Final Targets

- `CSI_CCSI`: lag 1, score=0.822, transform=diff
- `Imports`: lag 1, score=0.579, transform=log_diff
- `KOSPI`: lag 1, score=0.573, transform=log_diff
- `Industrial_Production`: lag 2, score=0.553, transform=log_diff
- `Import_Price_Index`: lag 2, score=0.535, transform=log_diff
- `Foreign_Bond_Investment`: lag 6, score=0.513, transform=diff
- `Foreign_Stock_Investment`: lag 2, score=0.398, transform=diff
- `Trade_Balance`: lag 4, score=0.393, transform=diff

## Top Ranking

- #1 `CSI_CCSI`: score=0.822, Granger p=4.258e-08 if available, ARDL lag=1.0, selected=True
- #2 `Imports`: score=0.579, Granger p=2.399e-05 if available, ARDL lag=1.0, selected=True
- #3 `KOSPI`: score=0.573, Granger p=0.00172 if available, ARDL lag=1.0, selected=True
- #4 `Industrial_Production`: score=0.553, Granger p=3.203e-06 if available, ARDL lag=2.0, selected=True
- #5 `Import_Price_Index`: score=0.535, Granger p=0.0001393 if available, ARDL lag=2.0, selected=True
- #6 `Foreign_Bond_Investment`: score=0.513, Granger p=0.06099 if available, ARDL lag=6.0, selected=True
- #7 `Exports`: score=0.498, Granger p=0.001057 if available, ARDL lag=3.0, selected=False
- #8 `WTI_Oil`: score=0.461, Granger p=0.009581 if available, ARDL lag=2.0, selected=False
- #9 `Current_Account`: score=0.443, Granger p=0.1245 if available, ARDL lag=4.0, selected=False
- #10 `Rate_Spread_KOR_USA`: score=0.427, Granger p=0.007752 if available, ARDL lag=4.0, selected=False
- #11 `BSI_All_Industry`: score=0.418, Granger p=0.05383 if available, ARDL lag=6.0, selected=False
- #12 `Foreign_Stock_Investment`: score=0.398, Granger p=0.3128 if available, ARDL lag=2.0, selected=True
