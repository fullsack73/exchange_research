# Roadmap to Predict Variables Impacted by USD/KRW Exchange Rate

This roadmap outlines the steps needed to build a predictive model for variables influenced by movements in the USD/KRW exchange rate, based on our current analytical goals.

## Step 1: Identify Target Variables (Y)
### Objective
Determine the macroeconomic variables most sensitive to exchange rate changes, focusing on:
- Import Price Index
- Consumer Price Index (CPI)
- Trade Balance
- Foreign Capital Inflow
- KOSPI Index (Korea Composite Stock Price Index)

### Action
Conduct an exploratory analysis to establish variable candidates for prediction.

---

## Step 2: Lead-Lag and Causality Analysis
### Objective
Establish temporal relationships and causal impacts of exchange rate changes on macroeconomic variables.

### Action
- Perform Cross-Correlation Analysis for various lag periods.
- Conduct Granger Causality Tests to confirm predictive relationships.

---

## Step 3: Error Propagation Mitigation
### Objective
Address dependencies between the MMF \u0026 LSTM prediction error and the target variable prediction stability.

### Action
Validate whether FX predictions can be reliably passed to subsequent models.

---
## Step 4: Baseline Model Validation
### Objective
Establish benchmarks with known FX data to test target models under controlled conditions before introducing prediction-driven FX inputs.

### Action
Use VAR or ARIMAX as control models for validation.