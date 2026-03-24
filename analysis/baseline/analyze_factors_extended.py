import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BASE_DIR = Path("/Applications/dollar_price")
DATA_PATH = BASE_DIR / "data" / "processed_daily_1995_2026_integrated.csv"
PERIOD_PATH = BASE_DIR / "analysis" / "anomaly" / "period_definition.json"
OUT_DIR = BASE_DIR / "analysis" / "baseline" / "extended_results"

FEATURES = ["policy_spread", "M2_KOR", "M2_USA"]
TARGET = "FX_rate"


def split_periods(df: pd.DataFrame, period_info: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if period_info.get("use_concatenated_blocks"):
        blocks = period_info.get("anomaly_blocks_for_analysis", period_info.get("all_contiguous_blocks", []))
        mask = pd.Series(False, index=df.index)
        for b in blocks:
            s = pd.to_datetime(b["start"])
            e = pd.to_datetime(b["end"])
            mask = mask | ((df["date"] >= s) & (df["date"] <= e))

        anomaly = df[mask].copy()
        baseline = df[~mask].copy()
        return baseline, anomaly

    b_start = pd.to_datetime(period_info["baseline_period"]["start"])
    b_end = pd.to_datetime(period_info["baseline_period"]["end"])
    a_start = pd.to_datetime(period_info["anomaly_period"]["start"])
    a_end = pd.to_datetime(period_info["anomaly_period"]["end"])

    baseline = df[(df["date"] >= b_start) & (df["date"] <= b_end)].copy()
    anomaly = df[(df["date"] >= a_start) & (df["date"] <= a_end)].copy()
    return baseline, anomaly


def fit_period(df_period: pd.DataFrame, name: str) -> dict:
    df_period = df_period.dropna(subset=FEATURES + [TARGET]).sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_period) * 0.8)
    split_idx = max(split_idx, 30)
    split_idx = min(split_idx, len(df_period) - 1)

    train = df_period.iloc[:split_idx]
    test = df_period.iloc[split_idx:]

    x_train, y_train = train[FEATURES], train[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)

    rf_imp = pd.DataFrame({"feature": FEATURES, f"importance_{name}": rf.feature_importances_})
    lr_coef = pd.DataFrame({"feature": FEATURES, f"coef_{name}": lr.coef_})

    return {
        "name": name,
        "rows": int(len(df_period)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "rf_r2": float(r2_score(y_test, rf_pred)),
        "lr_r2": float(r2_score(y_test, lr_pred)),
        "rf_importance": rf_imp,
        "lr_coefficients": lr_coef,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    with open(PERIOD_PATH, "r", encoding="utf-8") as f:
        period_info = json.load(f)

    baseline_df, anomaly_df = split_periods(df, period_info)

    baseline_result = fit_period(baseline_df, "baseline")
    anomaly_result = fit_period(anomaly_df, "anomaly")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    importance_cmp = baseline_result["rf_importance"].merge(
        anomaly_result["rf_importance"], on="feature", how="outer"
    )
    importance_cmp["delta_anomaly_minus_baseline"] = (
        importance_cmp["importance_anomaly"] - importance_cmp["importance_baseline"]
    )
    importance_cmp = importance_cmp.sort_values("delta_anomaly_minus_baseline", ascending=False)

    coef_cmp = baseline_result["lr_coefficients"].merge(
        anomaly_result["lr_coefficients"], on="feature", how="outer"
    )
    coef_cmp["delta_anomaly_minus_baseline"] = coef_cmp["coef_anomaly"] - coef_cmp["coef_baseline"]

    importance_cmp.to_csv(OUT_DIR / "feature_importance_comparison.csv", index=False)
    coef_cmp.to_csv(OUT_DIR / "linear_coefficient_comparison.csv", index=False)

    summary = {
        "baseline": {
            "rows": baseline_result["rows"],
            "train_rows": baseline_result["train_rows"],
            "test_rows": baseline_result["test_rows"],
            "rf_r2": baseline_result["rf_r2"],
            "linear_r2": baseline_result["lr_r2"],
        },
        "anomaly": {
            "rows": anomaly_result["rows"],
            "train_rows": anomaly_result["train_rows"],
            "test_rows": anomaly_result["test_rows"],
            "rf_r2": anomaly_result["rf_r2"],
            "linear_r2": anomaly_result["lr_r2"],
        },
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved extended baseline analysis results")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
