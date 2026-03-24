import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path("/Applications/dollar_price")
DATA_PATH = BASE_DIR / "data" / "processed_daily_1995_2026_integrated.csv"
PERIOD_PATH = BASE_DIR / "analysis" / "anomaly" / "period_definition.json"

FEATURES = ["policy_spread", "M2_KOR", "M2_USA"]
TARGET = "FX_rate"


def _run_one_period(df: pd.DataFrame, period_name: str, out_dir: Path) -> None:
    df = df.dropna(subset=FEATURES + [TARGET]).sort_values("date").reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"{period_name}: not enough rows ({len(df)})")

    split_idx = int(len(df) * 0.8)
    split_idx = max(split_idx, 100)
    split_idx = min(split_idx, len(df) - 1)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    x_train = train[FEATURES]
    y_train = train[TARGET]
    x_test = test[FEATURES]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(x_train, y_train)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(x_test)

    imp = pd.DataFrame({
        "feature": FEATURES,
        "rf_importance": rf.feature_importances_,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out_dir / "importance_summary.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, x_test, feature_names=FEATURES, show=False)
    plt.title(f"SHAP Summary - {period_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png")
    plt.close()

    for feature in FEATURES:
        plt.figure()
        shap.dependence_plot(feature, shap_values, x_test, show=False)
        plt.title(f"SHAP Dependence - {period_name} - {feature}")
        plt.tight_layout()
        plt.savefig(out_dir / f"dependence_{feature}.png")
        plt.close()

    meta = {
        "period": period_name,
        "rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    with open(PERIOD_PATH, "r", encoding="utf-8") as f:
        period_info = json.load(f)

    if period_info.get("use_concatenated_blocks"):
        blocks = period_info.get("anomaly_blocks_for_analysis", period_info.get("all_contiguous_blocks", []))
        mask = pd.Series(False, index=df.index)
        for b in blocks:
            s = pd.to_datetime(b["start"])
            e = pd.to_datetime(b["end"])
            mask = mask | ((df["date"] >= s) & (df["date"] <= e))
        anomaly_df = df[mask].copy()
        baseline_df = df[~mask].copy()
    else:
        b_start = pd.to_datetime(period_info["baseline_period"]["start"])
        b_end = pd.to_datetime(period_info["baseline_period"]["end"])
        a_start = pd.to_datetime(period_info["anomaly_period"]["start"])
        a_end = pd.to_datetime(period_info["anomaly_period"]["end"])

        baseline_df = df[(df["date"] >= b_start) & (df["date"] <= b_end)].copy()
        anomaly_df = df[(df["date"] >= a_start) & (df["date"] <= a_end)].copy()

    baseline_out = BASE_DIR / "analysis" / "shap_ml" / "results_baseline"
    anomaly_out = BASE_DIR / "analysis" / "shap_ml" / "results_anomaly"
    baseline_out.mkdir(parents=True, exist_ok=True)
    anomaly_out.mkdir(parents=True, exist_ok=True)

    _run_one_period(baseline_df, "baseline", baseline_out)
    _run_one_period(anomaly_df, "anomaly", anomaly_out)

    print("Saved extended SHAP outputs for baseline and anomaly periods")


if __name__ == "__main__":
    main()
