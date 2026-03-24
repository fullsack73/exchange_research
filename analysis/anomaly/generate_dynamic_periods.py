import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/Applications/dollar_price")
OUTPUT_PATH = BASE_DIR / "analysis" / "anomaly" / "dynamic_periods.json"

ROLLING_WINDOW = 30
CORR_THRESHOLD = -0.2
MIN_ANOMALY_DAYS = 20


def build_dynamic_periods() -> dict:
    spread_path = BASE_DIR / "data" / "policy_rate" / "spread_KOR_USA_processed.csv"
    fx_path = BASE_DIR / "data" / "exchange_rate" / "exchange_rate_processed.csv"

    spread = pd.read_csv(spread_path)
    spread["observation_date"] = pd.to_datetime(spread["observation_date"])

    fx = pd.read_csv(fx_path)
    fx["observation_date"] = pd.to_datetime(fx["observation_date"])

    df = fx.merge(spread, on="observation_date", how="inner").sort_values("observation_date")
    df = df[["observation_date", "USD_KRW", "RATE_SPREAD_KOR_USA"]].dropna().reset_index(drop=True)

    df["rolling_corr"] = (
        df["USD_KRW"]
        .rolling(window=ROLLING_WINDOW)
        .corr(df["RATE_SPREAD_KOR_USA"])
    )
    df = df[df["rolling_corr"].replace([float("inf"), float("-inf")], pd.NA).notna()].copy()
    df["is_anomaly"] = df["rolling_corr"] < CORR_THRESHOLD

    anomaly_rows = df[df["is_anomaly"]].copy()

    periods = []
    if not anomaly_rows.empty:
        anomaly_rows = anomaly_rows.reset_index().rename(columns={"index": "row_idx"})
        anomaly_rows["group"] = (
            anomaly_rows["row_idx"].diff().fillna(1).ne(1).cumsum()
        )

        for _, g in anomaly_rows.groupby("group"):
            start = g["observation_date"].min()
            end = g["observation_date"].max()
            days = int(len(g))
            mean_corr = float(g["rolling_corr"].mean())
            periods.append(
                {
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "days": days,
                    "mean_rolling_corr": mean_corr,
                }
            )

    candidates = [p for p in periods if p["days"] >= MIN_ANOMALY_DAYS]
    if not candidates:
        candidates = periods

    if not candidates:
        raise ValueError("No anomaly periods found under current rolling correlation settings.")

    latest_segment = sorted(candidates, key=lambda p: (p["end"], p["days"]), reverse=True)[0]
    primary = {
        "start": latest_segment["start"],
        "end": df["observation_date"].max().strftime("%Y-%m-%d"),
        "seed_end": latest_segment["end"],
        "days": int((df["observation_date"].max() - pd.to_datetime(latest_segment["start"])) .days + 1),
        "seed_mean_rolling_corr": latest_segment["mean_rolling_corr"],
    }

    anomaly_start = pd.to_datetime(primary["start"])
    full_start = df["observation_date"].min()
    normal_end = anomaly_start - pd.Timedelta(days=1)

    payload = {
        "config": {
            "rolling_window_days": ROLLING_WINDOW,
            "corr_threshold": CORR_THRESHOLD,
            "min_anomaly_days": MIN_ANOMALY_DAYS,
        },
        "data_range": {
            "start": full_start.strftime("%Y-%m-%d"),
            "end": df["observation_date"].max().strftime("%Y-%m-%d"),
        },
        "primary_anomaly_period": primary,
        "normal_period": {
            "start": full_start.strftime("%Y-%m-%d"),
            "end": normal_end.strftime("%Y-%m-%d"),
        },
        "all_anomaly_periods": periods,
    }
    return payload


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dynamic_periods()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    p = payload["primary_anomaly_period"]
    print(f"Saved {OUTPUT_PATH}")
    print(f"Primary anomaly: {p['start']} ~ {p['end']} ({p['days']} days)")


if __name__ == "__main__":
    main()
