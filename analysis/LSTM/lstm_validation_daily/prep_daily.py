import os
import pandas as pd


def prep_daily_data() -> None:
    out_dir = "analysis/lstm_validation_daily"
    os.makedirs(out_dir, exist_ok=True)

    # Core daily liquidity dataset (already aligned to business days)
    daily_liq = pd.read_csv("data/m2/KOR/merged_daily_liquid.csv")
    daily_liq["observation_date"] = pd.to_datetime(daily_liq["observation_date"])

    # Monthly spread is linearly interpolated to daily frequency
    spread_df = pd.read_csv("data/policy_rate/spread_KOR_USA_processed.csv")
    spread_df["observation_date"] = pd.to_datetime(spread_df["observation_date"])

    df = pd.merge(daily_liq, spread_df, on="observation_date", how="left")
    df = df.sort_values("observation_date").reset_index(drop=True)
    df["RATE_SPREAD_KOR_USA"] = df["RATE_SPREAD_KOR_USA"].interpolate(method="linear").ffill().bfill()

    keep_cols = ["observation_date", "USD_KRW", "MMF_total", "RATE_SPREAD_KOR_USA"]
    df = df[keep_cols].dropna().reset_index(drop=True)

    df.to_csv(f"{out_dir}/daily_dataset.csv", index=False)
    print(f"Saved {out_dir}/daily_dataset.csv with {len(df)} rows")


if __name__ == "__main__":
    prep_daily_data()
