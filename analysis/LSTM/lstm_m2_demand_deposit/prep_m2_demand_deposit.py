import os
import pandas as pd


def prep_m2_demand_deposit_data() -> None:
    out_dir = "analysis/lstm_m2_demand_deposit"
    os.makedirs(out_dir, exist_ok=True)

    # Base daily data with USD_KRW and spread (already daily-aligned)
    daily_base = pd.read_csv("analysis/lstm_validation_daily/daily_dataset.csv")
    daily_base["observation_date"] = pd.to_datetime(daily_base["observation_date"])

    # Monthly M2 details
    m2_details = pd.read_csv("data/m2/KOR/M2_details_processed.csv")
    m2_details["observation_date"] = pd.to_datetime(m2_details["observation_date"])

    # Keep only target M2 component
    target_col = "M2_수시입출식저축성예금"
    if target_col not in m2_details.columns:
        raise ValueError(f"Column not found: {target_col}")

    m2_subset = m2_details[["observation_date", target_col]].copy()
    m2_subset = m2_subset.sort_values("observation_date").reset_index(drop=True)

    # Interpolate monthly value to daily frequency
    date_range = pd.date_range(
        start=daily_base["observation_date"].min(),
        end=daily_base["observation_date"].max(),
        freq="D",
    )

    m2_daily = m2_subset.set_index("observation_date").reindex(date_range)
    m2_daily = m2_daily.interpolate(method="linear", limit_direction="both")
    m2_daily = m2_daily.reset_index().rename(columns={"index": "observation_date"})

    # Merge and keep required columns
    df = pd.merge(daily_base, m2_daily, on="observation_date", how="left")
    df = df.sort_values("observation_date").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)

    keep_cols = ["observation_date", "USD_KRW", "RATE_SPREAD_KOR_USA", target_col]
    df = df[keep_cols]

    out_path = f"{out_dir}/daily_dataset_m2_demand_deposit.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df)} rows")


if __name__ == "__main__":
    prep_m2_demand_deposit_data()
