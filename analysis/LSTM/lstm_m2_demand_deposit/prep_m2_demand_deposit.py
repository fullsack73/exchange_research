import os
import pandas as pd


def prep_m2_demand_deposit_data() -> None:
    out_dir = "analysis/LSTM/lstm_m2_demand_deposit"
    os.makedirs(out_dir, exist_ok=True)

    # Build daily base from raw exchange-rate and spread sources so the range can start in 2010.
    usd = pd.read_csv("data/exchange_rate/USD_KRW_processed.csv")
    usd["observation_date"] = pd.to_datetime(usd["observation_date"])
    usd = usd[["observation_date", "USD_KRW"]].sort_values("observation_date").reset_index(drop=True)

    spread_df = pd.read_csv("data/policy_rate/spread_KOR_USA_processed.csv")
    spread_df["observation_date"] = pd.to_datetime(spread_df["observation_date"])
    spread_df = spread_df[["observation_date", "RATE_SPREAD_KOR_USA"]]

    daily_base = pd.merge(usd, spread_df, on="observation_date", how="left")
    daily_base["RATE_SPREAD_KOR_USA"] = (
        daily_base["RATE_SPREAD_KOR_USA"].interpolate(method="linear").ffill().bfill()
    )

    start_date = pd.Timestamp("2010-12-01")
    end_date = pd.Timestamp("2026-03-16")
    daily_base = daily_base[
        (daily_base["observation_date"] >= start_date)
        & (daily_base["observation_date"] <= end_date)
    ].copy()
    daily_base = daily_base.sort_values("observation_date").reset_index(drop=True)

    # Monthly M2 details
    m2_details = pd.read_csv("data/m2/KOR/M2_details_processed.csv")
    m2_details["observation_date"] = pd.to_datetime(m2_details["observation_date"])

    # Keep only target M2 component
    target_col = "M2_수시입출식저축성예금"
    if target_col not in m2_details.columns:
        raise ValueError(f"Column not found: {target_col}")

    m2_subset = m2_details[["observation_date", target_col]].copy()
    m2_subset = m2_subset.sort_values("observation_date").reset_index(drop=True)

    # Interpolate monthly value onto the base daily dates.
    m2_daily = m2_subset.set_index("observation_date").reindex(daily_base["observation_date"])
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
