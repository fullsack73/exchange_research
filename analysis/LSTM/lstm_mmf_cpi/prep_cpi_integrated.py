import os
import pandas as pd
from pathlib import Path


def prep_cpi_integrated_data() -> None:
    out_dir = "analysis/lstm_cpi_integration"
    os.makedirs(out_dir, exist_ok=True)

    # Load daily liquidity dataset (already aligned to business days)
    daily_liq = pd.read_csv("analysis/lstm_validation_daily/daily_dataset.csv")
    daily_liq["observation_date"] = pd.to_datetime(daily_liq["observation_date"])

    # Load CPI monthly data with all components
    cpi_monthly = pd.read_csv("data/CPI/USA/CPI_components/final_processed_data.csv", index_col=0, parse_dates=True)
    
    # Top 5 CPI features from SHAP importance
    top_5_cpi_features = [
        "Energy_YoY_lag2",
        "Food_YoY",
        "Shelter_YoY_lag2",
        "Durables_YoY_lag3",
        "Headline_MoM_lag1"
    ]
    
    # Extract top 5 CPI features, fill NaN in case
    cpi_subset = cpi_monthly[top_5_cpi_features].fillna(method="ffill").fillna(method="bfill")
    cpi_subset = cpi_subset.reset_index()
    cpi_subset.columns = ["observation_date"] + top_5_cpi_features
    
    # Merge daily data with monthly CPI data by date (left join on daily business dates)
    # Interpolate CPI linearly to daily frequency
    cpi_subset["observation_date"] = pd.to_datetime(cpi_subset["observation_date"])
    
    # Create daily date range
    date_range = pd.date_range(start=daily_liq["observation_date"].min(), 
                                end=daily_liq["observation_date"].max(), 
                                freq="D")
    cpi_daily = cpi_subset.set_index("observation_date").reindex(date_range)
    cpi_daily = cpi_daily.interpolate(method="linear", limit_direction="both")
    cpi_daily = cpi_daily.reset_index()
    cpi_daily.columns = ["observation_date"] + top_5_cpi_features
    
    # Merge with daily data
    df = pd.merge(daily_liq, cpi_daily, on="observation_date", how="left")
    df = df.sort_values("observation_date").reset_index(drop=True)
    df = df.dropna()
    
    keep_cols = ["observation_date", "USD_KRW", "MMF_total", "RATE_SPREAD_KOR_USA"] + top_5_cpi_features
    df = df[keep_cols].reset_index(drop=True)
    
    df.to_csv(f"{out_dir}/daily_dataset_cpi_integrated.csv", index=False)
    print(f"Saved {out_dir}/daily_dataset_cpi_integrated.csv with {len(df)} rows")


if __name__ == "__main__":
    prep_cpi_integrated_data()
