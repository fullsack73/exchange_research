from pathlib import Path

import pandas as pd

BASE_DIR = Path("/Applications/dollar_price")
OUTPUT_PATH = BASE_DIR / "data" / "processed_daily_1995_2026_integrated.csv"


def _load_series(path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "observation_date" not in df.columns:
        raise ValueError(f"Missing observation_date in {path}")
    if value_col not in df.columns:
        raise ValueError(f"Missing {value_col} in {path}")

    out = df[["observation_date", value_col]].copy()
    out["observation_date"] = pd.to_datetime(out["observation_date"], errors="coerce")
    out = out.dropna(subset=["observation_date"]).sort_values("observation_date")
    return out


def main() -> None:
    fx = _load_series(BASE_DIR / "data" / "exchange_rate" / "exchange_rate_processed.csv", "USD_KRW")
    spread = _load_series(BASE_DIR / "data" / "policy_rate" / "spread_KOR_USA_processed.csv", "RATE_SPREAD_KOR_USA")
    m2_kor = _load_series(BASE_DIR / "data" / "m2" / "KOR" / "M2_KOR_processed.csv", "M2_KOR")
    m2_usa = _load_series(BASE_DIR / "data" / "m2" / "USA" / "M2SL.csv", "M2SL")

    start_date = min(
        fx["observation_date"].min(),
        spread["observation_date"].min(),
        m2_kor["observation_date"].min(),
        m2_usa["observation_date"].min(),
    )
    end_date = max(
        fx["observation_date"].max(),
        spread["observation_date"].max(),
        m2_kor["observation_date"].max(),
        m2_usa["observation_date"].max(),
    )

    daily_index = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})

    merged = daily_index.merge(fx.rename(columns={"observation_date": "date", "USD_KRW": "FX_rate"}), on="date", how="left")
    merged = merged.merge(
        spread.rename(columns={"observation_date": "date", "RATE_SPREAD_KOR_USA": "policy_spread"}),
        on="date",
        how="left",
    )

    merged["FX_rate"] = merged["FX_rate"].ffill().bfill()
    merged["policy_spread"] = merged["policy_spread"].ffill().bfill()

    m2_kor_daily = m2_kor.rename(columns={"observation_date": "date"}).set_index("date").reindex(merged["date"])
    m2_kor_daily["M2_KOR"] = m2_kor_daily["M2_KOR"].interpolate(method="linear", limit_direction="both")

    m2_usa_daily = m2_usa.rename(columns={"observation_date": "date", "M2SL": "M2_USA"}).set_index("date").reindex(merged["date"])
    m2_usa_daily["M2_USA"] = m2_usa_daily["M2_USA"].interpolate(method="linear", limit_direction="both")

    merged["M2_KOR"] = m2_kor_daily["M2_KOR"].values
    merged["M2_USA"] = m2_usa_daily["M2_USA"].values

    merged = merged.dropna(subset=["FX_rate", "policy_spread", "M2_KOR", "M2_USA"])
    merged = merged.sort_values("date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {OUTPUT_PATH}")
    print(f"Rows: {len(merged):,}")
    print(f"Range: {merged['date'].min().date()} ~ {merged['date'].max().date()}")
    print(f"Missing by column:\n{merged.isna().sum()}")


if __name__ == "__main__":
    main()
