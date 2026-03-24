import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/Applications/dollar_price")
DATA_DIR = BASE_DIR / "data"

DATE_COL_PATTERN = re.compile(r"^\d{4}/\d{2}(/\d{2})?$")


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "-": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if DATE_COL_PATTERN.match(str(c))]


def _parse_date_labels(labels: list[str]) -> pd.Series:
    parsed = []
    for label in labels:
        s = str(label)
        if len(s) == 7:
            parsed.append(pd.to_datetime(s, format="%Y/%m", errors="coerce"))
        elif len(s) == 10:
            parsed.append(pd.to_datetime(s, format="%Y/%m/%d", errors="coerce"))
        else:
            parsed.append(pd.to_datetime(s, errors="coerce"))
    return pd.Series(parsed)


def parse_wide_single_series(
    file_path: Path,
    filters: dict[str, str],
    value_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype=str)
    date_cols = _date_columns(df)

    target = df.copy()
    for col, val in filters.items():
        if col not in target.columns:
            raise ValueError(f"Column '{col}' not found in {file_path}")
        target = target[target[col] == val]

    if target.empty:
        raise ValueError(f"No rows matched {filters} in {file_path}")

    row = target.iloc[0]
    out = pd.DataFrame(
        {
            "observation_date": _parse_date_labels(date_cols),
            value_name: _clean_numeric(row[date_cols]).values,
        }
    )
    out = out.dropna(subset=["observation_date", value_name]).sort_values("observation_date")
    return out.reset_index(drop=True)


def _sanitize_m2_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    name = name.replace("(", "").replace(")", "")
    name = name.replace("/", "_")
    name = re.sub(r"\s+", "_", name)
    return f"M2_{name}"


def parse_m2_details(files: list[Path]) -> pd.DataFrame:
    frames = []

    for file_path in files:
        df = pd.read_csv(file_path, dtype=str)
        date_cols = _date_columns(df)

        if "계정항목" not in df.columns:
            raise ValueError(f"Column '계정항목' not found in {file_path}")

        long_rows = []
        for _, row in df.iterrows():
            raw_name = str(row.get("계정항목", "")).strip()
            if not raw_name or raw_name.lower() == "nan":
                continue

            col_name = _sanitize_m2_name(raw_name)
            vals = _clean_numeric(row[date_cols])

            temp = pd.DataFrame(
                {
                    "observation_date": _parse_date_labels(date_cols),
                    "variable": col_name,
                    "value": vals.values,
                }
            )
            long_rows.append(temp)

        if not long_rows:
            continue

        long_df = pd.concat(long_rows, ignore_index=True)
        long_df = long_df.dropna(subset=["observation_date", "value"])

        wide_df = (
            long_df.pivot_table(
                index="observation_date",
                columns="variable",
                values="value",
                aggfunc="last",
            )
            .reset_index()
            .sort_values("observation_date")
        )
        frames.append(wide_df)

    if not frames:
        raise ValueError("No M2 detail rows parsed")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("observation_date")
    merged = merged.groupby("observation_date", as_index=False).last()
    return merged


def parse_m2_total(files: list[Path]) -> pd.DataFrame:
    totals = []

    for file_path in files:
        df = pd.read_csv(file_path, dtype=str)
        date_cols = _date_columns(df)

        if "계정항목" not in df.columns:
            raise ValueError(f"Column '계정항목' not found in {file_path}")

        account = df["계정항목"].fillna("").astype(str)
        total_mask = account.str.strip().str.startswith("M2(")

        if total_mask.any():
            values = _clean_numeric(df.loc[total_mask].iloc[0][date_cols])
        else:
            # Legacy file has only component rows; reconstruct total by component sum.
            component_mask = account.str.startswith("  ")
            if not component_mask.any():
                component_mask = ~total_mask
            comp = df.loc[component_mask, date_cols].apply(_clean_numeric, axis=0)
            values = comp.sum(axis=0, skipna=True)

        temp = pd.DataFrame(
            {
                "observation_date": _parse_date_labels(date_cols),
                "M2_KOR": values.values,
            }
        )
        temp = temp.dropna(subset=["observation_date", "M2_KOR"])
        totals.append(temp)

    if not totals:
        raise ValueError("No M2 total rows parsed")

    out = pd.concat(totals, ignore_index=True)
    out = out.sort_values("observation_date")
    out = out.groupby("observation_date", as_index=False).last()
    return out


def parse_mmf_daily(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=2, dtype=str)
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"Unexpected MMF file format: {file_path}")

    date_col = df.columns[0]
    value_col = df.columns[1]

    out = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(df[date_col], errors="coerce"),
            "MMF_total": _clean_numeric(df[value_col]),
        }
    )
    out = out.dropna(subset=["observation_date", "MMF_total"])
    out = out.sort_values("observation_date").reset_index(drop=True)
    return out


def main() -> None:
    # 1) Daily exchange rate
    exchange_raw = DATA_DIR / "exchange_rate" / "주요국 통화의 대원화환율_24092601.csv"
    exchange = parse_wide_single_series(
        exchange_raw,
        filters={"계정항목": "원/미국달러(매매기준율)"},
        value_name="USD_KRW",
    )

    exchange_out_a = DATA_DIR / "exchange_rate" / "exchange_rate_processed.csv"
    exchange_out_b = DATA_DIR / "exchange_rate" / "USD_KRW_processed.csv"
    exchange.to_csv(exchange_out_a, index=False)
    exchange.to_csv(exchange_out_b, index=False)

    # 2) Daily Korea base rate
    kor_rate_raw = DATA_DIR / "policy_rate" / "KOR" / "한국은행 기준금리 및 여수신금리_24094328.csv"
    kor_rate = parse_wide_single_series(
        kor_rate_raw,
        filters={"계정항목": "한국은행 기준금리"},
        value_name="BASE_RATE_KOR",
    )

    kor_rate_out = DATA_DIR / "policy_rate" / "KOR" / "base_rate_KOR_processed.csv"
    kor_rate.to_csv(kor_rate_out, index=False)

    # 3) Daily US policy rate
    us_rate_path = DATA_DIR / "policy_rate" / "USA" / "FEDFUNDS.csv"
    us_rate = pd.read_csv(us_rate_path)
    us_rate["observation_date"] = pd.to_datetime(us_rate["observation_date"], errors="coerce")
    us_rate = us_rate[["observation_date", "FEDFUNDS"]].dropna().sort_values("observation_date")

    # 4) Daily spread and theoretical forward rate
    merged = exchange.merge(kor_rate, on="observation_date", how="inner")
    merged = merged.merge(us_rate, on="observation_date", how="inner")

    spread = merged[["observation_date"]].copy()
    spread["RATE_SPREAD_KOR_USA"] = merged["BASE_RATE_KOR"] - merged["FEDFUNDS"]
    spread_out = DATA_DIR / "policy_rate" / "spread_KOR_USA_processed.csv"
    spread.to_csv(spread_out, index=False)

    fwd = merged[["observation_date"]].copy()
    r_kor = merged["BASE_RATE_KOR"] / 100.0
    r_usa = merged["FEDFUNDS"] / 100.0
    fwd["THEORETICAL_FWD_RATE"] = merged["USD_KRW"] * (1.0 + r_kor) / (1.0 + r_usa)
    fwd_out = DATA_DIR / "exchange_rate" / "theoretical_fwd_rate_processed.csv"
    fwd.to_csv(fwd_out, index=False)

    # 5) M2 details (monthly)
    m2_files = [
        DATA_DIR / "m2" / "KOR" / "M2_1995_to_2004.csv",
        DATA_DIR / "m2" / "KOR" / "M2_2004_to_2026.csv",
    ]
    m2_details = parse_m2_details(m2_files)
    m2_out = DATA_DIR / "m2" / "KOR" / "M2_details_processed.csv"
    m2_details.to_csv(m2_out, index=False)

    m2_total = parse_m2_total(m2_files)

    m2_total_out = DATA_DIR / "m2" / "KOR" / "M2_KOR_processed.csv"
    m2_total.to_csv(m2_total_out, index=False)

    # 6) Daily liquidity merge for MMF LSTM
    mmf_raw = DATA_DIR / "m2" / "KOR" / "MMF" / "MMF_daily.csv"
    mmf_daily = parse_mmf_daily(mmf_raw)

    mmf_monthly_col = "M2_MMF"
    if mmf_monthly_col not in m2_details.columns:
        mmf_like = [c for c in m2_details.columns if "MMF" in c]
        if not mmf_like:
            raise ValueError("Could not find MMF column in M2 details")
        mmf_monthly_col = mmf_like[0]

    mmf_monthly = m2_details[["observation_date", mmf_monthly_col]].rename(
        columns={mmf_monthly_col: "MMF_monthly"}
    )

    merged_liq = exchange.merge(mmf_daily, on="observation_date", how="left")
    merged_liq = merged_liq.merge(mmf_monthly, on="observation_date", how="left")
    merged_liq["MMF_monthly"] = merged_liq["MMF_monthly"].interpolate(method="linear", limit_direction="both")
    merged_liq["MMF_total"] = merged_liq["MMF_total"].fillna(merged_liq["MMF_monthly"])
    merged_liq["MMF_total"] = merged_liq["MMF_total"].fillna(0.0)
    merged_liq = merged_liq.drop(columns=["MMF_monthly"])
    merged_liq = merged_liq.sort_values("observation_date").reset_index(drop=True)

    merged_liq_out = DATA_DIR / "m2" / "KOR" / "merged_daily_liquid.csv"
    merged_liq.to_csv(merged_liq_out, index=False)

    print("Rebuild completed.")
    print(f"exchange rows: {len(exchange):,}")
    print(f"spread rows: {len(spread):,}")
    print(f"m2 details rows: {len(m2_details):,}")
    print(f"m2 total rows: {len(m2_total):,}")
    print(f"merged liquidity rows: {len(merged_liq):,}")


if __name__ == "__main__":
    main()
