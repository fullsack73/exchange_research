import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/Applications/dollar_price")
DATA_PATH = BASE_DIR / "data" / "processed_daily_1995_2026_integrated.csv"
PERIOD_PATH = BASE_DIR / "analysis" / "anomaly" / "period_definition.json"
OUT_PATH = BASE_DIR / "analysis" / "anomaly" / "anomaly_concatenated_dataset.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    with open(PERIOD_PATH, "r", encoding="utf-8") as f:
        period = json.load(f)

    blocks = period.get("anomaly_blocks_for_analysis", period.get("all_contiguous_blocks", []))
    if not blocks:
        raise ValueError("No anomaly blocks found in period definition")

    parts = []
    for b in blocks:
        s = pd.to_datetime(b["start"])
        e = pd.to_datetime(b["end"])
        chunk = df[(df["date"] >= s) & (df["date"] <= e)].copy()
        if not chunk.empty:
            chunk["block_start"] = s
            chunk["block_end"] = e
            parts.append(chunk)

    if not parts:
        raise ValueError("No rows selected by anomaly blocks")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["block_start", "date"]).reset_index(drop=True)
    out["anomaly_seq_day"] = out.index + 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved {OUT_PATH}")
    print(f"Rows: {len(out):,}")
    print(f"Blocks: {len(parts):,}")


if __name__ == "__main__":
    main()
