import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/Applications/dollar_price")
INPUT_PATH = BASE_DIR / "data" / "processed_daily_1995_2026_integrated.csv"
OUTPUT_PATH = BASE_DIR / "analysis" / "anomaly" / "period_definition.json"

ROLLING_WINDOW = 30
CORR_THRESHOLD = -0.2
MIN_CONTIGUOUS_DAYS = 30


def _find_contiguous_blocks(flag_df: pd.DataFrame) -> list[dict]:
    blocks = []
    if flag_df.empty:
        return blocks

    start = flag_df.iloc[0]["date"]
    prev = start
    length = 1

    for i in range(1, len(flag_df)):
        cur = flag_df.iloc[i]["date"]
        if (cur - prev).days == 1:
            length += 1
        else:
            blocks.append({"start": start, "end": prev, "days": length})
            start = cur
            length = 1
        prev = cur

    blocks.append({"start": start, "end": prev, "days": length})
    return blocks


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "FX_rate", "policy_spread"]).sort_values("date").reset_index(drop=True)

    df["rolling_corr"] = df["FX_rate"].rolling(ROLLING_WINDOW).corr(df["policy_spread"])
    decoupled = df[df["rolling_corr"] <= CORR_THRESHOLD][["date", "rolling_corr"]].copy().reset_index(drop=True)

    blocks = _find_contiguous_blocks(decoupled)
    valid_blocks = [b for b in blocks if b["days"] >= MIN_CONTIGUOUS_DAYS]

    if not valid_blocks:
        raise ValueError("No contiguous anomaly block >= 30 days found with current settings.")

    first_block = valid_blocks[0]
    anomaly_start = first_block["start"]
    anomaly_end = df["date"].max()
    blocks_sorted = sorted(blocks, key=lambda b: b["start"])
    concat_total_days = int(sum(b["days"] for b in blocks_sorted))

    payload = {
        "config": {
            "rolling_window_days": ROLLING_WINDOW,
            "corr_threshold": CORR_THRESHOLD,
            "min_contiguous_days": MIN_CONTIGUOUS_DAYS,
        },
        "data_range": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
            "rows": int(len(df)),
        },
        "anomaly_seed_block": {
            "start": first_block["start"].strftime("%Y-%m-%d"),
            "end": first_block["end"].strftime("%Y-%m-%d"),
            "days": int(first_block["days"]),
        },
        "anomaly_period": {
            "start": anomaly_start.strftime("%Y-%m-%d"),
            "end": anomaly_end.strftime("%Y-%m-%d"),
            "days": int((anomaly_end - anomaly_start).days + 1),
        },
        "use_concatenated_blocks": True,
        "anomaly_concatenated_summary": {
            "blocks": int(len(blocks_sorted)),
            "total_days": concat_total_days,
            "first_start": blocks_sorted[0]["start"].strftime("%Y-%m-%d") if blocks_sorted else None,
            "last_end": blocks_sorted[-1]["end"].strftime("%Y-%m-%d") if blocks_sorted else None,
        },
        "baseline_period": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": (anomaly_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        },
        "anomaly_blocks_for_analysis": [
            {
                "start": b["start"].strftime("%Y-%m-%d"),
                "end": b["end"].strftime("%Y-%m-%d"),
                "days": int(b["days"]),
            }
            for b in blocks_sorted
        ],
        "all_contiguous_blocks": [
            {
                "start": b["start"].strftime("%Y-%m-%d"),
                "end": b["end"].strftime("%Y-%m-%d"),
                "days": int(b["days"]),
            }
            for b in blocks_sorted
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {OUTPUT_PATH}")
    print(f"Anomaly seed block: {payload['anomaly_seed_block']}")
    print(f"Anomaly period: {payload['anomaly_period']}")


if __name__ == "__main__":
    main()
