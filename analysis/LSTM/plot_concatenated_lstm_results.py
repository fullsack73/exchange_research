from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path("/Applications/dollar_price")
LSTM_DIR = BASE_DIR / "analysis" / "LSTM"

MODELS = [
    {
        "name": "lstm_mmf",
        "counterfactual_col": "pred_counterfactual_flat_mmf",
        "counterfactual_label": "Model B (counterfactual: flat MMF)",
    },
    {
        "name": "lstm_m2_demand_deposit",
        "counterfactual_col": "pred_counterfactual_flat_m2",
        "counterfactual_label": "Model B (counterfactual: flat demand deposit M2)",
    },
]


def _date_tag(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.strftime('%Y_%m')}_to_{end.strftime('%Y_%m')}"


def plot_full_range(pred: pd.DataFrame, model_dir: Path, model_name: str, counterfactual_col: str, counterfactual_label: str) -> Path:
    pred = pred.sort_values(["block_index", "date"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 7))

    for _, block_df in pred.groupby("block_index", sort=True):
        d = block_df["date"]
        ax.plot(d, block_df["actual_fx"], color="black", linewidth=2.0, alpha=0.65)
        ax.plot(d, block_df["pred_model_a"], color="#1f77b4", linewidth=1.4, alpha=0.8)
        ax.plot(d, block_df["pred_model_b"], color="#ff7f0e", linewidth=1.4, alpha=0.8)
        if counterfactual_col in block_df.columns:
            ax.plot(d, block_df[counterfactual_col], color="#2ca02c", linewidth=1.1, alpha=0.7, linestyle="--")

    handles = [
        plt.Line2D([0], [0], color="black", lw=2.0, label="Actual FX"),
        plt.Line2D([0], [0], color="#1f77b4", lw=1.4, label="Model A (spread only)"),
        plt.Line2D([0], [0], color="#ff7f0e", lw=1.4, label="Model B (spread + feature)"),
    ]
    if counterfactual_col in pred.columns:
        handles.append(plt.Line2D([0], [0], color="#2ca02c", lw=1.1, linestyle="--", label=counterfactual_label))

    start = pred["date"].min()
    end = pred["date"].max()
    ax.set_title(
        f"{model_name}: Full Date-Range Plot (all anomaly blocks)\n"
        f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/KRW")
    ax.grid(alpha=0.25)
    ax.legend(handles=handles, loc="best", fontsize=9)

    out_dir = model_dir / "full"
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_name = out_dir / "anomaly_concatenated_blocks_lstm_plot_full.png"
    fig.tight_layout()
    fig.savefig(canonical_name, dpi=140)

    dated_name = out_dir / f"anomaly_{_date_tag(start, end)}_lstm_plot_full.png"
    fig.savefig(dated_name, dpi=140)
    plt.close(fig)

    return canonical_name


def plot_full_range_linear_index(
    pred: pd.DataFrame,
    model_dir: Path,
    model_name: str,
    counterfactual_col: str,
    counterfactual_label: str,
) -> Path:
    pred = pred.sort_values(["block_index", "date"]).reset_index(drop=True)
    pred = pred.copy()
    pred["x_idx"] = range(1, len(pred) + 1)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(pred["x_idx"], pred["actual_fx"], color="black", linewidth=2.0, alpha=0.75, label="Actual FX")
    ax.plot(pred["x_idx"], pred["pred_model_a"], color="#1f77b4", linewidth=1.4, alpha=0.9, label="Model A (spread only)")
    ax.plot(pred["x_idx"], pred["pred_model_b"], color="#ff7f0e", linewidth=1.4, alpha=0.9, label="Model B (spread + feature)")

    if counterfactual_col in pred.columns:
        ax.plot(
            pred["x_idx"],
            pred[counterfactual_col],
            color="#2ca02c",
            linewidth=1.1,
            alpha=0.75,
            linestyle="--",
            label=counterfactual_label,
        )

    boundary_steps = pred.groupby("block_index", sort=True)["x_idx"].max().tolist()
    for s in boundary_steps[:-1]:
        ax.axvline(s, color="gray", alpha=0.2, linewidth=0.8)

    start = pred["date"].min()
    end = pred["date"].max()
    blocks = pred["block_index"].nunique()
    ax.set_title(
        f"{model_name}: Full Plot (linear index x=1..n, all anomaly blocks)\n"
        f"blocks={blocks}, samples={len(pred)}, date span {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    )
    ax.set_xlabel("Linear index (x=1..n)")
    ax.set_ylabel("USD/KRW")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    out_dir = model_dir / "full"
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_name = out_dir / "anomaly_concatenated_blocks_lstm_plot_full_linear.png"
    fig.tight_layout()
    fig.savefig(canonical_name, dpi=140)

    dated_name = out_dir / f"anomaly_{_date_tag(start, end)}_lstm_plot_full_linear.png"
    fig.savefig(dated_name, dpi=140)
    plt.close(fig)

    return canonical_name


def plot_concatenated(pred: pd.DataFrame, model_dir: Path, model_name: str, counterfactual_col: str, counterfactual_label: str) -> Path:
    pred = pred.sort_values(["block_index", "date"]).reset_index(drop=True)
    pred = pred.copy()
    pred["concat_step"] = range(1, len(pred) + 1)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(pred["concat_step"], pred["actual_fx"], color="black", linewidth=2.0, alpha=0.75, label="Actual FX")
    ax.plot(pred["concat_step"], pred["pred_model_a"], color="#1f77b4", linewidth=1.4, alpha=0.9, label="Model A (spread only)")
    ax.plot(pred["concat_step"], pred["pred_model_b"], color="#ff7f0e", linewidth=1.4, alpha=0.9, label="Model B (spread + feature)")

    if counterfactual_col in pred.columns:
        ax.plot(
            pred["concat_step"],
            pred[counterfactual_col],
            color="#2ca02c",
            linewidth=1.1,
            alpha=0.75,
            linestyle="--",
            label=counterfactual_label,
        )

    boundary_steps = pred.groupby("block_index", sort=True)["concat_step"].max().tolist()
    for s in boundary_steps[:-1]:
        ax.axvline(s, color="gray", alpha=0.2, linewidth=0.8)

    start = pred["date"].min()
    end = pred["date"].max()
    blocks = pred["block_index"].nunique()
    ax.set_title(
        f"{model_name}: Concatenated Anomaly Plot (all blocks stitched)\n"
        f"blocks={blocks}, samples={len(pred)}, date span {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    )
    ax.set_xlabel("Concatenated time step")
    ax.set_ylabel("USD/KRW")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    out_dir = model_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_name = out_dir / "anomaly_concatenated_blocks_lstm_plot_eval.png"
    fig.tight_layout()
    fig.savefig(canonical_name, dpi=140)

    dated_name = out_dir / f"anomaly_{_date_tag(start, end)}_lstm_plot_eval.png"
    fig.savefig(dated_name, dpi=140)
    plt.close(fig)

    return canonical_name


def run_one(model_cfg: dict) -> None:
    model_name = model_cfg["name"]
    model_dir = LSTM_DIR / model_name
    pred_path = model_dir / "eval" / "predictions.csv"

    if not pred_path.exists():
        print(f"[SKIP] {model_name}: predictions.csv not found")
        return

    pred = pd.read_csv(pred_path)
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    pred = pred.dropna(subset=["date", "actual_fx", "pred_model_a", "pred_model_b", "block_index"])

    full_out = plot_full_range(
        pred,
        model_dir=model_dir,
        model_name=model_name,
        counterfactual_col=model_cfg["counterfactual_col"],
        counterfactual_label=model_cfg["counterfactual_label"],
    )
    full_linear_out = plot_full_range_linear_index(
        pred,
        model_dir=model_dir,
        model_name=model_name,
        counterfactual_col=model_cfg["counterfactual_col"],
        counterfactual_label=model_cfg["counterfactual_label"],
    )
    eval_out = plot_concatenated(
        pred,
        model_dir=model_dir,
        model_name=model_name,
        counterfactual_col=model_cfg["counterfactual_col"],
        counterfactual_label=model_cfg["counterfactual_label"],
    )

    print(f"[OK] {model_name}: full -> {full_out}")
    print(f"[OK] {model_name}: full_linear -> {full_linear_out}")
    print(f"[OK] {model_name}: eval -> {eval_out}")


def main() -> None:
    for cfg in MODELS:
        run_one(cfg)


if __name__ == "__main__":
    main()
