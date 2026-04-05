import copy
import argparse
import itertools
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fcfcfc",
    "axes.edgecolor": "#d0d0d0",
    "axes.linewidth": 0.8,
    "grid.color": "#d9d9d9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.55,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

parser = argparse.ArgumentParser(description="Hybrid log-return multi-step training with HPO")
parser.add_argument("target", nargs="?", default="mmf", choices=["mmf", "m2"], help="target dataset type")
parser.add_argument(
    "--hpo-level",
    default="standard",
    choices=["quick", "standard", "aggressive"],
    help="HPO search budget preset",
)
args = parser.parse_args()
target_type = args.target

BASE_DIR = Path("/Applications/dollar_price")
PERIOD_DEF_PATH = BASE_DIR / "analysis" / "anomaly" / "period_definition.json"

if target_type == "m2":
    OUTPUT_DIR = BASE_DIR / "analysis" / "LSTM" / "Hybrid" / "log_multistep_m2"
    DATA_PATH = BASE_DIR / "analysis" / "LSTM" / "lstm_m2_demand_deposit" / "daily_dataset_m2_demand_deposit.csv"
else:
    OUTPUT_DIR = BASE_DIR / "analysis" / "LSTM" / "Hybrid" / "log_multistep_mmf"
    DATA_PATH = BASE_DIR / "analysis" / "LSTM" / "lstm_mmf" / "daily_dataset.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "full", exist_ok=True)
os.makedirs(OUTPUT_DIR / "eval", exist_ok=True)
os.makedirs(OUTPUT_DIR / "hpo", exist_ok=True)

HORIZON = 5

HPO_PRESETS = {
    "quick": {
        "trials_a": 10,
        "trials_b": 10,
        "epochs_tune": 25,
        "epochs_final": 70,
        "patience": 6,
    },
    "standard": {
        "trials_a": 24,
        "trials_b": 24,
        "epochs_tune": 40,
        "epochs_final": 100,
        "patience": 8,
    },
    "aggressive": {
        "trials_a": 40,
        "trials_b": 40,
        "epochs_tune": 60,
        "epochs_final": 140,
        "patience": 10,
    },
}
HPO_CONFIG = HPO_PRESETS[args.hpo_level]


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len - horizon + 1):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + horizon, 0])
    return np.array(xs), np.array(ys)


def load_period_definition() -> dict:
    with open(PERIOD_DEF_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_anomaly_concatenated(df: pd.DataFrame, period_def: dict) -> pd.DataFrame:
    blocks = period_def.get("anomaly_blocks_for_analysis", [])
    rows = []
    for idx, block in enumerate(blocks, start=1):
        start = pd.to_datetime(block["start"])
        end = pd.to_datetime(block["end"])
        blk = df[(df["observation_date"] >= start) & (df["observation_date"] <= end)].copy()
        if blk.empty:
            continue
        blk["block_index"] = idx
        blk["block_start"] = start
        blk["block_end"] = end
        rows.append(blk)
    if not rows:
        raise ValueError("No anomaly blocks found.")
    out = pd.concat(rows, axis=0, ignore_index=True)
    return out.sort_values(["block_index", "observation_date"]).reset_index(drop=True)


def build_multistep_forecast_frame(
    df_source: pd.DataFrame,
    pred_a_scaled: np.ndarray,
    pred_b_scaled: np.ndarray,
    scaler_y: StandardScaler,
    seq_length: int,
    horizon: int,
    start_offset: int,
    min_target_start: int | None = None,
):
    dates = pd.to_datetime(df_source["observation_date"])
    actual_prices = df_source["USD_KRW"].to_numpy()
    arima_log = df_source["ARIMA_Log_pred"].to_numpy()
    has_blocks = "block_index" in df_source.columns

    rows = []
    for pred_idx in range(0, len(pred_a_scaled), horizon):
        global_start = start_offset + pred_idx
        target_start = global_start + seq_length
        target_end = target_start + horizon

        if target_end > len(df_source):
            break
        if min_target_start is not None and target_start < min_target_start:
            continue

        if has_blocks:
            block_values = df_source["block_index"].iloc[target_start:target_end]
            if block_values.nunique() != 1:
                continue
            block_index = int(block_values.iloc[0])
        else:
            block_index = 1

        base_price = actual_prices[target_start - 1]
        arima_logs_h = arima_log[target_start:target_end]
        resid_a = scaler_y.inverse_transform(pred_a_scaled[pred_idx].reshape(-1, 1)).flatten()
        resid_b = scaler_y.inverse_transform(pred_b_scaled[pred_idx].reshape(-1, 1)).flatten()

        pred_arima = base_price * np.exp(np.cumsum(arima_logs_h))
        pred_a = base_price * np.exp(np.cumsum(arima_logs_h + resid_a))
        pred_b = base_price * np.exp(np.cumsum(arima_logs_h + resid_b))

        for step in range(horizon):
            row_idx = target_start + step
            rows.append(
                {
                    "date": dates.iloc[row_idx],
                    "row_index": row_idx,
                    "actual_fx": float(actual_prices[row_idx]),
                    "pred_arima": float(pred_arima[step]),
                    "pred_model_a": float(pred_a[step]),
                    "pred_model_b": float(pred_b[step]),
                    "block_index": block_index,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["date", "row_index", "actual_fx", "pred_arima", "pred_model_a", "pred_model_b", "block_index", "concat_step"]
        )
    if has_blocks:
        out = out.sort_values(["block_index", "row_index"]).reset_index(drop=True)
        # Align forecast x-position to the same concatenated index used by actual_df.
        out["concat_step"] = out["row_index"] + 1
    else:
        out = out.sort_values("date").reset_index(drop=True)
    return out


def plot_multistep_forecast(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    use_concat_axis: bool = False,
    split_date: pd.Timestamp | None = None,
):
    fig, ax = plt.subplots(figsize=(16, 7))

    if use_concat_axis:
        actual_sorted = actual_df.sort_values(["block_index", "observation_date"]).reset_index(drop=True).copy()
        actual_sorted["concat_step"] = np.arange(1, len(actual_sorted) + 1)
        ax.plot(actual_sorted["concat_step"], actual_sorted["USD_KRW"], color="#2f2f2f", linewidth=1.9, label="Actual USD/KRW")

        x_forecast = forecast_df["concat_step"]
        ax.set_xlabel("Concatenated time step")
        if not forecast_df.empty:
            x_min = int(forecast_df["concat_step"].min())
            x_max = int(forecast_df["concat_step"].max())
            pad = max(3, int((x_max - x_min) * 0.04))
            ax.set_xlim(max(1, x_min - pad), x_max + pad)
        ax.xaxis.set_major_locator(MaxNLocator(12))
    else:
        actual_sorted = actual_df.sort_values("observation_date").reset_index(drop=True)
        actual_x = pd.to_datetime(actual_sorted["observation_date"])
        ax.plot(actual_x, actual_sorted["USD_KRW"], color="#2f2f2f", linewidth=1.9, label="Actual USD/KRW")
        x_forecast = pd.to_datetime(forecast_df["date"])
        ax.set_xlabel("Date")
        if not forecast_df.empty:
            x_min = pd.to_datetime(forecast_df["date"]).min()
            x_max = pd.to_datetime(forecast_df["date"]).max()
            ax.set_xlim(x_min - pd.Timedelta(days=14), x_max + pd.Timedelta(days=14))
        if split_date is not None:
            ax.axvline(split_date, color="#888888", linestyle=":", linewidth=1.0, alpha=0.8, label="Train/Test split")
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.plot(x_forecast, forecast_df["pred_arima"], color="#7f7f7f", linestyle="--", linewidth=1.4, label="ARIMA baseline")
    ax.plot(x_forecast, forecast_df["pred_model_a"], color="#1f77b4", linewidth=1.8, label="Model A (LSTM)")
    ax.plot(x_forecast, forecast_df["pred_model_b"], color="#d62728", linewidth=1.8, label="Model B (CNN-LSTM)")

    ax.set_title(title)
    ax.set_ylabel("USD/KRW")
    if use_concat_axis:
        ax.grid(axis="y", alpha=0.28)
    else:
        ax.grid(alpha=0.35)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.96)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def prepare_log_data_for_period(df_period: pd.DataFrame, seq_length: int = 10, horizon: int = 5, test_ratio: float = 0.2, seasonal_order=(0, 0, 0, 0)):
    target_col = "USD_KRW"
    feature_cols = ["M2_수시입출식저축성예금", "RATE_SPREAD_KOR_USA"] if target_type == "m2" else ["MMF_total", "RATE_SPREAD_KOR_USA"]

    df_period = df_period.copy()
    
    # 1. Calculate Log Return
    # Using np.log(Y_t / Y_{t-1})
    df_period["Log_Return"] = np.log(df_period[target_col] / df_period[target_col].shift(1))
    df_period = df_period.dropna(subset=["Log_Return"]).reset_index(drop=True)

    n_total = len(df_period)
    test_size = max(int(n_total * test_ratio), 40)
    train_size = n_total - test_size

    # Fit SARIMAX(1,0,1) on Log_Return (since it's already stationary)
    train_ts = df_period["Log_Return"].iloc[:train_size]
    arima_model = SARIMAX(train_ts, order=(1, 0, 1), seasonal_order=seasonal_order)
    arima_result = arima_model.fit(disp=False)

    res_full = arima_result.apply(df_period["Log_Return"])
    df_period["ARIMA_Log_pred"] = res_full.fittedvalues
    df_period["Residuals"] = df_period["Log_Return"] - df_period["ARIMA_Log_pred"]

    train_df = df_period.iloc[:train_size].copy()
    test_df = df_period.iloc[train_size:].copy()

    X_cols = feature_cols + ["Residuals"]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    train_X = scaler_X.fit_transform(train_df[X_cols].values)
    train_y = scaler_y.fit_transform(train_df[["Residuals"]].values)

    test_overlap = pd.concat([train_df.iloc[-seq_length:], test_df], axis=0)
    test_X = scaler_X.transform(test_overlap[X_cols].values)
    test_y = scaler_y.transform(test_overlap[["Residuals"]].values)
    
    full_X = scaler_X.transform(df_period[X_cols].values)
    full_y = scaler_y.transform(df_period[["Residuals"]].values)

    X_train_seq, y_train_seq = create_sequences(train_X, train_y, seq_length, horizon)
    # Note: test test_overlap doesn't generate targets properly for the end bounds unless handled
    X_test_seq, y_test_seq = create_sequences(test_X, test_y, seq_length, horizon)
    X_full_seq, y_full_seq = create_sequences(full_X, full_y, seq_length, horizon)

    return {
        "X_train": X_train_seq, "y_train": y_train_seq,
        "X_test": X_test_seq, "y_test": y_test_seq,
        "X_full": X_full_seq, "y_full": y_full_seq,
        "df_ready": df_period, "seq_length": seq_length, "horizon": horizon,
        "test_df": test_df.reset_index(drop=True),
        "scaler_y": scaler_y, "train_rows": len(train_df),
        "test_rows": len(test_df), "all_rows": len(df_period),
    }


class LSTM_Multi_Step(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2, horizon=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class CNN_LSTM_Multi_Step(nn.Module):
    def __init__(self, input_dim, cnn_filters=16, kernel_size=3, hidden_dim=32, num_layers=1, dropout=0.2, horizon=5):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim + cnn_filters, horizon)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_feat = self.dropout(lstm_out[:, -1, :])

        x_cnn = x.transpose(1, 2)
        c_out = self.relu(self.conv1d(x_cnn))
        cnn_feat = c_out.mean(dim=-1)

        combined = torch.cat((lstm_feat, cnn_feat), dim=1)
        out = self.fc(combined)
        return out


class Hybrid_Model_Trainer:
    def __init__(self, model, lr=0.001, weight_decay=1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, X_train, y_train, epochs=100, batch_size=32, patience=5):
        val_size = max(int(len(X_train) * 0.1), 1)
        X_t, y_t = torch.FloatTensor(X_train[:-val_size]), torch.FloatTensor(y_train[:-val_size])
        X_v, y_v = torch.FloatTensor(X_train[-val_size:]), torch.FloatTensor(y_train[-val_size:])
        
        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=batch_size, shuffle=False)
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        for epoch in range(epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(bx)
                loss = self.criterion(preds, by)
                loss.backward()
                self.optimizer.step()
                
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    val_preds = self.model(bx)
                    val_loss += self.criterion(val_preds, by).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        self.model.load_state_dict(best_model_wts)
        return float(best_loss)

    def predict(self, X_test):
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            preds = self.model(X_test_t).cpu().numpy()
        return preds


def build_grids():
    # Model A (LSTM): wide candidate pool, then deterministic sampling by trial budget.
    cand_a = []
    for hidden_dim, lr, dropout, batch_size in itertools.product(
        [16, 32, 48, 64, 96],
        [0.0003, 0.0005, 0.001, 0.0015, 0.002],
        [0.1, 0.2, 0.3],
        [32, 64],
    ):
        cand_a.append(
            {
                "hidden_dim": hidden_dim,
                "lr": lr,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": HPO_CONFIG["epochs_tune"],
                "patience": HPO_CONFIG["patience"],
            }
        )

    # Model B (CNN-LSTM): include CNN-specific search dimensions.
    cand_b = []
    for hidden_dim, cnn_filters, kernel_size, lr, dropout, batch_size in itertools.product(
        [16, 32, 48],
        [8, 12, 16, 20, 24],
        [3, 5, 7],
        [0.0005, 0.001, 0.0015],
        [0.1, 0.2],
        [32, 64],
    ):
        cand_b.append(
            {
                "hidden_dim": hidden_dim,
                "cnn_filters": cnn_filters,
                "kernel_size": kernel_size,
                "lr": lr,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": HPO_CONFIG["epochs_tune"],
                "patience": HPO_CONFIG["patience"],
            }
        )

    rng = np.random.default_rng(42)
    idx_a = rng.permutation(len(cand_a))[: HPO_CONFIG["trials_a"]]
    idx_b = rng.permutation(len(cand_b))[: HPO_CONFIG["trials_b"]]
    gA = [cand_a[i] for i in idx_a]
    gB = [cand_b[i] for i in idx_b]
    return gA, gB


def run_period(period_name: str, period_df: pd.DataFrame):
    print(f"\n[{period_name}] Processing Period (Log Return Multi-Step)")
    seq_length, horizon = 10, HORIZON
    prepared = prepare_log_data_for_period(period_df, seq_length=seq_length, horizon=horizon, test_ratio=0.2)

    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]
    X_full = prepared["X_full"]
    scaler_y = prepared["scaler_y"]
    test_df = prepared["test_df"]
    df_ready = prepared["df_ready"]
    input_dim = X_train.shape[2]

    # Expanded HPO with saved trial logs.
    gridA, gridB = build_grids()
    best_a_cfg, best_val_a = gridA[0], float('inf')
    best_b_cfg, best_val_b = gridB[0], float('inf')
    trials_a, trials_b = [], []

    print(f"Tuning Model A... ({len(gridA)} trials)")
    for i, cfg in enumerate(gridA, start=1):
        trial_start = time.time()
        torch.manual_seed(1000 + i)
        np.random.seed(1000 + i)

        net = LSTM_Multi_Step(
            input_dim,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            horizon=horizon,
        )
        trainer = Hybrid_Model_Trainer(net, lr=cfg["lr"])
        val_rmse = np.sqrt(
            trainer.fit(
                X_train,
                y_train,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                patience=cfg["patience"],
            )
        )
        trials_a.append(
            {
                "trial": i,
                **cfg,
                "val_rmse_scaled": float(val_rmse),
                "duration_sec": float(time.time() - trial_start),
            }
        )
        if val_rmse < best_val_a:
            best_val_a = val_rmse
            best_a_cfg = cfg

    print(f"Tuning Model B... ({len(gridB)} trials)")
    for i, cfg in enumerate(gridB, start=1):
        trial_start = time.time()
        torch.manual_seed(2000 + i)
        np.random.seed(2000 + i)

        net = CNN_LSTM_Multi_Step(
            input_dim,
            cnn_filters=cfg["cnn_filters"],
            kernel_size=cfg["kernel_size"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            horizon=horizon,
        )
        trainer = Hybrid_Model_Trainer(net, lr=cfg["lr"])
        val_rmse = np.sqrt(
            trainer.fit(
                X_train,
                y_train,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                patience=cfg["patience"],
            )
        )
        trials_b.append(
            {
                "trial": i,
                **cfg,
                "val_rmse_scaled": float(val_rmse),
                "duration_sec": float(time.time() - trial_start),
            }
        )
        if val_rmse < best_val_b:
            best_val_b = val_rmse
            best_b_cfg = cfg

    pd.DataFrame(trials_a).sort_values("val_rmse_scaled").to_csv(
        OUTPUT_DIR / "hpo" / f"{period_name}_model_a_trials.csv", index=False
    )
    pd.DataFrame(trials_b).sort_values("val_rmse_scaled").to_csv(
        OUTPUT_DIR / "hpo" / f"{period_name}_model_b_trials.csv", index=False
    )
    with open(OUTPUT_DIR / "hpo" / f"{period_name}_hpo_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "hpo_level": args.hpo_level,
                "target": target_type,
                "period": period_name,
                "model_a_best": {**best_a_cfg, "best_val_rmse_scaled": float(best_val_a)},
                "model_b_best": {**best_b_cfg, "best_val_rmse_scaled": float(best_val_b)},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Training Final Models...")
    net_A = LSTM_Multi_Step(
        input_dim,
        hidden_dim=best_a_cfg["hidden_dim"],
        dropout=best_a_cfg["dropout"],
        horizon=horizon,
    )
    trainer_A = Hybrid_Model_Trainer(net_A, lr=best_a_cfg["lr"])
    trainer_A.fit(
        X_train,
        y_train,
        epochs=HPO_CONFIG["epochs_final"],
        batch_size=best_a_cfg["batch_size"],
        patience=max(best_a_cfg["patience"], 8),
    )

    net_B = CNN_LSTM_Multi_Step(
        input_dim,
        cnn_filters=best_b_cfg["cnn_filters"],
        kernel_size=best_b_cfg["kernel_size"],
        hidden_dim=best_b_cfg["hidden_dim"],
        dropout=best_b_cfg["dropout"],
        horizon=horizon,
    )
    trainer_B = Hybrid_Model_Trainer(net_B, lr=best_b_cfg["lr"])
    trainer_B.fit(
        X_train,
        y_train,
        epochs=HPO_CONFIG["epochs_final"],
        batch_size=best_b_cfg["batch_size"],
        patience=max(best_b_cfg["patience"], 8),
    )

    # Predictions
    pred_a_scaled = trainer_A.predict(X_test) # shape: (N, horizon)
    pred_b_scaled = trainer_B.predict(X_test)
    pred_a_full_scaled = trainer_A.predict(X_full)
    pred_b_full_scaled = trainer_B.predict(X_full)

    # Build stitched forecast frames for cleaner plots
    train_size = prepared["train_rows"]
    split_idx = train_size
    split_date = pd.to_datetime(df_ready["observation_date"].iloc[split_idx]) if split_idx < len(df_ready) else None

    eval_forecast_df = build_multistep_forecast_frame(
        df_ready,
        pred_a_scaled,
        pred_b_scaled,
        scaler_y,
        seq_length=seq_length,
        horizon=horizon,
        start_offset=train_size - seq_length,
        min_target_start=train_size,
    )
    full_forecast_df = build_multistep_forecast_frame(
        df_ready,
        pred_a_full_scaled,
        pred_b_full_scaled,
        scaler_y,
        seq_length=seq_length,
        horizon=horizon,
        start_offset=0,
        min_target_start=None,
    )
    
    # We will pick non-overlapping windows of horizon to reconstruct
    # e.g., i=0, i=5, i=10
    actual_prices = df_ready["USD_KRW"].values
    arima_log = df_ready["ARIMA_Log_pred"].values
    
    # Let's reconstruct the 5-day absolute price forecast for the first subset of test
    # Test indices start after train_size
    all_dates = pd.to_datetime(df_ready.get("observation_date", range(len(df_ready))))
    eval_plot_path = OUTPUT_DIR / "eval" / f"{period_name}_5_day_forecast_samples.png"
    full_plot_path = OUTPUT_DIR / "full" / f"{period_name}_5_day_forecast_full.png"

    if "block_index" in df_ready.columns and df_ready["block_index"].nunique() > 1:
        plot_multistep_forecast(
            df_ready,
            full_forecast_df,
            full_plot_path,
            f"{period_name}: Full stitched 5-day forecasts",
            use_concat_axis=True,
        )
        plot_multistep_forecast(
            df_ready,
            eval_forecast_df,
            eval_plot_path,
            f"{period_name}: Out-of-sample stitched 5-day forecasts",
            use_concat_axis=True,
        )
    else:
        plot_multistep_forecast(
            df_ready,
            full_forecast_df,
            full_plot_path,
            f"{period_name}: Full stitched 5-day forecasts",
            use_concat_axis=False,
            split_date=split_date,
        )
        plot_multistep_forecast(
            df_ready,
            eval_forecast_df,
            eval_plot_path,
            f"{period_name}: Out-of-sample stitched 5-day forecasts",
            use_concat_axis=False,
            split_date=split_date,
        )

    # Keep the existing sampled RMSE summary, but compute it from the stitched eval windows.
    rmse_a_list, rmse_b_list, rmse_naive_list = [], [], []
    samples_to_plot = min(len(X_test) // horizon, 10)
    for i in range(samples_to_plot):
        idx = i * horizon * 2
        if idx + horizon > len(X_test):
            break

        start_t = train_size + idx
        base_price = actual_prices[start_t - 1]
        true_h_days = actual_prices[start_t : start_t + horizon]
        pred_scaled_vecA = pred_a_scaled[idx].reshape(-1, 1)
        pred_scaled_vecB = pred_b_scaled[idx].reshape(-1, 1)
        pred_residA = scaler_y.inverse_transform(pred_scaled_vecA).flatten()
        pred_residB = scaler_y.inverse_transform(pred_scaled_vecB).flatten()
        arima_logs_H = arima_log[start_t : start_t + horizon]
        log_pred_A = arima_logs_H + pred_residA
        log_pred_B = arima_logs_H + pred_residB
        abs_pred_A = base_price * np.exp(np.cumsum(log_pred_A))
        abs_pred_B = base_price * np.exp(np.cumsum(log_pred_B))
        abs_naive = np.full(horizon, base_price)

        rmse_a_list.append(np.sqrt(mean_squared_error(true_h_days, abs_pred_A)))
        rmse_b_list.append(np.sqrt(mean_squared_error(true_h_days, abs_pred_B)))
        rmse_naive_list.append(np.sqrt(mean_squared_error(true_h_days, abs_naive)))

    print(f"\n--- {horizon}-Day RMSE Evaluation (Sampled Windows) ---")
    print(f"Model A (LSTM) Avg {horizon}-Day RMSE: {np.mean(rmse_a_list):.4f}")
    print(f"Model B (CNN-LSTM) Avg {horizon}-Day RMSE: {np.mean(rmse_b_list):.4f}")
    print(f"Naive Baseline Avg {horizon}-Day RMSE: {np.mean(rmse_naive_list):.4f}")

    return {
        "rmse_a": np.mean(rmse_a_list),
        "rmse_b": np.mean(rmse_b_list),
        "rmse_naive": np.mean(rmse_naive_list),
        "hpo_level": args.hpo_level,
        "best_a": best_a_cfg,
        "best_b": best_b_cfg,
        "plot": str(eval_plot_path.relative_to(BASE_DIR)),
        "plot_full": str(full_plot_path.relative_to(BASE_DIR)),
    }

def main():
    df = pd.read_csv(DATA_PATH)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    period_def = load_period_definition()
    range_start = pd.to_datetime(period_def["data_range"]["start"])
    range_end = pd.to_datetime(period_def["data_range"]["end"])
    df_full = df[(df["observation_date"] >= range_start) & (df["observation_date"] <= range_end)].copy()

    df_anomaly_concat = build_anomaly_concatenated(df_full, period_def)

    print(f"Running Multi-Step Log Return Models... (target={target_type}, hpo_level={args.hpo_level})")
    res_full = run_period("full_1995_2026", df_full)
    res_full["period"] = "full_1995_2026"

    res_anomaly = run_period("anomaly_concatenated_blocks", df_anomaly_concat)
    res_anomaly["period"] = "anomaly_concatenated_blocks"

    results_data = [res_full, res_anomaly]

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    lines = [
        f"Hybrid {target_type.upper()} Multi-Step (Log Return) ARIMA-LSTM vs ARIMA-CNN-LSTM",
        f"Data range: {range_start.date()} to {range_end.date()}",
        "Anomaly definition: period_definition.json -> anomaly_blocks_for_analysis",
        "",
    ]
    for r in results_data:
        lines.append(f"[{r['period']}]")
        lines.append(f"Model A (LSTM) Avg 5-Day RMSE: {r['rmse_a']:.4f}")
        lines.append(f"Model B (CNN-LSTM) Avg 5-Day RMSE: {r['rmse_b']:.4f}")
        lines.append(f"Naive Baseline Avg 5-Day RMSE: {r['rmse_naive']:.4f}")
        lines.append(f"HPO level: {r['hpo_level']}")
        lines.append(f"Best A cfg: {r['best_a']}")
        lines.append(f"Best B cfg: {r['best_b']}")
        lines.append(f"Plot: {r['plot']}")

        better = "A" if r['rmse_a'] < r['rmse_b'] else "B"
        if r['rmse_naive'] < min(r['rmse_a'], r['rmse_b']):
            better = "Naive"
        lines.append(f"Better model: {better}")
        lines.append("")

    with open(OUTPUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nDONE.")

if __name__ == "__main__":
    main()
