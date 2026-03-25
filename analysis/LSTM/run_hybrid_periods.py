import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

BASE_DIR = Path("/Applications/dollar_price")
OUTPUT_DIR = BASE_DIR / "analysis" / "LSTM" / "hybrid_mmf"
DATA_PATH = BASE_DIR / "analysis" / "LSTM" / "lstm_mmf" / "daily_dataset.csv"
PERIOD_DEF_PATH = BASE_DIR / "analysis" / "anomaly" / "period_definition.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "full", exist_ok=True)
os.makedirs(OUTPUT_DIR / "eval", exist_ok=True)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
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
        raise ValueError("No anomaly blocks found in dataset using period_definition.json")

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out.sort_values(["block_index", "observation_date"]).reset_index(drop=True)


def prepare_hybrid_data_for_period(df_period: pd.DataFrame, seq_length: int = 10, test_ratio: float = 0.2):
    target_col = "USD_KRW"
    feature_cols = ["MMF_total", "RATE_SPREAD_KOR_USA"]

    n_total = len(df_period)
    test_size = max(int(n_total * test_ratio), 40)
    train_size = n_total - test_size
    if train_size <= seq_length + 10:
        raise ValueError(f"Not enough train rows: {train_size}")

    train_ts = df_period[target_col].iloc[:train_size]
    arima_model = ARIMA(train_ts, order=(1, 1, 1))
    arima_result = arima_model.fit()

    df_period = df_period.copy()
    res_full = arima_result.apply(df_period[target_col])
    df_period["ARIMA_pred"] = res_full.fittedvalues
    df_period["Residuals"] = df_period[target_col] - df_period["ARIMA_pred"]

    # Stabilize ARIMA warm-up rows.
    df_period = df_period.iloc[5:].reset_index(drop=True)

    train_size_adj = len(df_period) - test_size
    if train_size_adj <= seq_length + 10:
        raise ValueError(f"Not enough adjusted train rows: {train_size_adj}")

    train_df = df_period.iloc[:train_size_adj].copy()
    test_df = df_period.iloc[train_size_adj:].copy()

    X_cols = feature_cols + ["Residuals"]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    train_X = scaler_X.fit_transform(train_df[X_cols].values)
    train_y = scaler_y.fit_transform(train_df[["Residuals"]].values)

    full_X = scaler_X.transform(df_period[X_cols].values)
    full_y = scaler_y.transform(df_period[["Residuals"]].values)

    test_overlap = pd.concat([train_df.iloc[-seq_length:], test_df], axis=0)
    test_X = scaler_X.transform(test_overlap[X_cols].values)
    test_y = scaler_y.transform(test_overlap[["Residuals"]].values)

    X_train_seq, y_train_seq = create_sequences(train_X, train_y, seq_length)
    X_test_seq, y_test_seq = create_sequences(test_X, test_y, seq_length)
    X_full_seq, y_full_seq = create_sequences(full_X, full_y, seq_length)

    return {
        "X_train": X_train_seq,
        "y_train": y_train_seq,
        "X_test": X_test_seq,
        "y_test": y_test_seq,
        "X_full": X_full_seq,
        "y_full": y_full_seq,
        "df_ready": df_period,
        "seq_length": seq_length,
        "test_df": test_df.reset_index(drop=True),
        "scaler_y": scaler_y,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "all_rows": len(df_period),
    }

# --- 2. Model A: ARIMA-LSTM Hybrid ---
class LSTM_Residual_Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2):
        super(LSTM_Residual_Predictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

class ARIMA_LSTM_Model:
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2, lr=0.001, weight_decay=1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = LSTM_Residual_Predictor(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def fit(self, X_train, y_train, epochs=100, batch_size=32, patience=10):
        val_size = max(int(len(X_train) * 0.1), 1) # At least 1 element
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
            val_loss /= max(len(val_loader), 1)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        self.model.load_state_dict(best_model_wts)
        
    def predict(self, X_test):
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            preds = self.model(X_test_t).cpu().numpy()
        return preds

# --- 3. Model B: ARIMA-CNN-LSTM Hybrid Architecture ---
class CNN_LSTM_Residual_Predictor(nn.Module):
    def __init__(self, input_dim, cnn_filters=16, kernel_size=3, hidden_dim=32, num_layers=1, dropout=0.2):
        super(CNN_LSTM_Residual_Predictor, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_filters, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        c_out = self.relu(self.conv1d(x))
        c_out = c_out.transpose(1, 2)
        lstm_out, _ = self.lstm(c_out)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

class ARIMA_CNN_LSTM_Model(ARIMA_LSTM_Model):
    def __init__(self, input_dim, cnn_filters=16, kernel_size=3, hidden_dim=32, num_layers=1, dropout=0.2, lr=0.001, weight_decay=1e-5):
        super(ARIMA_CNN_LSTM_Model, self).__init__(input_dim, hidden_dim, num_layers, dropout, lr, weight_decay)
        self.model = CNN_LSTM_Residual_Predictor(input_dim, cnn_filters, kernel_size, hidden_dim, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

def build_eval_predictions(test_df: pd.DataFrame, pred_a: np.ndarray, pred_b: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(test_df["observation_date"]).values,
            "actual_fx": test_df["USD_KRW"].values,
            "pred_arima": test_df["ARIMA_pred"].values,
            "pred_model_a": pred_a,
            "pred_model_b": pred_b,
        }
    )
    for c in ["block_index", "block_start", "block_end"]:
        if c in test_df.columns:
            out[c] = test_df[c].values
    return out


def plot_full_regular(period_name: str, df_period: pd.DataFrame, pred_df: pd.DataFrame) -> Path:
    out_path = OUTPUT_DIR / "full" / f"{period_name}_hybrid_plot_full.png"
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df_period["observation_date"], df_period["USD_KRW"], color="black", linewidth=1.4, label="Actual USD/KRW")
    ax.plot(pred_df["date"], pred_df["pred_arima"], color="grey", linestyle="--", alpha=0.8, label="ARIMA Baseline")
    ax.plot(pred_df["date"], pred_df["pred_model_a"], color="#1f77b4", alpha=0.9, label="Model A (ARIMA-LSTM)")
    ax.plot(pred_df["date"], pred_df["pred_model_b"], color="#d62728", alpha=0.9, label="Model B (ARIMA-CNN-LSTM)")
    ax.set_title(f"{period_name}: Full Range (1995-2026)")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/KRW")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_eval_regular(period_name: str, pred_df: pd.DataFrame) -> Path:
    out_path = OUTPUT_DIR / "eval" / f"{period_name}_hybrid_plot_eval.png"
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(pred_df["date"], pred_df["actual_fx"], color="black", linewidth=1.6, label="Actual USD/KRW")
    ax.plot(pred_df["date"], pred_df["pred_arima"], color="grey", linestyle="--", alpha=0.8, label="ARIMA Baseline")
    ax.plot(pred_df["date"], pred_df["pred_model_a"], color="#1f77b4", alpha=0.9, label="Model A (ARIMA-LSTM)")
    ax.plot(pred_df["date"], pred_df["pred_model_b"], color="#d62728", alpha=0.9, label="Model B (ARIMA-CNN-LSTM)")
    ax.set_title(f"{period_name}: Test/Eval Range")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/KRW")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_anomaly_block_full(pred_df: pd.DataFrame, period_name: str) -> Path:
    out_path = OUTPUT_DIR / "full" / f"{period_name}_hybrid_plot_full.png"
    fig, ax = plt.subplots(figsize=(16, 7))
    for _, blk in pred_df.groupby("block_index", sort=True):
        d = pd.to_datetime(blk["date"])
        ax.plot(d, blk["actual_fx"], color="black", linewidth=2.0, alpha=0.65)
        ax.plot(d, blk["pred_model_a"], color="#1f77b4", linewidth=1.4, alpha=0.85)
        ax.plot(d, blk["pred_model_b"], color="#d62728", linewidth=1.4, alpha=0.85)
    handles = [
        plt.Line2D([0], [0], color="black", lw=2.0, label="Actual FX"),
        plt.Line2D([0], [0], color="#1f77b4", lw=1.4, label="Model A (ARIMA-LSTM)"),
        plt.Line2D([0], [0], color="#d62728", lw=1.4, label="Model B (ARIMA-CNN-LSTM)"),
    ]
    s = pd.to_datetime(pred_df["date"]).min()
    e = pd.to_datetime(pred_df["date"]).max()
    ax.set_title(
        f"{period_name}: Full Date-Range Plot (all anomaly blocks)\n"
        f"{s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/KRW")
    ax.grid(alpha=0.25)
    ax.legend(handles=handles, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_anomaly_block_eval(pred_df: pd.DataFrame, period_name: str) -> Path:
    out_path = OUTPUT_DIR / "eval" / f"{period_name}_hybrid_plot_eval.png"
    pred_df = pred_df.sort_values(["block_index", "date"]).reset_index(drop=True).copy()
    pred_df["concat_step"] = np.arange(1, len(pred_df) + 1)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(pred_df["concat_step"], pred_df["actual_fx"], color="black", linewidth=2.0, alpha=0.75, label="Actual FX")
    ax.plot(pred_df["concat_step"], pred_df["pred_model_a"], color="#1f77b4", linewidth=1.4, alpha=0.9, label="Model A (ARIMA-LSTM)")
    ax.plot(pred_df["concat_step"], pred_df["pred_model_b"], color="#d62728", linewidth=1.4, alpha=0.9, label="Model B (ARIMA-CNN-LSTM)")

    boundary_steps = pred_df.groupby("block_index", sort=True)["concat_step"].max().tolist()
    for s in boundary_steps[:-1]:
        ax.axvline(s, color="gray", alpha=0.2, linewidth=0.8)

    blocks = pred_df["block_index"].nunique()
    d0 = pd.to_datetime(pred_df["date"]).min().strftime("%Y-%m-%d")
    d1 = pd.to_datetime(pred_df["date"]).max().strftime("%Y-%m-%d")
    ax.set_title(
        f"{period_name}: Concatenated Eval Plot (anomaly blocks stitched)\n"
        f"blocks={blocks}, samples={len(pred_df)}, date span {d0} to {d1}"
    )
    ax.set_xlabel("Concatenated time step")
    ax.set_ylabel("USD/KRW")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def run_period(period_name: str, period_df: pd.DataFrame, is_anomaly_blocks: bool = False):
    print(f"\\n[{period_name}] Processing Period")
    seq_length = 10
    prepared = prepare_hybrid_data_for_period(period_df, seq_length=seq_length, test_ratio=0.2)

    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_test = prepared["X_test"]
    X_full = prepared["X_full"]
    scaler_y = prepared["scaler_y"]
    test_df = prepared["test_df"]
    df_ready = prepared["df_ready"]
    seq_length = prepared["seq_length"]

    input_dim = X_train.shape[2]
    print(f"Train rows: {prepared['train_rows']}, Test rows: {prepared['test_rows']}")

    print(f"[{period_name}] Training Model A (ARIMA-LSTM)...")
    model_A = ARIMA_LSTM_Model(input_dim=input_dim, hidden_dim=32, dropout=0.2)
    model_A.fit(X_train, y_train, epochs=100)
    pred_a_scaled = model_A.predict(X_test)
    pred_a_resid = scaler_y.inverse_transform(pred_a_scaled).flatten()

    print(f"[{period_name}] Training Model B (ARIMA-CNN-LSTM)...")
    model_B = ARIMA_CNN_LSTM_Model(input_dim=input_dim, cnn_filters=16, kernel_size=3, hidden_dim=32, dropout=0.2)
    model_B.fit(X_train, y_train, epochs=100)
    pred_b_scaled = model_B.predict(X_test)
    pred_b_resid = scaler_y.inverse_transform(pred_b_scaled).flatten()

    pred_a_full_scaled = model_A.predict(X_full)
    pred_b_full_scaled = model_B.predict(X_full)
    pred_a_full_resid = scaler_y.inverse_transform(pred_a_full_scaled).flatten()
    pred_b_full_resid = scaler_y.inverse_transform(pred_b_full_scaled).flatten()

    test_arima = test_df["ARIMA_pred"].values
    actual = test_df["USD_KRW"].values
    final_a = test_arima + pred_a_resid
    final_b = test_arima + pred_b_resid

    df_full_aligned = df_ready.iloc[seq_length:].reset_index(drop=True)
    full_arima = df_full_aligned["ARIMA_pred"].values
    full_final_a = full_arima + pred_a_full_resid
    full_final_b = full_arima + pred_b_full_resid

    rmse_base = float(np.sqrt(mean_squared_error(actual, test_arima)))
    rmse_a = float(np.sqrt(mean_squared_error(actual, final_a)))
    rmse_b = float(np.sqrt(mean_squared_error(actual, final_b)))
    mae_a = float(mean_absolute_error(actual, final_a))
    mae_b = float(mean_absolute_error(actual, final_b))
    better = "A" if rmse_a < rmse_b else "B"

    pred_df = build_eval_predictions(test_df, final_a, final_b)
    pred_df_full = build_eval_predictions(df_full_aligned, full_final_a, full_final_b)

    if is_anomaly_blocks:
        plot_full_path = plot_anomaly_block_full(pred_df_full, period_name)
        plot_eval_path = plot_anomaly_block_eval(pred_df, period_name)
        pred_csv_path = OUTPUT_DIR / "eval" / "predictions.csv"
        pred_df_full.to_csv(pred_csv_path, index=False)

        by_block = []
        for bid, blk in pred_df_full.groupby("block_index", sort=True):
            if len(blk) < 2:
                continue
            by_block.append(
                {
                    "block_index": int(bid),
                    "rows": int(len(blk)),
                    "start": str(pd.to_datetime(blk["date"]).min().date()),
                    "end": str(pd.to_datetime(blk["date"]).max().date()),
                    "rmse_model_a": float(np.sqrt(mean_squared_error(blk["actual_fx"], blk["pred_model_a"]))),
                    "rmse_model_b": float(np.sqrt(mean_squared_error(blk["actual_fx"], blk["pred_model_b"]))),
                    "mae_model_a": float(mean_absolute_error(blk["actual_fx"], blk["pred_model_a"])),
                    "mae_model_b": float(mean_absolute_error(blk["actual_fx"], blk["pred_model_b"])),
                }
            )
        pd.DataFrame(by_block).to_csv(OUTPUT_DIR / "eval" / "block_metrics.csv", index=False)
    else:
        plot_full_path = plot_full_regular(period_name, period_df, pred_df_full)
        plot_eval_path = plot_eval_regular(period_name, pred_df)

    print(f"Base ARIMA RMSE: {rmse_base:.4f}")
    print(f"Model A RMSE: {rmse_a:.4f} | MAE: {mae_a:.4f}")
    print(f"Model B RMSE: {rmse_b:.4f} | MAE: {mae_b:.4f}")
    print(f"Better model for {period_name}: {better}")

    return {
        "period": period_name,
        "rows": prepared["all_rows"],
        "train_rows": prepared["train_rows"],
        "test_rows": prepared["test_rows"],
        "rmse_base_arima": rmse_base,
        "rmse_model_a": rmse_a,
        "rmse_model_b": rmse_b,
        "mae_model_a": mae_a,
        "mae_model_b": mae_b,
        "better_model": better,
        "plot_full": str(plot_full_path.relative_to(BASE_DIR)),
        "plot_eval": str(plot_eval_path.relative_to(BASE_DIR)),
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

    results = []
    results.append(run_period("full_1995_2026", df_full, is_anomaly_blocks=False))
    results.append(run_period("anomaly_concatenated_blocks", df_anomaly_concat, is_anomaly_blocks=True))

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [
        "Hybrid ARIMA-LSTM vs ARIMA-CNN-LSTM",
        f"Data range: {range_start.date()} to {range_end.date()}",
        "Anomaly definition: period_definition.json -> anomaly_blocks_for_analysis",
        "",
    ]
    for r in results:
        lines.append(f"[{r['period']}]")
        lines.append(f"Rows: {r['rows']} (train={r['train_rows']}, test={r['test_rows']})")
        lines.append(f"RMSE base: {r['rmse_base_arima']:.4f}")
        lines.append(f"RMSE A: {r['rmse_model_a']:.4f} | MAE A: {r['mae_model_a']:.4f}")
        lines.append(f"RMSE B: {r['rmse_model_b']:.4f} | MAE B: {r['mae_model_b']:.4f}")
        lines.append(f"Better model: {r['better_model']}")
        lines.append(f"Plot Full: {r['plot_full']}")
        lines.append(f"Plot Eval: {r['plot_eval']}")
        lines.append("")

    with open(OUTPUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))

if __name__ == '__main__':
    main()
