import copy
import json
import os
import sys
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
os.makedirs(OUTPUT_DIR / "hpo", exist_ok=True)

HORIZON = 5


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
        rows.append(blk)
    if not rows:
        raise ValueError("No anomaly blocks found.")
    out = pd.concat(rows, axis=0, ignore_index=True)
    return out.sort_values(["block_index", "observation_date"]).reset_index(drop=True)


def prepare_log_data_for_period(df_period: pd.DataFrame, seq_length: int = 10, horizon: int = 5, test_ratio: float = 0.2):
    target_col = "USD_KRW"
    feature_cols = ["MMF_total", "RATE_SPREAD_KOR_USA"]

    df_period = df_period.copy()
    df_period["Log_Return"] = np.log(df_period[target_col] / df_period[target_col].shift(1))
    df_period = df_period.dropna(subset=["Log_Return"]).reset_index(drop=True)

    n_total = len(df_period)
    test_size = max(int(n_total * test_ratio), 40)
    train_size = n_total - test_size

    train_ts = df_period["Log_Return"].iloc[:train_size]
    arima_model = ARIMA(train_ts, order=(1, 0, 1))
    arima_result = arima_model.fit()

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
    X_test_seq, y_test_seq = create_sequences(test_X, test_y, seq_length, horizon)
    X_full_seq, y_full_seq = create_sequences(full_X, full_y, seq_length, horizon)

    return {
        "X_train": X_train_seq, "y_train": y_train_seq,
        "X_test": X_test_seq, "y_test": y_test_seq,
        "X_full": X_full_seq, "y_full": y_full_seq,
        "df_ready": df_period, "seq_length": seq_length, "horizon": horizon,
        "test_df": test_df.reset_index(drop=True),
        "scaler_y": scaler_y, "train_rows": train_size,
    }


class LSTM_Multi_Step(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2, horizon=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


class CNN_LSTM_Multi_Step(nn.Module):
    def __init__(self, input_dim, cnn_filters=16, kernel_size=3, hidden_dim=32, num_layers=1, dropout=0.2, horizon=5):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_filters, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        x = x.transpose(1, 2)
        c_out = self.relu(self.conv1d(x)).transpose(1, 2)
        _, (hn, _) = self.lstm(c_out)
        out = self.dropout(hn[-1])
        return self.fc(out)


class Hybrid_Model_Trainer:
    def __init__(self, model, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X_train, y_train, epochs=20):
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(bx), by)
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X_test).to(self.device)).cpu().numpy()


def generate_plot(period_name, dates, actual, arima_log, pred_a_scaled, pred_b_scaled, scaler_y, mode="eval"):
    # Reconstruct 1-step logic for a continuous line, which matches the aesthetic of the original plots
    # We take the first element (t+1) of the horizon prediction
    pred_res_a = scaler_y.inverse_transform(pred_a_scaled[:, 0].reshape(-1, 1)).flatten()
    pred_res_b = scaler_y.inverse_transform(pred_b_scaled[:, 0].reshape(-1, 1)).flatten()
    
    # Absolute prices
    # Y_{t+1} = Y_t * exp(arima_log_{t+1} + resid_{t+1})
    # actual is shifted by 1 relative to pred index.
    
    # Ensure length match
    min_len = min(len(actual)-1, len(pred_res_a))
    actual_t = actual[:-1][:min_len] # Y_t
    actual_t1 = actual[1:][:min_len] # True Y_{t+1} for plotting
    dates_t1 = dates[1:][:min_len]
    
    arima_log_t1 = arima_log[1:][:min_len]
    
    pred_A_t1 = actual_t * np.exp(arima_log_t1 + pred_res_a[:min_len])
    pred_B_t1 = actual_t * np.exp(arima_log_t1 + pred_res_b[:min_len])
    naive_t1 = actual_t # Naive is just Y_t
    
    rmse_a = np.sqrt(mean_squared_error(actual_t1, pred_A_t1))
    rmse_b = np.sqrt(mean_squared_error(actual_t1, pred_B_t1))
    rmse_naive = np.sqrt(mean_squared_error(actual_t1, naive_t1))
    mae_a = mean_absolute_error(actual_t1, pred_A_t1)
    mae_b = mean_absolute_error(actual_t1, pred_B_t1)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates_t1, actual_t1, color='black', label="Actual FX", alpha=0.7)
    ax.plot(dates_t1, pred_A_t1, color='#1f77b4', label="Model A (LSTM) 1-Step")
    ax.plot(dates_t1, pred_B_t1, color='#d62728', label="Model B (CNN-LSTM) 1-Step")
    ax.plot(dates_t1, naive_t1, color='green', linestyle='--', label="Naive (y_t=y_{t-1})", alpha=0.5)
    
    ax.set_title(f"[{mode.upper()}] {period_name} - Log Return Multi-Step Model (Eval of Next Day)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = OUTPUT_DIR / mode / f"{period_name}_hybrid_plot_{mode}.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")
    
    return float(rmse_a), float(rmse_b), float(mae_a), float(mae_b), float(rmse_naive)


def run_period(period_name: str, period_df: pd.DataFrame):
    print(f"Generating for {period_name}...")
    seq_length, horizon = 10, HORIZON
    prepared = prepare_log_data_for_period(period_df, seq_length=seq_length, horizon=horizon, test_ratio=0.2)

    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_test, X_full = prepared["X_test"], prepared["X_full"]
    scaler_y = prepared["scaler_y"]
    df_ready = prepared["df_ready"]
    input_dim = X_train.shape[2]
    
    # Train
    net_A = LSTM_Multi_Step(input_dim, horizon=horizon)
    trainer_A = Hybrid_Model_Trainer(net_A, lr=0.005)
    trainer_A.fit(X_train, y_train, epochs=30)

    net_B = CNN_LSTM_Multi_Step(input_dim, horizon=horizon)
    trainer_B = Hybrid_Model_Trainer(net_B, lr=0.005)
    trainer_B.fit(X_train, y_train, epochs=30)

    # Predict Test
    pred_a_test_scaled = trainer_A.predict(X_test)
    pred_b_test_scaled = trainer_B.predict(X_test)
    
    # Predict Full
    pred_a_full_scaled = trainer_A.predict(X_full)
    pred_b_full_scaled = trainer_B.predict(X_full)
    
    # Actuals for Eval (Test portion)
    start_test_idx = prepared["train_rows"]
    actual_test = df_ready["USD_KRW"].values[start_test_idx - 1:] # test
    dates_test = pd.to_datetime(df_ready.get("observation_date", range(len(df_ready)))[start_test_idx - 1:])
    arima_test = df_ready["ARIMA_Log_pred"].values[start_test_idx - 1:]
    
    rmse_a, rmse_b, mae_a, mae_b, rmse_naive = generate_plot(period_name, dates_test.values, actual_test, arima_test, pred_a_test_scaled, pred_b_test_scaled, scaler_y, mode="eval")

    # Actuals for Full
    actual_full = df_ready["USD_KRW"].values[seq_length - 1:]
    dates_full = pd.to_datetime(df_ready.get("observation_date", range(len(df_ready)))[seq_length - 1:])
    arima_full = df_ready["ARIMA_Log_pred"].values[seq_length - 1:]
    generate_plot(period_name, dates_full.values, actual_full, arima_full, pred_a_full_scaled, pred_b_full_scaled, scaler_y, mode="full")

    # Build response obj
    return {
        "period": period_name,
        "rows": len(df_ready),
        "train_rows": prepared["train_rows"],
        "test_rows": len(actual_test),
        "rmse_base_arima": rmse_naive, # storing naive as base explicitly
        "rmse_model_a": rmse_a,
        "rmse_model_b": rmse_b,
        "mae_model_a": mae_a,
        "mae_model_b": mae_b,
        "better_model": "A" if rmse_a < rmse_b else "B",
        "plot_full": f"analysis/LSTM/hybrid_mmf/full/{period_name}_hybrid_plot_full.png",
        "plot_eval": f"analysis/LSTM/hybrid_mmf/eval/{period_name}_hybrid_plot_eval.png"
    }

def main():
    df = pd.read_csv(DATA_PATH)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    period_def = load_period_definition()
    
    # Define periods
    p1 = df[(df["observation_date"] >= "1995-01-01") & (df["observation_date"] <= "2026-03-20")].copy()
    p2 = build_anomaly_concatenated(p1, period_def)
    p3 = df[(df["observation_date"] >= "2024-11-01") & (df["observation_date"] <= "2026-03-20")].copy()
    p4 = df[(df["observation_date"] >= "2010-01-01") & (df["observation_date"] <= "2026-03-20")].copy()
    
    results = [
        run_period("full_1995_2026", p1),
        run_period("anomaly_concatenated_blocks", p2),
        run_period("anomaly_2024_11_to_2026_03", p3),
        run_period("full_2010_2026", p4),
    ]

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [
        "Upgraded Architecture: Log Return & Multi-step Loss Hybrid",
        "",
    ]
    for r in results:
        lines.append(f"[{r['period']}]")
        lines.append(f"Rows: {r['rows']} (train={r['train_rows']}, test={r['test_rows']})")
        lines.append(f"RMSE Naive Base: {r['rmse_base_arima']:.4f}")
        lines.append(f"RMSE A (LSTM): {r['rmse_model_a']:.4f} | MAE A: {r['mae_model_a']:.4f}")
        lines.append(f"RMSE B (CNN-LSTM): {r['rmse_model_b']:.4f} | MAE B: {r['mae_model_b']:.4f}")
        lines.append(f"Better model: {r['better_model']}")
        lines.append(f"Plot Full: {r['plot_full']}")
        lines.append(f"Plot Eval: {r['plot_eval']}")
        lines.append("")

    with open(OUTPUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print("\nGeneration Complete.")

if __name__ == "__main__":
    main()
