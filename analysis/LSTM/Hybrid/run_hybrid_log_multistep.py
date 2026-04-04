import copy
import itertools
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
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

import sys
target_type = sys.argv[1] if len(sys.argv) > 1 else "mmf"

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

TOTAL_HPO_TRIALS = 10
TRIALS_PER_MODEL = 5
MAX_TUNE_TRAIN = 1200
MAX_TUNE_VAL = 300
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
        blk["block_start"] = start
        blk["block_end"] = end
        rows.append(blk)
    if not rows:
        raise ValueError("No anomaly blocks found.")
    out = pd.concat(rows, axis=0, ignore_index=True)
    return out.sort_values(["block_index", "observation_date"]).reset_index(drop=True)


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


# Very simple HPO grid to save time for proof of concept
def build_grids():
    gA = [{"hidden_dim":32, "lr":0.001}, {"hidden_dim":64, "lr":0.005}, {"hidden_dim":16, "lr":0.001}]
    gB = [{"hidden_dim":32, "cnn_filters":16, "kernel_size":3, "lr":0.001}, {"hidden_dim":64, "cnn_filters":32, "kernel_size":5, "lr":0.005}]
    return gA, gB


def run_period(period_name: str, period_df: pd.DataFrame):
    print(f"\n[{period_name}] Processing Period (Log Return Multi-Step)")
    seq_length, horizon = 10, HORIZON
    prepared = prepare_log_data_for_period(period_df, seq_length=seq_length, horizon=horizon, test_ratio=0.2)

    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]
    scaler_y = prepared["scaler_y"]
    test_df = prepared["test_df"]
    df_ready = prepared["df_ready"]
    input_dim = X_train.shape[2]

    # Quick HPO
    gridA, gridB = build_grids()
    best_a_cfg, best_val_a = gridA[0], float('inf')
    best_b_cfg, best_val_b = gridB[0], float('inf')

    print("Tuning Model A...")
    for cfg in gridA:
        net = LSTM_Multi_Step(input_dim, hidden_dim=cfg["hidden_dim"], horizon=horizon)
        trainer = Hybrid_Model_Trainer(net, lr=cfg["lr"])
        val_rmse = np.sqrt(trainer.fit(X_train, y_train, epochs=20))
        if val_rmse < best_val_a:
            best_val_a = val_rmse
            best_a_cfg = cfg

    print("Tuning Model B...")
    for cfg in gridB:
        net = CNN_LSTM_Multi_Step(input_dim, cnn_filters=cfg["cnn_filters"], kernel_size=cfg["kernel_size"], hidden_dim=cfg["hidden_dim"], horizon=horizon)
        trainer = Hybrid_Model_Trainer(net, lr=cfg["lr"])
        val_rmse = np.sqrt(trainer.fit(X_train, y_train, epochs=20))
        if val_rmse < best_val_b:
            best_val_b = val_rmse
            best_b_cfg = cfg

    print("Training Final Models...")
    net_A = LSTM_Multi_Step(input_dim, hidden_dim=best_a_cfg["hidden_dim"], horizon=horizon)
    trainer_A = Hybrid_Model_Trainer(net_A, lr=best_a_cfg["lr"])
    trainer_A.fit(X_train, y_train, epochs=50)

    net_B = CNN_LSTM_Multi_Step(input_dim, cnn_filters=best_b_cfg["cnn_filters"], kernel_size=best_b_cfg["kernel_size"], hidden_dim=best_b_cfg["hidden_dim"], horizon=horizon)
    trainer_B = Hybrid_Model_Trainer(net_B, lr=best_b_cfg["lr"])
    trainer_B.fit(X_train, y_train, epochs=50)

    # Predictions
    pred_a_scaled = trainer_A.predict(X_test) # shape: (N, horizon)
    pred_b_scaled = trainer_B.predict(X_test)
    
    # We will pick non-overlapping windows of horizon to reconstruct
    # e.g., i=0, i=5, i=10
    actual_prices = df_ready["USD_KRW"].values
    arima_log = df_ready["ARIMA_Log_pred"].values
    
    # Let's reconstruct the 5-day absolute price forecast for the first subset of test
    # Test indices start after train_size
    train_size = prepared["train_rows"]
    
    all_dates = pd.to_datetime(df_ready.get("observation_date", range(len(df_ready))))
    plot_path = OUTPUT_DIR / "eval" / f"{period_name}_5_day_forecast_samples.png"
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot true actual lines in grey
    d_test = all_dates.iloc[train_size:]
    y_test_abs = actual_prices[train_size:]
    ax.plot(d_test, y_test_abs, color='black', alpha=0.3, label="Actual USD/KRW")

    samples_to_plot = min(len(X_test) // horizon, 10) # 10 disjoint 5-day samples
    
    # For baseline naive, we will take y_T and flatline it 5 days.
    rmse_a_list, rmse_b_list, rmse_naive_list = [], [], []

    for i in range(samples_to_plot):
        idx = i * horizon * 2 # Space them out
        if idx + horizon > len(X_test): break
        
        # Absolute indices corresponding to this prediction window
        # The sequence ending index in df_ready is: train_size - seq_length + seq_length + idx
        start_t = train_size + idx 
        base_price = actual_prices[start_t - 1] # Y_{t-1}
        
        # true prices for H days
        true_h_days = actual_prices[start_t : start_t + horizon]
        
        # Reconstruct ARIMA + Residual back into log return
        # Shape of pred_a_scaled[idx]: (horizon,)
        pred_scaled_vecA = pred_a_scaled[idx].reshape(-1,1)
        pred_scaled_vecB = pred_b_scaled[idx].reshape(-1,1)
        
        pred_residA = scaler_y.inverse_transform(pred_scaled_vecA).flatten()
        pred_residB = scaler_y.inverse_transform(pred_scaled_vecB).flatten()
        
        # ARIMA logs for this window
        arima_logs_H = arima_log[start_t : start_t + horizon]
        
        # Final predicted log returns
        log_pred_A = arima_logs_H + pred_residA
        log_pred_B = arima_logs_H + pred_residB
        
        # Reverse log returns: Y_T * exp(cumsum(R_t))
        abs_pred_A = base_price * np.exp(np.cumsum(log_pred_A))
        abs_pred_B = base_price * np.exp(np.cumsum(log_pred_B))
        abs_naive = np.full(horizon, base_price)
        
        rmse_a_list.append(np.sqrt(mean_squared_error(true_h_days, abs_pred_A)))
        rmse_b_list.append(np.sqrt(mean_squared_error(true_h_days, abs_pred_B)))
        rmse_naive_list.append(np.sqrt(mean_squared_error(true_h_days, abs_naive)))

        d_h = all_dates.iloc[start_t : start_t + horizon]
        if i == 0:
            ax.plot(d_h, true_h_days, color='black', linewidth=2, label="True Prices")
            ax.plot(d_h, abs_pred_A, color='#1f77b4', marker='o', label="Model A 5-Day")
            ax.plot(d_h, abs_pred_B, color='#d62728', marker='x', label="Model B 5-Day")
            ax.plot(d_h, abs_naive, color='green', linestyle='--', label="Naive 5-Day ($Y_{t-1}$)")
        else:
            ax.plot(d_h, true_h_days, color='black', linewidth=2)
            ax.plot(d_h, abs_pred_A, color='#1f77b4', marker='o')
            ax.plot(d_h, abs_pred_B, color='#d62728', marker='x')
            ax.plot(d_h, abs_naive, color='green', linestyle='--')
            
    ax.set_title(f"{period_name}: True Out-Of-Sample 5-Day Multi-Step Trajectories")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print(f"\n--- {horizon}-Day RMSE Evaluation (Sampled Windows) ---")
    print(f"Model A (LSTM) Avg {horizon}-Day RMSE: {np.mean(rmse_a_list):.4f}")
    print(f"Model B (CNN-LSTM) Avg {horizon}-Day RMSE: {np.mean(rmse_b_list):.4f}")
    print(f"Naive Baseline Avg {horizon}-Day RMSE: {np.mean(rmse_naive_list):.4f}")

    return {
        "rmse_a": np.mean(rmse_a_list),
        "rmse_b": np.mean(rmse_b_list),
        "rmse_naive": np.mean(rmse_naive_list),
        "plot": str(plot_path.relative_to(BASE_DIR))
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

    print("Running Multi-Step Log Return Models...")
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
