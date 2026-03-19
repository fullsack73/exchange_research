"""
LSTM Comparison with CPI Integration - Using validation_daily parameters
Exact same setup as lstm_validation_daily but with CPI features added
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)


class ExRateLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data: np.ndarray, seq_length: int, pred_step: int):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_step + 1):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length + pred_step - 1, 0])
    return np.array(xs), np.array(ys).reshape(-1, 1)


def train_model(model, train_loader, criterion, optimizer, num_epochs: int = 50):
    model.train()
    for _ in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()


def inverse_target(scaler: StandardScaler, y_scaled: np.ndarray, n_features: int) -> np.ndarray:
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, 0] = y_scaled[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]


def train_eval_fixed_params():
    """Train and evaluate using FIXED parameters (no tuning) to match validation_daily"""
    out_dir = "analysis/lstm_cpi_integration"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(f"{out_dir}/daily_dataset_cpi_integrated.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    print(f"Loaded data with shape: {df.shape}")

    results = {}
    seq_length, pred_step = 30, 5

    # Fixed parameters (matching validation_daily)
    fixed_params = {
        "full": {"hidden_dim": 32, "num_layers": 1, "lr": 0.001, "num_epochs": 30, "batch_size": 32},
        "anomaly": {"hidden_dim": 32, "num_layers": 1, "lr": 0.001, "num_epochs": 50, "batch_size": 32}
    }

    # ===== FULL PERIOD (all data) =====
    print("\n=== Full Period (all data) ===")
    
    test_ratio = 0.2
    split_idx = int(len(df) * (1 - test_ratio))
    df_train = df[:split_idx].reset_index(drop=True)
    df_test = df[split_idx:].reset_index(drop=True)

    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

    for model_idx, (features, model_name) in enumerate([
        (["USD_KRW", "RATE_SPREAD_KOR_USA"], "Model A (Spread only)"),
        (["USD_KRW", "RATE_SPREAD_KOR_USA", "MMF_total"], "Model B (Spread + MMF)"),
        (["USD_KRW", "RATE_SPREAD_KOR_USA", "MMF_total", "Energy_YoY_lag2", "Food_YoY", "Shelter_YoY_lag2", "Durables_YoY_lag3", "Headline_MoM_lag1"], "Model C (Spread + MMF + CPI)")
    ]):
        print(f"\n{model_name}")
        
        # Train data
        data_train = df_train[features].values
        scaler = StandardScaler()
        data_train_scaled = scaler.fit_transform(data_train)
        x_train, y_train = create_sequences(data_train_scaled, seq_length, pred_step)

        # Test data
        data_test = df_test[features].values
        data_test_scaled = scaler.transform(data_test)
        x_test, y_test = create_sequences(data_test_scaled, seq_length, pred_step)

        print(f"Train sequences: {len(x_train)}, Test sequences: {len(x_test)}")

        params = fixed_params["full"]
        print(f"Fixed params: {params}")

        # Train model
        model = ExRateLSTM(x_train.shape[2], params["hidden_dim"], params["num_layers"])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

        train_set = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=False)
        train_model(model, train_loader, criterion, optimizer, params["num_epochs"])

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(x_test)).numpy()

        y_test_inv = inverse_target(scaler, y_test, len(features))
        y_pred_inv = inverse_target(scaler, y_pred, len(features))

        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae = mean_absolute_error(y_test_inv, y_pred_inv)

        results[f"full_{model_idx}"] = {
            "model_name": model_name,
            "rmse": float(rmse),
            "mae": float(mae),
            "test_dates": df_test.iloc[seq_length + pred_step - 1::pred_step]["observation_date"].astype(str).tolist()[:len(y_test_inv)],
            "y_test": y_test_inv.tolist(),
            "y_pred": y_pred_inv.tolist(),
        }

        print(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # ===== ANOMALY PERIOD (2024-11 ~ 2026-03) =====
    print("\n=== Anomaly Period (2024-11-01 ~ 2026-03-16) ===")
    
    df_anom = df[(df["observation_date"] >= "2024-11-01") & (df["observation_date"] <= "2026-03-16")].reset_index(drop=True)
    test_ratio_anom = 0.2
    split_idx_anom = int(len(df_anom) * (1 - test_ratio_anom))
    df_train_anom = df_anom[:split_idx_anom].reset_index(drop=True)
    df_test_anom = df_anom[split_idx_anom:].reset_index(drop=True)

    print(f"Train size: {len(df_train_anom)}, Test size: {len(df_test_anom)}")

    for model_idx, (features, model_name) in enumerate([
        (["USD_KRW", "RATE_SPREAD_KOR_USA"], "Model A (Spread only)"),
        (["USD_KRW", "RATE_SPREAD_KOR_USA", "MMF_total"], "Model B (Spread + MMF)"),
        (["USD_KRW", "RATE_SPREAD_KOR_USA", "MMF_total", "Energy_YoY_lag2", "Food_YoY", "Shelter_YoY_lag2", "Durables_YoY_lag3", "Headline_MoM_lag1"], "Model C (Spread + MMF + CPI)")
    ]):
        print(f"\n{model_name}")
        
        # Train data
        data_train_anom = df_train_anom[features].values
        scaler_anom = StandardScaler()
        data_train_anom_scaled = scaler_anom.fit_transform(data_train_anom)
        x_train_anom, y_train_anom = create_sequences(data_train_anom_scaled, seq_length, pred_step)

        # Test data
        data_test_anom = df_test_anom[features].values
        data_test_anom_scaled = scaler_anom.transform(data_test_anom)
        x_test_anom, y_test_anom = create_sequences(data_test_anom_scaled, seq_length, pred_step)

        print(f"Train sequences: {len(x_train_anom)}, Test sequences: {len(x_test_anom)}")

        params = fixed_params["anomaly"]
        print(f"Fixed params: {params}")

        # Train model
        model_anom = ExRateLSTM(x_train_anom.shape[2], params["hidden_dim"], params["num_layers"])
        criterion_anom = nn.MSELoss()
        optimizer_anom = torch.optim.Adam(model_anom.parameters(), lr=params["lr"])

        train_set_anom = TensorDataset(torch.FloatTensor(x_train_anom), torch.FloatTensor(y_train_anom))
        train_loader_anom = DataLoader(train_set_anom, batch_size=params["batch_size"], shuffle=False)
        train_model(model_anom, train_loader_anom, criterion_anom, optimizer_anom, params["num_epochs"])

        # Evaluate
        model_anom.eval()
        with torch.no_grad():
            y_pred_anom = model_anom(torch.FloatTensor(x_test_anom)).numpy()

        y_test_anom_inv = inverse_target(scaler_anom, y_test_anom, len(features))
        y_pred_anom_inv = inverse_target(scaler_anom, y_pred_anom, len(features))

        rmse_anom = np.sqrt(mean_squared_error(y_test_anom_inv, y_pred_anom_inv))
        mae_anom = mean_absolute_error(y_test_anom_inv, y_pred_anom_inv)

        results[f"anomaly_{model_idx}"] = {
            "model_name": model_name,
            "rmse": float(rmse_anom),
            "mae": float(mae_anom),
            "test_dates": df_test_anom.iloc[seq_length + pred_step - 1::pred_step]["observation_date"].astype(str).tolist()[:len(y_test_anom_inv)],
            "y_test": y_test_anom_inv.tolist(),
            "y_pred": y_pred_anom_inv.tolist(),
        }

        print(f"RMSE: {rmse_anom:.6f}, MAE: {mae_anom:.6f}")

    # Save results
    with open(f"{out_dir}/results_fixed_params.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Training complete ===")
    print(f"Results saved to {out_dir}/results_fixed_params.json")


if __name__ == "__main__":
    train_eval_fixed_params()
