import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)


class ExRateLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2):
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


def train_model(model, train_loader, criterion, optimizer, num_epochs: int = 120):
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


def run_one_period(df_period: pd.DataFrame, period_name: str, out_dir: str):
    # Baseline vs Proposed feature sets
    features_a = ["USD_KRW", "RATE_SPREAD_KOR_USA"]
    features_b = ["USD_KRW", "RATE_SPREAD_KOR_USA", "MMF_total"]

    seq_length = 30
    pred_step = 5

    # Inside each period, use chronological split (80% train / 20% test)
    split_idx = int(len(df_period) * 0.8)
    split_idx = max(split_idx, seq_length + pred_step + 1)

    train_df = df_period.iloc[:split_idx].copy()
    test_df = df_period.iloc[split_idx:].copy()

    scaler_a = StandardScaler()
    scaler_b = StandardScaler()

    scaler_a.fit(train_df[features_a])
    scaler_b.fit(train_df[features_b])

    data_a = scaler_a.transform(df_period[features_a])
    data_b = scaler_b.transform(df_period[features_b])

    x_a, y_a = create_sequences(data_a, seq_length, pred_step)
    x_b, y_b = create_sequences(data_b, seq_length, pred_step)

    train_seq_len = split_idx - seq_length - pred_step + 1
    if train_seq_len < 10:
        raise ValueError(f"{period_name}: not enough train sequences")

    x_a_train, y_a_train = x_a[:train_seq_len], y_a[:train_seq_len]
    x_a_test, y_a_test = x_a[train_seq_len:], y_a[train_seq_len:]

    x_b_train, y_b_train = x_b[:train_seq_len], y_b[:train_seq_len]
    x_b_test, y_b_test = x_b[train_seq_len:], y_b[train_seq_len:]

    loader_a = DataLoader(TensorDataset(torch.FloatTensor(x_a_train), torch.FloatTensor(y_a_train)), batch_size=32, shuffle=True)
    loader_b = DataLoader(TensorDataset(torch.FloatTensor(x_b_train), torch.FloatTensor(y_b_train)), batch_size=32, shuffle=True)

    model_a = ExRateLSTM(input_dim=len(features_a))
    model_b = ExRateLSTM(input_dim=len(features_b))

    criterion = nn.MSELoss()
    opt_a = torch.optim.Adam(model_a.parameters(), lr=0.001)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=0.001)

    train_model(model_a, loader_a, criterion, opt_a)
    train_model(model_b, loader_b, criterion, opt_b)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        pred_a = model_a(torch.FloatTensor(x_a_test)).numpy()
        pred_b = model_b(torch.FloatTensor(x_b_test)).numpy()

    actual = inverse_target(scaler_a, y_a_test, len(features_a))
    pred_a_true = inverse_target(scaler_a, pred_a, len(features_a))
    pred_b_true = inverse_target(scaler_b, pred_b, len(features_b))

    rmse_a = float(np.sqrt(mean_squared_error(actual, pred_a_true)))
    rmse_b = float(np.sqrt(mean_squared_error(actual, pred_b_true)))
    mae_a = float(mean_absolute_error(actual, pred_a_true))
    mae_b = float(mean_absolute_error(actual, pred_b_true))

    # Counterfactual only for proposed model
    df_cf = df_period.copy()
    last_train_mmf = train_df["MMF_total"].iloc[-1]
    df_cf.loc[test_df.index, "MMF_total"] = last_train_mmf
    data_cf = scaler_b.transform(df_cf[features_b])
    x_cf, _ = create_sequences(data_cf, seq_length, pred_step)
    x_cf_test = x_cf[train_seq_len:]

    with torch.no_grad():
        pred_cf = model_b(torch.FloatTensor(x_cf_test)).numpy()
    pred_cf_true = inverse_target(scaler_b, pred_cf, len(features_b))

    dates = df_period["observation_date"].values
    dates_test = dates[-len(actual):]

    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(dates_test), actual, label="Actual USD/KRW", color="black", linewidth=2)
    plt.plot(pd.to_datetime(dates_test), pred_a_true, label="Model A (Spread only)", color="orange", linestyle="--")
    plt.plot(pd.to_datetime(dates_test), pred_b_true, label="Model B (Spread + MMF)", color="red")
    plt.plot(pd.to_datetime(dates_test), pred_cf_true, label="Counterfactual (Flat MMF)", color="blue", linestyle=":")
    plt.title(f"Daily LSTM Evaluation - {period_name}")
    plt.ylabel("USD / KRW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"{out_dir}/{period_name}_lstm_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return {
        "period": period_name,
        "rows": int(len(df_period)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "rmse_model_a": rmse_a,
        "rmse_model_b": rmse_b,
        "mae_model_a": mae_a,
        "mae_model_b": mae_b,
        "better_model": "B" if rmse_b < rmse_a else "A",
        "plot": plot_path,
    }


def run_experiment():
    out_dir = "analysis/lstm_validation_daily"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(f"{out_dir}/daily_dataset.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    periods = {
        "full_2010_2025": ("2010-01-01", "2025-12-31"),
        "anomaly_2024_11_to_2025_12": ("2024-11-01", "2025-12-31"),
    }

    results = []
    for name, (start, end) in periods.items():
        period_df = df[(df["observation_date"] >= pd.to_datetime(start)) & (df["observation_date"] <= pd.to_datetime(end))].copy()
        period_df = period_df.reset_index(drop=True)
        if len(period_df) < 120:
            raise ValueError(f"{name}: not enough rows ({len(period_df)}) for daily LSTM with seq_length=30")
        results.append(run_one_period(period_df, name, out_dir))

    with open(f"{out_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = [
        "Daily LSTM Comparison by Period",
        "- Compare Model A (Spread only) vs Model B (Spread + MMF)",
        "",
    ]
    for r in results:
        lines.append(f"[{r['period']}]")
        lines.append(f"Rows: {r['rows']} (train={r['train_rows']}, test={r['test_rows']})")
        lines.append(f"RMSE A: {r['rmse_model_a']:.2f}")
        lines.append(f"RMSE B: {r['rmse_model_b']:.2f}")
        lines.append(f"MAE A: {r['mae_model_a']:.2f}")
        lines.append(f"MAE B: {r['mae_model_b']:.2f}")
        lines.append(f"Better: Model {r['better_model']}")
        lines.append(f"Plot: {r['plot']}")
        lines.append("")

    with open(f"{out_dir}/results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    run_experiment()
