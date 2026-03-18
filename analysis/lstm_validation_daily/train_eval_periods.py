import os
import json
from itertools import product
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
torch.set_num_threads(1)


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


def split_train_val(x_train: np.ndarray, y_train: np.ndarray, val_ratio: float = 0.2):
    n = len(x_train)
    val_n = max(20, int(n * val_ratio))
    val_n = min(val_n, n - 20)
    train_n = n - val_n
    return x_train[:train_n], y_train[:train_n], x_train[train_n:], y_train[train_n:]


def tune_hyperparams(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    period_name: str,
    model_name: str,
):
    x_fit, y_fit, x_val, y_val = split_train_val(x_train, y_train)

    # Cap tuning sample size for speed, then retrain on full train set with the best config.
    max_tune_train = 1200
    max_tune_val = 300
    if len(x_fit) > max_tune_train:
        x_fit = x_fit[-max_tune_train:]
        y_fit = y_fit[-max_tune_train:]
    if len(x_val) > max_tune_val:
        x_val = x_val[-max_tune_val:]
        y_val = y_val[-max_tune_val:]

    if period_name.startswith("full_"):
        grid = {
            "hidden_dim": [16, 32],
            "num_layers": [1],
            "lr": [0.001],
            "num_epochs": [30],
            "batch_size": [32],
        }
    else:
        grid = {
            "hidden_dim": [16, 32],
            "num_layers": [1],
            "lr": [0.001],
            "num_epochs": [50, 90],
            "batch_size": [32],
        }

    best = None
    best_rmse = float("inf")

    for hidden_dim, num_layers, lr, num_epochs, batch_size in product(
        grid["hidden_dim"],
        grid["num_layers"],
        grid["lr"],
        grid["num_epochs"],
        grid["batch_size"],
    ):
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_fit), torch.FloatTensor(y_fit)),
            batch_size=batch_size,
            shuffle=True,
        )

        model = ExRateLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_model(model, loader, criterion, optimizer, num_epochs=num_epochs)

        model.eval()
        with torch.no_grad():
            pred_val = model(torch.FloatTensor(x_val)).numpy()

        rmse_val = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best = {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "lr": lr,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "val_rmse_scaled": rmse_val,
            }

    print(f"[{period_name}] Best {model_name} params: {best}")
    return best


def run_one_period(df_period: pd.DataFrame, period_name: str, out_dir: str, enable_tuning: bool = True):
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

    if enable_tuning:
        best_a = tune_hyperparams(x_a_train, y_a_train, len(features_a), period_name, "Model A")
        best_b = tune_hyperparams(x_b_train, y_b_train, len(features_b), period_name, "Model B")
    else:
        best_a = {
            "hidden_dim": 32,
            "num_layers": 1,
            "lr": 0.001,
            "num_epochs": 80,
            "batch_size": 32,
            "val_rmse_scaled": None,
        }
        best_b = {
            "hidden_dim": 32,
            "num_layers": 1,
            "lr": 0.001,
            "num_epochs": 80,
            "batch_size": 32,
            "val_rmse_scaled": None,
        }

    loader_a = DataLoader(
        TensorDataset(torch.FloatTensor(x_a_train), torch.FloatTensor(y_a_train)),
        batch_size=best_a["batch_size"],
        shuffle=True,
    )
    loader_b = DataLoader(
        TensorDataset(torch.FloatTensor(x_b_train), torch.FloatTensor(y_b_train)),
        batch_size=best_b["batch_size"],
        shuffle=True,
    )

    model_a = ExRateLSTM(input_dim=len(features_a), hidden_dim=best_a["hidden_dim"], num_layers=best_a["num_layers"])
    model_b = ExRateLSTM(input_dim=len(features_b), hidden_dim=best_b["hidden_dim"], num_layers=best_b["num_layers"])

    criterion = nn.MSELoss()
    opt_a = torch.optim.Adam(model_a.parameters(), lr=best_a["lr"])
    opt_b = torch.optim.Adam(model_b.parameters(), lr=best_b["lr"])

    train_model(model_a, loader_a, criterion, opt_a, num_epochs=best_a["num_epochs"])
    train_model(model_b, loader_b, criterion, opt_b, num_epochs=best_b["num_epochs"])

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        # Test-split predictions for evaluation metrics
        pred_a_test = model_a(torch.FloatTensor(x_a_test)).numpy()
        pred_b_test = model_b(torch.FloatTensor(x_b_test)).numpy()

        # Full-range predictions for plotting (all available sequences in the period)
        pred_a_full_scaled = model_a(torch.FloatTensor(x_a)).numpy()
        pred_b_full_scaled = model_b(torch.FloatTensor(x_b)).numpy()

    actual = inverse_target(scaler_a, y_a_test, len(features_a))
    pred_a_true = inverse_target(scaler_a, pred_a_test, len(features_a))
    pred_b_true = inverse_target(scaler_b, pred_b_test, len(features_b))

    pred_a_full_true = inverse_target(scaler_a, pred_a_full_scaled, len(features_a))
    pred_b_full_true = inverse_target(scaler_b, pred_b_full_scaled, len(features_b))

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

    with torch.no_grad():
        pred_cf_full_scaled = model_b(torch.FloatTensor(x_cf)).numpy()
    pred_cf_full_true = inverse_target(scaler_b, pred_cf_full_scaled, len(features_b))

    full_plot_dir = f"{out_dir}/full"
    eval_plot_dir = f"{out_dir}/eval"
    os.makedirs(full_plot_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)

    dates_full = pd.to_datetime(df_period["observation_date"].values)
    pred_start_idx = seq_length + pred_step - 1
    dates_pred_full = dates_full[pred_start_idx:]
    dates_pred_eval = dates_pred_full[train_seq_len:]
    actual_full = df_period["USD_KRW"].values
    actual_eval = actual
    pred_cf_eval_true = pred_cf_full_true[train_seq_len:]

    # Full-range figure: whole period actual + whole available-range predictions
    plt.figure(figsize=(12, 6))
    plt.plot(dates_full, actual_full, label="Actual USD/KRW", color="black", linewidth=2)
    plt.plot(dates_pred_full, pred_a_full_true, label="Model A (Spread only)", color="orange", linestyle="--")
    plt.plot(dates_pred_full, pred_b_full_true, label="Model B (Spread + MMF)", color="red")
    plt.plot(dates_pred_full, pred_cf_full_true, label="Counterfactual (Flat MMF)", color="blue", linestyle=":")
    plt.title(f"Daily LSTM Full-Range - {period_name}")
    plt.ylabel("USD / KRW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_full_path = f"{full_plot_dir}/{period_name}_lstm_plot_full.png"
    plt.savefig(plot_full_path)
    plt.close()

    # Eval figure: only test 20% range (original behavior)
    plt.figure(figsize=(12, 6))
    plt.plot(dates_pred_eval, actual_eval, label="Actual USD/KRW", color="black", linewidth=2)
    plt.plot(dates_pred_eval, pred_a_true, label="Model A (Spread only)", color="orange", linestyle="--")
    plt.plot(dates_pred_eval, pred_b_true, label="Model B (Spread + MMF)", color="red")
    plt.plot(dates_pred_eval, pred_cf_eval_true, label="Counterfactual (Flat MMF)", color="blue", linestyle=":")
    plt.title(f"Daily LSTM Evaluation (20% Test) - {period_name}")
    plt.ylabel("USD / KRW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_eval_path = f"{eval_plot_dir}/{period_name}_lstm_plot_eval.png"
    plt.savefig(plot_eval_path)
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
        "best_params_a": best_a,
        "best_params_b": best_b,
        "plot_full": plot_full_path,
        "plot_eval": plot_eval_path,
    }


def run_experiment():
    out_dir = "analysis/lstm_validation_daily"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(f"{out_dir}/daily_dataset.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    periods = {
        "full_2010_2026": ("2010-01-01", "2026-03-16"),
        "anomaly_2024_11_to_2026_03": ("2024-11-01", "2026-3-16"),
    }

    results = []
    for name, (start, end) in periods.items():
        period_df = df[(df["observation_date"] >= pd.to_datetime(start)) & (df["observation_date"] <= pd.to_datetime(end))].copy()
        period_df = period_df.reset_index(drop=True)
        if len(period_df) < 120:
            raise ValueError(f"{name}: not enough rows ({len(period_df)}) for daily LSTM with seq_length=30")
        use_tuning = name != "full_2010_2025"
        results.append(run_one_period(period_df, name, out_dir, enable_tuning=use_tuning))

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
        lines.append(f"Best Params A: {r['best_params_a']}")
        lines.append(f"Best Params B: {r['best_params_b']}")
        lines.append(f"Plot Full: {r['plot_full']}")
        lines.append(f"Plot Eval: {r['plot_eval']}")
        lines.append("")

    with open(f"{out_dir}/results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    run_experiment()
