import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

BASE_DIR = Path("/Applications/dollar_price")
OUT_DIR = BASE_DIR / "analysis" / "LSTM" / "lstm_mmf"
DATA_PATH = OUT_DIR / "daily_dataset.csv"
PERIOD_PATH = BASE_DIR / "analysis" / "anomaly" / "period_definition.json"


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
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length + pred_step - 1, 0])
    return np.array(xs), np.array(ys).reshape(-1, 1)


def inverse_target(scaler: StandardScaler, y_scaled: np.ndarray, n_features: int) -> np.ndarray:
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, 0] = y_scaled[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]


def train_model(model, train_loader, criterion, optimizer, num_epochs: int):
    model.train()
    for _ in range(num_epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()


def tune_hyperparams(x_train, y_train, input_dim: int):
    val_n = max(30, int(len(x_train) * 0.2))
    val_n = min(val_n, len(x_train) - 30)

    x_fit, y_fit = x_train[:-val_n], y_train[:-val_n]
    x_val, y_val = x_train[-val_n:], y_train[-val_n:]

    grid = {
        "hidden_dim": [16, 32, 64],
        "num_layers": [1, 2],
        "lr": [0.001, 0.005],
        "num_epochs": [50],
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
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_model(model, loader, criterion, opt, num_epochs)

        model.eval()
        with torch.no_grad():
            pred_val = model(torch.FloatTensor(x_val)).numpy()

        rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        if rmse < best_rmse:
            best_rmse = rmse
            best = {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "lr": lr,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "val_rmse_scaled": rmse,
            }

    return best


def build_anomaly_mask(df: pd.DataFrame, period: dict) -> pd.Series:
    blocks = period.get("anomaly_blocks_for_analysis", period.get("all_contiguous_blocks", []))
    mask = pd.Series(False, index=df.index)
    for b in blocks:
        s = pd.to_datetime(b["start"])
        e = pd.to_datetime(b["end"])
        mask = mask | ((df["observation_date"] >= s) & (df["observation_date"] <= e))
    return mask


def evaluate_block(
    full_df: pd.DataFrame,
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    scaler_a: StandardScaler,
    scaler_b: StandardScaler,
    model_a: ExRateLSTM,
    model_b: ExRateLSTM,
    seq_length: int,
    pred_step: int,
    flat_mmf_value: float,
):
    feat_a = ["USD_KRW", "RATE_SPREAD_KOR_USA"]
    feat_b = ["USD_KRW", "RATE_SPREAD_KOR_USA", "MMF_total"]

    data_a_all = scaler_a.transform(full_df[feat_a])
    data_b_all = scaler_b.transform(full_df[feat_b])

    x_a_all, y_a_all = create_sequences(data_a_all, seq_length, pred_step)
    x_b_all, y_b_all = create_sequences(data_b_all, seq_length, pred_step)
    if len(x_a_all) == 0 or len(x_b_all) == 0:
        return None, None

    target_dates = full_df["observation_date"].iloc[seq_length + pred_step - 1:].reset_index(drop=True)
    mask = (target_dates >= block_start) & (target_dates <= block_end)
    if int(mask.sum()) == 0:
        return None, None

    x_a = x_a_all[mask.values]
    y_a = y_a_all[mask.values]
    x_b = x_b_all[mask.values]

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        pred_a = model_a(torch.FloatTensor(x_a)).numpy()
        pred_b = model_b(torch.FloatTensor(x_b)).numpy()

    actual = inverse_target(scaler_a, y_a, len(feat_a))
    pred_a_true = inverse_target(scaler_a, pred_a, len(feat_a))
    pred_b_true = inverse_target(scaler_b, pred_b, len(feat_b))

    cf_df = full_df.copy()
    cf_mask = (cf_df["observation_date"] >= block_start) & (cf_df["observation_date"] <= block_end)
    cf_df.loc[cf_mask, "MMF_total"] = flat_mmf_value
    x_cf_all, _ = create_sequences(scaler_b.transform(cf_df[feat_b]), seq_length, pred_step)
    x_cf = x_cf_all[mask.values]
    with torch.no_grad():
        pred_cf = model_b(torch.FloatTensor(x_cf)).numpy()
    pred_cf_true = inverse_target(scaler_b, pred_cf, len(feat_b))

    pred_dates = target_dates[mask.values].reset_index(drop=True)
    preds = pd.DataFrame(
        {
            "date": pred_dates,
            "actual_fx": actual,
            "pred_model_a": pred_a_true,
            "pred_model_b": pred_b_true,
            "pred_counterfactual_flat_mmf": pred_cf_true,
        }
    )

    metrics = {
        "samples": int(len(actual)),
        "rmse_model_a": float(np.sqrt(mean_squared_error(actual, pred_a_true))),
        "rmse_model_b": float(np.sqrt(mean_squared_error(actual, pred_b_true))),
        "mae_model_a": float(mean_absolute_error(actual, pred_a_true)),
        "mae_model_b": float(mean_absolute_error(actual, pred_b_true)),
    }
    return metrics, preds


def weighted_metric(block_metrics: list[dict], key: str) -> float:
    weights = np.array([m["samples"] for m in block_metrics], dtype=float)
    values = np.array([m[key] for m in block_metrics], dtype=float)
    return float(np.sum(weights * values) / np.sum(weights))


def main() -> None:
    seq_length = 30
    pred_step = 5

    df = pd.read_csv(DATA_PATH)
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df = df.sort_values("observation_date").dropna().reset_index(drop=True)

    with open(PERIOD_PATH, "r", encoding="utf-8") as f:
        period = json.load(f)

    mask_anom = build_anomaly_mask(df, period)
    train_df = df[~mask_anom].copy().reset_index(drop=True)

    feat_a = ["USD_KRW", "RATE_SPREAD_KOR_USA"]
    feat_b = ["USD_KRW", "RATE_SPREAD_KOR_USA", "MMF_total"]

    scaler_a = StandardScaler().fit(train_df[feat_a])
    scaler_b = StandardScaler().fit(train_df[feat_b])

    x_a_train, y_a_train = create_sequences(scaler_a.transform(train_df[feat_a]), seq_length, pred_step)
    x_b_train, y_b_train = create_sequences(scaler_b.transform(train_df[feat_b]), seq_length, pred_step)

    best_a = tune_hyperparams(x_a_train, y_a_train, len(feat_a))
    best_b = tune_hyperparams(x_b_train, y_b_train, len(feat_b))

    loader_a = DataLoader(TensorDataset(torch.FloatTensor(x_a_train), torch.FloatTensor(y_a_train)), batch_size=best_a["batch_size"], shuffle=True)
    loader_b = DataLoader(TensorDataset(torch.FloatTensor(x_b_train), torch.FloatTensor(y_b_train)), batch_size=best_b["batch_size"], shuffle=True)

    model_a = ExRateLSTM(len(feat_a), best_a["hidden_dim"], best_a["num_layers"])
    model_b = ExRateLSTM(len(feat_b), best_b["hidden_dim"], best_b["num_layers"])

    criterion = nn.MSELoss()
    opt_a = torch.optim.Adam(model_a.parameters(), lr=best_a["lr"])
    opt_b = torch.optim.Adam(model_b.parameters(), lr=best_b["lr"])

    train_model(model_a, loader_a, criterion, opt_a, best_a["num_epochs"])
    train_model(model_b, loader_b, criterion, opt_b, best_b["num_epochs"])

    blocks = period.get("anomaly_blocks_for_analysis", period.get("all_contiguous_blocks", []))
    block_metrics = []
    all_preds = []
    flat_mmf_value = float(train_df["MMF_total"].iloc[-1])

    for idx, b in enumerate(blocks, start=1):
        s = pd.to_datetime(b["start"])
        e = pd.to_datetime(b["end"])

        metrics, preds = evaluate_block(
            df,
            s,
            e,
            scaler_a,
            scaler_b,
            model_a,
            model_b,
            seq_length,
            pred_step,
            flat_mmf_value,
        )
        if metrics is None:
            continue

        metrics["block_start"] = s.strftime("%Y-%m-%d")
        metrics["block_end"] = e.strftime("%Y-%m-%d")
        metrics["block_index"] = idx
        block_metrics.append(metrics)

        preds["block_index"] = idx
        preds["block_start"] = s.strftime("%Y-%m-%d")
        preds["block_end"] = e.strftime("%Y-%m-%d")
        all_preds.append(preds)

    if not block_metrics:
        raise ValueError("No anomaly blocks had enough rows for sequence evaluation")

    summary = {
        "train_rows": int(len(train_df)),
        "evaluated_blocks": int(len(block_metrics)),
        "weighted_rmse_model_a": weighted_metric(block_metrics, "rmse_model_a"),
        "weighted_rmse_model_b": weighted_metric(block_metrics, "rmse_model_b"),
        "weighted_mae_model_a": weighted_metric(block_metrics, "mae_model_a"),
        "weighted_mae_model_b": weighted_metric(block_metrics, "mae_model_b"),
        "best_params_a": best_a,
        "best_params_b": best_b,
    }
    summary["better_model"] = "B" if summary["weighted_rmse_model_b"] < summary["weighted_rmse_model_a"] else "A"

    eval_dir = OUT_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(block_metrics).to_csv(eval_dir / "block_metrics.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(eval_dir / "predictions.csv", index=False)

    with open(OUT_DIR / "results_extended.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved block metrics: {eval_dir / 'block_metrics.csv'}")
    print(f"Saved predictions: {eval_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
