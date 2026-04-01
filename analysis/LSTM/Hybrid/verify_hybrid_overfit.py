import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

sys.path.append("/Applications/dollar_price")

from analysis.LSTM.Hybrid.run_hybrid_periods import (
    load_period_definition, DATA_PATH, OUTPUT_DIR, BASE_DIR,
    build_anomaly_concatenated, prepare_hybrid_data_for_period,
    ARIMA_LSTM_Model, ARIMA_CNN_LSTM_Model
)

def evaluate_metrics(y_true, y_pred, name="Model"):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    print(f"{name} -> RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae

def plot_zoomed(date_series, actual, pred_a, pred_b, naive, title, out_path, num_days=100):
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Zoom in on the last `num_days` points
    d = date_series[-num_days:]
    y_act = actual[-num_days:]
    p_a = pred_a[-num_days:]
    p_b = pred_b[-num_days:]
    n_v = naive[-num_days:]

    ax.plot(d, y_act, color="black", linewidth=2.0, label="Actual FX")
    ax.plot(d, p_a, color="#1f77b4", linewidth=1.5, marker="o", markersize=3, alpha=0.8, label="Model A (LSTM)")
    ax.plot(d, p_b, color="#d62728", linewidth=1.5, marker="x", markersize=3, alpha=0.8, label="Model B (CNN-LSTM)")
    ax.plot(d, n_v, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Naive Baseline ($y_t = y_{t-1}$)")
    
    ax.set_title(title)
    ax.set_xlabel("Date/Step")
    ax.set_ylabel("USD/KRW")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def multi_step_forecast(model, X_test_seq, scaler_y, steps=30):
    # X_test_seq shape: (N, seq_length, num_features)
    # The true observations are in X_test_seq
    # We will simulate 30 days of autoregressive forecasting starting from the beginning of X_test_seq
    
    # Start with the first sequence of the test set
    curr_seq = X_test_seq[0:1].copy() # shape (1, seq_length, num_features)
    
    preds_scaled = []
    
    for i in range(steps):
        # Predict the next residual
        with torch.no_grad():
            tensor_in = torch.FloatTensor(curr_seq).to(model.device)
            pred_res_scaled = model.model(tensor_in).cpu().numpy()[0, 0]
        preds_scaled.append(pred_res_scaled)
        
        if i < steps - 1:
            # Shift sequence to the left
            next_seq = curr_seq[:, 1:, :].copy() 
            # We need standard features (MMF, RATE) for the new step. 
            # We can take them from the TRUE next step in X_test_seq.
            # X_test_seq[i+1] is the sequence ending at step `i` (to predict `i+1`).
            # The last element of X_test_seq[i+1] is the data at time `i`.
            # features are at index 0 and 1.
            true_covariates = X_test_seq[i+1, -1, 0:2] 
            
            new_step = np.zeros((1, 1, curr_seq.shape[2]))
            new_step[0, 0, 0:2] = true_covariates
            new_step[0, 0, 2] = pred_res_scaled  # The ONLY difference is using predicted residual instead of true!
            
            curr_seq = np.concatenate([next_seq, new_step], axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_resid = scaler_y.inverse_transform(preds_scaled).flatten()
    return preds_resid

def plot_residuals(res_a, res_b, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    plot_acf(res_a, ax=axes[0, 0], lags=40, title="ACF - Model A Residuals")
    plot_pacf(res_a, ax=axes[0, 1], lags=40, title="PACF - Model A Residuals", method="ywm")
    
    plot_acf(res_b, ax=axes[1, 0], lags=40, title="ACF - Model B Residuals")
    plot_pacf(res_b, ax=axes[1, 1], lags=40, title="PACF - Model B Residuals", method="ywm")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def plot_gap(period_name):
    try:
        df_a = pd.read_csv(OUTPUT_DIR / "hpo" / f"{period_name}_model_a_trials.csv")
        df_b = pd.read_csv(OUTPUT_DIR / "hpo" / f"{period_name}_model_b_trials.csv")
    except FileNotFoundError:
        print(f"HPO files missing for {period_name}.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Model A
    ax[0].scatter(df_a["train_rmse_scaled"], df_a["val_rmse_scaled"], alpha=0.7)
    gap_a = df_a["val_rmse_scaled"].mean() - df_a["train_rmse_scaled"].mean()
    ax[0].set_title(f"Model A Gap (Avg: {gap_a:.4f})")
    ax[0].set_xlabel("Train RMSE")
    ax[0].set_ylabel("Val RMSE")
    min_val_a = min(df_a["train_rmse_scaled"].min(), df_a["val_rmse_scaled"].min())
    max_val_a = max(df_a["train_rmse_scaled"].max(), df_a["val_rmse_scaled"].max())
    ax[0].plot([min_val_a, max_val_a], [min_val_a, max_val_a], 'r--', alpha=0.5)

    # Model B
    ax[1].scatter(df_b["train_rmse_scaled"], df_b["val_rmse_scaled"], alpha=0.7)
    gap_b = df_b["val_rmse_scaled"].mean() - df_b["train_rmse_scaled"].mean()
    ax[1].set_title(f"Model B Gap (Avg: {gap_b:.4f})")
    ax[1].set_xlabel("Train RMSE")
    ax[1].set_ylabel("Val RMSE")
    min_val_b = min(df_b["train_rmse_scaled"].min(), df_b["val_rmse_scaled"].min())
    max_val_b = max(df_b["train_rmse_scaled"].max(), df_b["val_rmse_scaled"].max())
    ax[1].plot([min_val_b, max_val_b], [min_val_b, max_val_b], 'r--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval" / f"{period_name}_hpo_gap.png", dpi=120)
    plt.close(fig)


def verify_period(period_name, period_df):
    print(f"\n======================================")
    print(f"VERIFYING [{period_name}]")
    print(f"======================================")
    
    # 1. Prepare data
    prepared = prepare_hybrid_data_for_period(period_df)
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]
    scaler_y = prepared["scaler_y"]
    test_df = prepared["test_df"]
    input_dim = X_train.shape[2]

    # Load best params
    hpo_file = OUTPUT_DIR / "hpo" / f"{period_name}_hpo_summary.json"
    if not hpo_file.exists():
        print(f"Cannot find {hpo_file}")
        return

    with open(hpo_file, "r") as f:
        hpo_data = json.load(f)
    
    best_a = hpo_data["best_params_a"]
    best_b = hpo_data["best_params_b"]

    print("Retraining Model A & B with best configs...")
    model_A = ARIMA_LSTM_Model(input_dim=input_dim, hidden_dim=best_a["hidden_dim"], 
                               num_layers=best_a["num_layers"], dropout=best_a["dropout"], 
                               lr=best_a["lr"], weight_decay=best_a["weight_decay"])
    model_A.fit(X_train, y_train, epochs=best_a["epochs"], batch_size=best_a["batch_size"], patience=best_a["patience"])

    model_B = ARIMA_CNN_LSTM_Model(input_dim=input_dim, cnn_filters=best_b["cnn_filters"], kernel_size=best_b["kernel_size"],
                                   hidden_dim=best_b["hidden_dim"], num_layers=best_b["num_layers"], 
                                   dropout=best_b["dropout"], lr=best_b["lr"], weight_decay=best_b["weight_decay"])
    model_B.fit(X_train, y_train, epochs=best_b["epochs"], batch_size=best_b["batch_size"], patience=best_b["patience"])

    # Predictions (1-Step Ahead)
    pred_a_resid = scaler_y.inverse_transform(model_A.predict(X_test)).flatten()
    pred_b_resid = scaler_y.inverse_transform(model_B.predict(X_test)).flatten()
    
    test_arima = test_df["ARIMA_pred"].values
    actual = test_df["USD_KRW"].values
    
    final_a = test_arima + pred_a_resid
    final_b = test_arima + pred_b_resid

    # >>> 1. Naive Baseline Validation
    print("\n--- [Test 1] Naive/Persistence Comparison ---")
    naive_pred = np.roll(actual, 1)
    # The first element doesn't have a valid previous element in `actual`, we use the last train element.
    # We can get it from train_df or just drop index 0 from metrics. We will drop index 0.
    naive_pred[0] = actual[0] # just to avoid extreme errors
    print("Evaluating 1-Step Ahead Metrics (Excluding first day for naive alignment):")
    evaluate_metrics(actual[1:], test_arima[1:], "Base ARIMA")
    evaluate_metrics(actual[1:], final_a[1:], "Model A")
    evaluate_metrics(actual[1:], final_b[1:], "Model B")
    evaluate_metrics(actual[1:], naive_pred[1:], "Naive Baseline (Shift by 1)")

    # >>> 2. Visual Lag Inspection (Zoomed Plot)
    print("\n--- [Test 2] Zoomed-In Plot for Lag Detection ---")
    dates = test_df.get("date", test_df["observation_date"]) if "observation_date" in test_df else np.arange(len(actual))
    out_zoom = OUTPUT_DIR / "eval" / f"{period_name}_zoom_eval.png"
    plot_zoomed(dates.values, actual, final_a, final_b, naive_pred, 
                title=f"{period_name}: Zoomed 1-Step Forecast (Last 100 Days)", out_path=out_zoom)
    print(f"Saved {out_zoom}")

    # >>> 3. Multi-step Autoregressive Forecasting (30 days)
    print("\n--- [Test 3] 30-Day Multi-Step Prediction ---")
    steps_to_forecast = min(30, len(X_test))
    
    # Needs a 30-day TRUE ARIMA forecast starting from train data end
    train_ts = prepared["df_ready"]["USD_KRW"].iloc[:prepared["train_rows"]]
    arima_fit = ARIMA(train_ts, order=(1,1,1)).fit()
    arima_30 = arima_fit.forecast(steps=steps_to_forecast).values
    actual_30 = actual[:steps_to_forecast]

    multi_a_resid = multi_step_forecast(model_A, X_test, scaler_y, steps=steps_to_forecast)
    multi_b_resid = multi_step_forecast(model_B, X_test, scaler_y, steps=steps_to_forecast)

    multi_a_final = arima_30 + multi_a_resid
    multi_b_final = arima_30 + multi_b_resid
    
    print(f"Multi-step (30 Days) Evaluation:")
    evaluate_metrics(actual_30, arima_30, "Base ARIMA (30-day forecast)")
    evaluate_metrics(actual_30, multi_a_final, "Model A (30-day AR)")
    evaluate_metrics(actual_30, multi_b_final, "Model B (30-day AR)")

    # Plot Multi-step
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_30, label="Actual FX", color="black", marker="o", markersize=3)
    ax.plot(arima_30, label="ARIMA Base", color="grey", linestyle="--")
    ax.plot(multi_a_final, label="Model A AR", color="#1f77b4")
    ax.plot(multi_b_final, label="Model B AR", color="#d62728")
    ax.set_title(f"{period_name}: 30-Day Multi-Step Forecast (Out of sample)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ms_path = OUTPUT_DIR / "eval" / f"{period_name}_multistep.png"
    fig.savefig(ms_path)
    plt.close(fig)
    print(f"Saved {ms_path}")

    # >>> 4. Residual Diagnostics
    print("\n--- [Test 4] Residual Diagnostics (White Noise) ---")
    res_a = actual - final_a
    res_b = actual - final_b
    
    # Ljung-Box Test (H0: The data are independently distributed / White Noise)
    # p-value < 0.05 implies residuals are NOT white noise (patterns exist)
    lb_a = acorr_ljungbox(res_a, lags=[10], return_df=True)
    lb_b = acorr_ljungbox(res_b, lags=[10], return_df=True)
    
    print(f"Ljung-Box p-value (lag 10) Model A: {lb_a['lb_pvalue'].values[0]:.4e}")
    print(f"Ljung-Box p-value (lag 10) Model B: {lb_b['lb_pvalue'].values[0]:.4e}")
    if lb_a['lb_pvalue'].values[0] < 0.05:
        print(" -> Model A residuals have significant autocorrelation (patterns remain).")
    
    plot_res_path = OUTPUT_DIR / "eval" / f"{period_name}_residuals.png"
    plot_residuals(res_a, res_b, plot_res_path)
    print(f"Saved ACF/PACF plots: {plot_res_path}")

    # >>> 5. Generalization Gap Plot
    print("\n--- [Test 5] Plotting Generalization Gap ---")
    plot_gap(period_name)


def main():
    df = pd.read_csv(DATA_PATH)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    period_def = load_period_definition()
    range_start = pd.to_datetime(period_def["data_range"]["start"])
    range_end = pd.to_datetime(period_def["data_range"]["end"])
    df_full = df[(df["observation_date"] >= range_start) & (df["observation_date"] <= range_end)].copy()

    df_anomaly_concat = build_anomaly_concatenated(df_full, period_def)

    # Verify both periods
    verify_period("full_1995_2026", df_full)
    verify_period("anomaly_concatenated_blocks", df_anomaly_concat)

if __name__ == "__main__":
    main()
