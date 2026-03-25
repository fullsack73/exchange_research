import os
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# Ensure directories exist
OUTPUT_DIR = 'analysis/LSTM/hybrid_mmf'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'full'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'eval'), exist_ok=True)

# --- 1. Data Preprocessing Utility ---
def prepare_hybrid_data_for_period(df_period, seq_length=10, test_ratio=0.2):
    target_col = 'USD_KRW'
    feature_cols = ['MMF_total', 'RATE_SPREAD_KOR_USA']
    
    n_total = len(df_period)
    test_size = int(n_total * test_ratio)
    train_size = n_total - test_size
    
    train_ts = df_period[target_col].iloc[:train_size]
    
    # ARIMA for baseline trend extraction
    arima_model = ARIMA(train_ts, order=(1, 1, 1))
    arima_result = arima_model.fit()
    
    # Get 1-step ahead predictions for the whole series to avoid data leakage 
    res_full = arima_result.apply(df_period[target_col])
    df_period = df_period.copy()
    df_period['ARIMA_pred'] = res_full.fittedvalues
    df_period['Residuals'] = df_period[target_col] - df_period['ARIMA_pred']
    
    X_cols = feature_cols + ['Residuals']
    
    # Drop first few rows because of ARIMA initialization instability
    drop_init = 5
    df_period = df_period.iloc[drop_init:].reset_index(drop=True)
    
    # Re-calculate train/test split after dropping initial rows
    n_adj = len(df_period)
    test_size_adj = test_size
    train_size_adj = n_adj - test_size_adj
        
    train_df = df_period.iloc[:train_size_adj].copy()
    test_df = df_period.iloc[train_size_adj:].copy()
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_X = scaler_X.fit_transform(train_df[X_cols].values)
    train_y = scaler_y.fit_transform(train_df[['Residuals']].values)
    
    # For sequences we need the previous data for testing
    # So we don't just transform the test_df blindly, we overlap
    # We take the last sequence from train
    test_overlap = pd.concat([train_df.iloc[-seq_length:], test_df])
    test_X = scaler_X.transform(test_overlap[X_cols].values)
    test_y = scaler_y.transform(test_overlap[['Residuals']].values)
    
    def create_sequences(X, y, seq_len):
        xs, ys = [], []
        for i in range(len(X) - seq_len):
            xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(xs), np.array(ys)
        
    X_train_seq, y_train_seq = create_sequences(train_X, train_y, seq_length)
    X_test_seq, y_test_seq = create_sequences(test_X, test_y, seq_length)
    
    test_arima_preds = test_df['ARIMA_pred'].values
    test_actuals = test_df[target_col].values
    test_dates = pd.to_datetime(test_df['observation_date']).values
    
    return (X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
            scaler_y, test_arima_preds, test_actuals, test_dates, len(train_df), len(test_df))

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

# --- 4. Main Pipeline ---
def run_period(period_name, period_df):
    print(f"\n[{period_name}] Processing Period")
    seq_length = 10
    
    (X_train, y_train, X_test, y_test, scaler_y, 
     test_arima_preds, test_actuals, test_dates,
     train_cnt, test_cnt) = prepare_hybrid_data_for_period(period_df, seq_length=seq_length, test_ratio=0.2)
     
    input_dim = X_train.shape[2]
    print(f"Train samples: {train_cnt}, Test samples: {test_cnt}")
    
    # Train Model A
    print(f"[{period_name}] Training Model A (ARIMA-LSTM)...")
    model_A = ARIMA_LSTM_Model(input_dim=input_dim, hidden_dim=32, dropout=0.2)
    model_A.fit(X_train, y_train, epochs=100)
    preds_y_A = model_A.predict(X_test)
    preds_residual_A = scaler_y.inverse_transform(preds_y_A).flatten()
    final_preds_A = test_arima_preds + preds_residual_A
    
    # Train Model B
    print(f"[{period_name}] Training Model B (ARIMA-CNN-LSTM)...")
    model_B = ARIMA_CNN_LSTM_Model(input_dim=input_dim, cnn_filters=16, kernel_size=3, hidden_dim=32, dropout=0.2)
    model_B.fit(X_train, y_train, epochs=100)
    preds_y_B = model_B.predict(X_test)
    preds_residual_B = scaler_y.inverse_transform(preds_y_B).flatten()
    final_preds_B = test_arima_preds + preds_residual_B
    
    rmse_base = np.sqrt(mean_squared_error(test_actuals, test_arima_preds))
    rmse_A = np.sqrt(mean_squared_error(test_actuals, final_preds_A))
    mae_A = mean_absolute_error(test_actuals, final_preds_A)
    
    rmse_B = np.sqrt(mean_squared_error(test_actuals, final_preds_B))
    mae_B = mean_absolute_error(test_actuals, final_preds_B)
    
    better = "A" if rmse_A < rmse_B else "B"
    print(f"Base ARIMA RMSE: {rmse_base:.4f}")
    print(f"Model A RMSE: {rmse_A:.4f} | MAE: {mae_A:.4f}")
    print(f"Model B RMSE: {rmse_B:.4f} | MAE: {mae_B:.4f}")
    print(f"Better model for {period_name} is Model {better}")
    
    # Plotting
    plot_full_path = os.path.join(OUTPUT_DIR, 'full', f'{period_name}_hybrid_plot_full.png')
    plot_eval_path = os.path.join(OUTPUT_DIR, 'eval', f'{period_name}_hybrid_plot_eval.png')
    
    plt.figure(figsize=(15, 6))
    plt.plot(test_dates, test_actuals, label='Actual USD/KRW', color='black', linewidth=1.5)
    plt.plot(test_dates, test_arima_preds, label='ARIMA Baseline', color='grey', linestyle='--', alpha=0.7)
    plt.plot(test_dates, final_preds_A, label='Model A (ARIMA-LSTM)', color='blue', alpha=0.8)
    plt.plot(test_dates, final_preds_B, label='Model B (ARIMA-CNN-LSTM)', color='red', alpha=0.9)
    plt.title(f'{period_name}: USD/KRW Hybrid Forecasting', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Exchange Rate', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_eval_path)
    plt.close()
    
    return {
        "period": period_name,
        "rows": train_cnt + test_cnt,
        "train_rows": train_cnt,
        "test_rows": test_cnt,
        "rmse_base_arima": rmse_base,
        "rmse_model_a": rmse_A,
        "rmse_model_b": rmse_B,
        "mae_model_a": mae_A,
        "mae_model_b": mae_B,
        "better_model": better,
        "plot_full": plot_full_path,
        "plot_eval": plot_eval_path
    }

def main():
    data_path = 'analysis/LSTM/lstm_mmf/daily_dataset.csv'
    df = pd.read_csv(data_path)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values('observation_date').reset_index(drop=True)
    
    # 1. Full Period (2010 to 2026 roughly similar to existing results.json)
    df_full = df[df['observation_date'] >= '2010-01-01'].copy()
    
    # 2. Anomaly Period (2024-11 to 2026-03)
    df_anomaly = df[df['observation_date'] >= '2024-11-01'].copy()
    
    results = []
    
    # Run Full Period
    res_full = run_period("full_2010_2026", df_full)
    results.append(res_full)
    
    # Run Anomaly Period
    res_anomaly = run_period("anomaly_2024_11_to_2026_03", df_anomaly)
    results.append(res_anomaly)
    
    # Save Results
    res_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
