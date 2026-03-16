import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)

class ExRateLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ExRateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length, pred_step):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_step + 1):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length + pred_step - 1, 0])
    return np.array(xs), np.array(ys).reshape(-1, 1)

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

def run_experiment():
    out_dir = 'analysis/lstm_validation_monthly'
    df = pd.read_csv(f'{out_dir}/monthly_dataset.csv')
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df.sort_values('observation_date', inplace=True)
    dates = df['observation_date'].values
    
    features_A = ['USD_KRW', 'RATE_SPREAD_KOR_USA']
    features_B = ['USD_KRW', 'RATE_SPREAD_KOR_USA', 'M2_MMF_지분']
    
    split_date = pd.to_datetime('2022-01-01')
    train_mask = df['observation_date'] < split_date
    test_mask = df['observation_date'] >= split_date
    
    seq_length = 6 # 6 months lookback
    pred_step = 1  # 1 month ahead
    
    scaler_A = StandardScaler()
    data_A = scaler_A.fit_transform(df[features_A])
    X_A, y_A = create_sequences(data_A, seq_length, pred_step)
    
    train_size = len(df[train_mask]) - seq_length - pred_step + 1
    
    X_A_train, y_A_train = X_A[:train_size], y_A[:train_size]
    X_A_test, y_A_test = X_A[train_size:], y_A[train_size:]
    
    scaler_B = StandardScaler()
    data_B = scaler_B.fit_transform(df[features_B])
    X_B, y_B = create_sequences(data_B, seq_length, pred_step)
    
    X_B_train, y_B_train = X_B[:train_size], y_B[:train_size]
    X_B_test, y_B_test = X_B[train_size:], y_B[train_size:]
    
    train_loader_A = DataLoader(TensorDataset(torch.FloatTensor(X_A_train), torch.FloatTensor(y_A_train)), batch_size=16, shuffle=True)
    train_loader_B = DataLoader(TensorDataset(torch.FloatTensor(X_B_train), torch.FloatTensor(y_B_train)), batch_size=16, shuffle=True)
    
    model_A = ExRateLSTM(input_dim=len(features_A), hidden_dim=16, num_layers=2, output_dim=1)
    model_B = ExRateLSTM(input_dim=len(features_B), hidden_dim=16, num_layers=2, output_dim=1)
    
    criterion = nn.MSELoss()
    optimizer_A = torch.optim.Adam(model_A.parameters(), lr=0.005)
    optimizer_B = torch.optim.Adam(model_B.parameters(), lr=0.005)
    
    print("Training Monthly Model A...")
    train_model(model_A, train_loader_A, criterion, optimizer_A, num_epochs=100)
    print("Training Monthly Model B...")
    train_model(model_B, train_loader_B, criterion, optimizer_B, num_epochs=100)
    
    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        pred_A = model_A(torch.FloatTensor(X_A_test)).numpy()
        pred_B = model_B(torch.FloatTensor(X_B_test)).numpy()
        
    def inverse_transform(scaler, pred_scaled, num_features):
        dummy = np.zeros((len(pred_scaled), num_features))
        dummy[:, 0] = pred_scaled[:, 0]
        return scaler.inverse_transform(dummy)[:, 0]
        
    pred_A_true = inverse_transform(scaler_A, pred_A, len(features_A))
    pred_B_true = inverse_transform(scaler_B, pred_B, len(features_B))
    actual_y = inverse_transform(scaler_A, y_A_test, len(features_A))
    
    # Counterfactual Simulation: Fix M2_MMF_지분 in test period
    df_cf = df.copy()
    last_train_mmf = df[train_mask]['M2_MMF_지분'].iloc[-1]
    df_cf.loc[test_mask, 'M2_MMF_지분'] = last_train_mmf
    data_cf = scaler_B.transform(df_cf[features_B])
    X_cf, _ = create_sequences(data_cf, seq_length, pred_step)
    X_cf_test = X_cf[train_size:]
    
    with torch.no_grad():
        pred_cf = model_B(torch.FloatTensor(X_cf_test)).numpy()
    pred_cf_true = inverse_transform(scaler_B, pred_cf, len(features_B))
    
    dates_test = dates[-len(pred_A_true):]
    
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(dates_test), actual_y, label='Actual USD/KRW', color='black', linewidth=2)
    plt.plot(pd.to_datetime(dates_test), pred_A_true, label='Model A (Spread only)', color='orange', linestyle='--')
    plt.plot(pd.to_datetime(dates_test), pred_B_true, label='Model B (Spread + MMF)', color='red')
    plt.plot(pd.to_datetime(dates_test), pred_cf_true, label='Counterfactual (Flat MMF)', color='blue', linestyle=':')
    plt.title('Monthly LSTM Evaluation')
    plt.ylabel('USD / KRW')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/monthly_lstm_plot.png')
    
    rmse_A = np.sqrt(mean_squared_error(actual_y, pred_A_true))
    rmse_B = np.sqrt(mean_squared_error(actual_y, pred_B_true))
    
    print(f"Monthly RMSE - Model A: {rmse_A:.2f}, Model B: {rmse_B:.2f}")

    with open(f'{out_dir}/results.txt', 'w') as f:
        f.write(f"Monthly RMSE - Model A (Spread only): {rmse_A:.2f}\n")
        f.write(f"Monthly RMSE - Model B (Spread + MMF): {rmse_B:.2f}\n")

if __name__ == '__main__':
    run_experiment()
