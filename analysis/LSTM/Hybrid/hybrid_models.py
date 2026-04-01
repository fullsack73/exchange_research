import os
import copy
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

# --- 1. Data Preprocessing Utility ---
def prepare_hybrid_data(data_path, seq_length=10, test_size=200):
    df = pd.read_csv(data_path)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values('observation_date').reset_index(drop=True)
    
    target_col = 'USD_KRW'
    # Use features including the exogenous liquidity variables
    feature_cols = ['MMF_total', 'RATE_SPREAD_KOR_USA']
    
    train_ts = df[target_col].iloc[:-test_size]
    # Simple ARIMA (1, 1, 1) for baseline trend extraction
    arima_model = ARIMA(train_ts, order=(1, 1, 1))
    arima_result = arima_model.fit()
    
    # Get 1-step ahead predictions for the whole series to avoid data leakage 
    # (using the parameters fitted on train)
    res_full = arima_result.apply(df[target_col])
    df['ARIMA_pred'] = res_full.fittedvalues
    df['Residuals'] = df[target_col] - df['ARIMA_pred']
    
    # We will predict Residuals using the exogenous features and past Residuals
    X_cols = feature_cols + ['Residuals']
    
    # Drop first few rows because of ARIMA initialization
    df = df.iloc[5:].reset_index(drop=True)
        
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_X = scaler_X.fit_transform(train_df[X_cols].values)
    train_y = scaler_y.fit_transform(train_df[['Residuals']].values)
    
    test_X = scaler_X.transform(test_df[X_cols].values)
    test_y = scaler_y.transform(test_df[['Residuals']].values)
    
    def create_sequences(X, y, seq_len):
        xs, ys = [], []
        for i in range(len(X) - seq_len):
            xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(xs), np.array(ys)
        
    X_train_seq, y_train_seq = create_sequences(train_X, train_y, seq_length)
    X_test_seq, y_test_seq = create_sequences(test_X, test_y, seq_length)
    
    # Need to save the original ARIMA preds for calculating final test evaluation
    # Shape of y_test_seq is (len(test_X) - seq_length, 1)
    # So we need to align the ARIMA predictions
    test_arima_preds = test_df['ARIMA_pred'].values[seq_length:]
    test_actuals = test_df[target_col].values[seq_length:]
    test_dates = test_df['observation_date'].values[seq_length:]
    
    return (X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
            scaler_y, test_arima_preds, test_actuals, test_dates)

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTM_Residual_Predictor(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay) # L2 Regularization
        
    def fit(self, X_train, y_train, epochs=100, batch_size=32, patience=10):
        # We'll just use a small validation split from train for early stopping
        val_size = int(len(X_train) * 0.1)
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
                # Early stopping
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
        # Conv1D expects (batch, channels, length)
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        # After Conv1D: (batch, cnn_filters, length). Transpose back to (batch, length, cnn_filters) for LSTM
        self.lstm = nn.LSTM(cnn_filters, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        c_out = self.relu(self.conv1d(x))
        # c_out: (batch, filters, seq_len) -> (batch, seq_len, filters)
        c_out = c_out.transpose(1, 2)
        lstm_out, _ = self.lstm(c_out)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

class ARIMA_CNN_LSTM_Model(ARIMA_LSTM_Model):
    def __init__(self, input_dim, cnn_filters=16, kernel_size=3, hidden_dim=32, num_layers=1, dropout=0.2, lr=0.001, weight_decay=1e-5):
        super(ARIMA_CNN_LSTM_Model, self).__init__(input_dim, hidden_dim, num_layers, dropout, lr, weight_decay)
        # Override model with CNN-LSTM Architecture
        self.model = CNN_LSTM_Residual_Predictor(input_dim, cnn_filters, kernel_size, hidden_dim, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

# --- 4. Main Pipeline ---
def main():
    data_path = 'analysis/LSTM/lstm_mmf/daily_dataset.csv'
    seq_length = 10
    test_size = 300 # Roughly 1 year of trading days
    
    print("Preparing data and running Base ARIMA...")
    X_train, y_train, X_test, y_test, scaler_y, test_arima_preds, test_actuals, test_dates = prepare_hybrid_data(
        data_path, seq_length=seq_length, test_size=test_size
    )
    
    input_dim = X_train.shape[2]
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    # Train Model A
    print("Training Model A: ARIMA-LSTM...")
    model_A = ARIMA_LSTM_Model(input_dim=input_dim, hidden_dim=32, dropout=0.2)
    model_A.fit(X_train, y_train, epochs=100)
    preds_y_A = model_A.predict(X_test)
    preds_residual_A = scaler_y.inverse_transform(preds_y_A).flatten()
    final_preds_A = test_arima_preds + preds_residual_A
    
    # Train Model B
    print("Training Model B: ARIMA-CNN-LSTM...")
    model_B = ARIMA_CNN_LSTM_Model(input_dim=input_dim, cnn_filters=16, kernel_size=3, hidden_dim=32, dropout=0.2)
    model_B.fit(X_train, y_train, epochs=100)
    preds_y_B = model_B.predict(X_test)
    preds_residual_B = scaler_y.inverse_transform(preds_y_B).flatten()
    final_preds_B = test_arima_preds + preds_residual_B
    
    # Base ARIMA evaluation
    rmse_base = np.sqrt(mean_squared_error(test_actuals, test_arima_preds))
    mae_base = mean_absolute_error(test_actuals, test_arima_preds)
    
    rmse_A = np.sqrt(mean_squared_error(test_actuals, final_preds_A))
    mae_A = mean_absolute_error(test_actuals, final_preds_A)
    
    rmse_B = np.sqrt(mean_squared_error(test_actuals, final_preds_B))
    mae_B = mean_absolute_error(test_actuals, final_preds_B)
    
    print(f"=====================================")
    print(f"Base ARIMA RMSE: {rmse_base:.4f} | MAE: {mae_base:.4f}")
    print(f"Model A (ARIMA-LSTM) RMSE: {rmse_A:.4f} | MAE: {mae_A:.4f}")
    print(f"Model B (ARIMA-CNN-LSTM) RMSE: {rmse_B:.4f} | MAE: {mae_B:.4f}")
    print(f"=====================================")
    
    plt.figure(figsize=(15, 6))
    plt.plot(test_dates, test_actuals, label='Actual USD/KRW', color='black', linewidth=1.5)
    plt.plot(test_dates, test_arima_preds, label='ARIMA Baseline', color='grey', linestyle='--', alpha=0.7)
    plt.plot(test_dates, final_preds_A, label='ARIMA-LSTM (Model A)', color='blue', alpha=0.8)
    plt.plot(test_dates, final_preds_B, label='ARIMA-CNN-LSTM (Model B)', color='red', alpha=0.9)
    plt.title('USD/KRW Forecasting using Hybrid Deep Learning Models', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Exchange Rate', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/LSTM/hybrid_models_forecast.png')
    print("Plot saved to analysis/LSTM/hybrid_models_forecast.png")

if __name__ == '__main__':
    main()
