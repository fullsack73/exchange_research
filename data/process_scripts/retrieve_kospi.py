import yfinance as yf
import pandas as pd
import os

# Set output path
output_dir = '/Applications/dollar_price/data/KOSPI/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'KOSPI_processed.csv')

# Define ticker symbol for KOSPI
kospi_ticker = '^KS11'

# Fetch historical KOSPI data
def fetch_kospi():
    print(f"Fetching KOSPI data for ticker {kospi_ticker}...")
    try:
        data = yf.download(kospi_ticker, start='1995-01-01', end='2026-04-26', progress=False)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.reset_index(inplace=True)

        # Save to CSV
        print(f"Saving KOSPI data to {output_path}")
        data.to_csv(output_path, index=False)
        print("Data preview:")
        print(data.head())
    except Exception as e:
        print(f"Error fetching KOSPI data: {e}")

if __name__ == "__main__":
    fetch_kospi()