import pandas as pd
import os

def main():
    # Set base directory relative to the script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    
    print("Loading datasets...")
    macro_df = pd.read_csv(os.path.join(base_dir, 'data/macro_dataset_processed.csv'))
    fin_df = pd.read_csv(os.path.join(base_dir, 'data/financial_indicators_monthly_1995_2026.csv'))

    # Convert to datetime
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    fin_df['Date'] = pd.to_datetime(fin_df['Date'])

    # Standardize date to month-end
    macro_df['Date'] = macro_df['Date'] + pd.offsets.MonthEnd(0)
    fin_df['Date'] = fin_df['Date'] + pd.offsets.MonthEnd(0)

    # Merge on Date
    print("Merging datasets...")
    integrated_df = pd.merge(macro_df, fin_df, on='Date', how='inner')

    # Sort by date
    integrated_df = integrated_df.sort_values('Date').reset_index(drop=True)

    output_path = os.path.join(base_dir, 'data/integrated_macro_targets.csv')
    integrated_df.to_csv(output_path, index=False)
    print(f"Data merged successfully. Output saved to {output_path}")
    print(f"Integrated dataset shape: {integrated_df.shape}")

if __name__ == '__main__':
    main()
