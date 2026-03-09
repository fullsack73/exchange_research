import pandas as pd
import os
import io

base_dir = '/Applications/dollar_price'

def clean_numeric(series):
    # Convert series to string, handle potential weird representations, remove commas
    # Handle cases where value might be just '-' or something similar
    s = series.astype(str).str.replace(',', '').str.strip()
    s = pd.to_numeric(s, errors='coerce')
    return s

def process_cma():
    path = os.path.join(base_dir, 'm2/KOR/CMA/운용대상별 CMA잔고 추이.csv')
    print(f"Reading {path}...")
    try:
        # Read the file line by line first to see encoding and structure
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print("CMA first 5 lines:", lines[:5])
            
        df = pd.read_csv(path, skiprows=2)
        print("CMA columns:", df.columns.tolist())
        
        # Rename columns for clarity
        df.rename(columns={'일자': 'observation_date', '합계': 'CMA_total'}, inplace=True)
        
        # Convert dates
        df['observation_date'] = pd.to_datetime(df['observation_date'], format='%Y/%m/%d')
        
        # Clean the numeric column
        df['CMA_total'] = clean_numeric(df['CMA_total'])
        
        # Keep only necessary columns
        df = df[['observation_date', 'CMA_total']].dropna()
        return df
    except Exception as e:
        print(f"Error processing CMA: {e}")
        return pd.DataFrame()

def process_mmf():
    path = os.path.join(base_dir, 'm2/KOR/MMF/MMF_daily.csv')
    print(f"Reading {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print("MMF first 8 lines:", lines[:8])

        # Find the header row
        header_idx = 0
        for i, line in enumerate(lines):
            if '기준일자' in line:
                header_idx = i
                break
        
        df = pd.read_csv(path, skiprows=header_idx)
        
        # There's a sub header row in the file.
        # Find index where valid data starts (first row where date parses)
        for idx in range(min(5, len(df))):
            try:
                pd.to_datetime(df.iloc[idx, 0])
                data_start_idx = idx
                break
            except:
                continue
                
        df = df.iloc[data_start_idx:].reset_index(drop=True)
        
        print("MMF columns:", df.columns.tolist())
        
        # Rename columns
        # Since column names are tricky due to multi-row headers, we'll just use integer index
        # 0: date, 1: total MMF
        df.rename(columns={df.columns[0]: 'observation_date', df.columns[1]: 'MMF_total'}, inplace=True)

        df['observation_date'] = pd.to_datetime(df['observation_date'], format='%Y/%m/%d')
        df['MMF_total'] = clean_numeric(df['MMF_total'])
        
        df = df[['observation_date', 'MMF_total']].dropna()
        return df
    except Exception as e:
        print(f"Error processing MMF: {e}")
        return pd.DataFrame()

def process_exchange_rate():
    path = os.path.join(base_dir, 'exchange_rate/환율_일별.csv')
    print(f"Reading {path}...")
    try:
        df = pd.read_csv(path)
        print("ER columns count:", len(df.columns))
        print("ER head:\n", df.head(2))
        
        target_row = df.iloc[0]
        
        potential_metadata = ['통계표', '계정항목', '단위', '변환']
        metadata_cols = [c for c in df.columns if c in potential_metadata]
        date_cols = [c for c in df.columns if c not in metadata_cols]
        
        melted_df = df.melt(id_vars=metadata_cols, value_vars=date_cols, var_name='original_date', value_name='USD_KRW')
        
        melted_df['observation_date'] = pd.to_datetime(melted_df['original_date'], format='%Y/%m/%d')
        melted_df['USD_KRW'] = clean_numeric(melted_df['USD_KRW'])
        
        final_df = melted_df[['observation_date', 'USD_KRW']].dropna().sort_values('observation_date')
        return final_df
    except Exception as e:
        print(f"Error processing ER: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    cma_df = process_cma()
    print("CMA shape:", cma_df.shape)
    
    mmf_df = process_mmf()
    print("MMF shape:", mmf_df.shape)
    
    er_df = process_exchange_rate()
    print("ER shape:", er_df.shape)
    
    if not cma_df.empty and not mmf_df.empty and not er_df.empty:
        # Merge datasets
        merged = pd.merge(er_df, cma_df, on='observation_date', how='inner')
        merged = pd.merge(merged, mmf_df, on='observation_date', how='inner')
        
        # Calculate M2 Proxy
        merged['M2_proxy'] = merged['CMA_total'] + merged['MMF_total']
        
        merged.sort_values('observation_date', inplace=True)
        
        out_path = os.path.join(base_dir, 'm2/KOR/merged_daily_liquid.csv')
        merged.to_csv(out_path, index=False)
        print(f"Succefully saved {out_path}")
        print(merged.head())
        print(merged.tail())
        print("Total matching daily points:", len(merged))

