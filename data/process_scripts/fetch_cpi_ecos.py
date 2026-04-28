import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ECOS API Configuration
API_KEY = os.getenv('ECOS_API_KEY')
SERVICE_URL = 'https://ecos.bok.or.kr/api/StatisticSearch/API_KEY/json/kr/1/10000/901Y144/'

# Output paths
output_path = '/Applications/dollar_price/data/CPI/KOR/ECOS_KOR_CPI_processed.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def fetch_cpi():
    print("Fetching CPI data from ECOS API...")

    try:
        query_url = SERVICE_URL.replace('API_KEY', API_KEY)
        response = requests.get(query_url)
        if response.status_code == 200:
            data = response.json()
            cpi_records = data['StatisticSearch']['row']
            df = pd.DataFrame(cpi_records)

            # Process data frame
            df_processed = df[['TIME', 'DATA']]
            df_processed.rename(columns={'TIME': 'observation_date', 'DATA': 'CPI'}, inplace=True)

            # Save to CSV
            df_processed.to_csv(output_path, index=False)
            print(f"Saved CPI data to {output_path}")
        else:
            print(f"Failed to fetch CPI. HTTP {response.status_code}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_cpi()