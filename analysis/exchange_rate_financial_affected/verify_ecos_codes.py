import os
import requests
from dotenv import load_dotenv
import sys

# Set encoding for Windows terminal
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

load_dotenv()
API_KEY = os.getenv("ECOS_API_KEY")

def get_items(stat_code):
    url = f"http://ecos.bok.or.kr/api/StatisticItemList/{API_KEY}/json/kr/1/500/{stat_code}/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "StatisticItemList" in data:
            return data["StatisticItemList"]["row"]
        else:
            print(f"Error fetching items for {stat_code}: {data}")
    return []

# Check BOP (301Y013)
print(f"--- Items for 301Y013 (BOP) ---")
items = get_items("301Y013")
for i in items:
    print(f"{i['ITEM_CODE']}: {i['ITEM_NAME']}")

# Check KOSPI (901Y102 or similar)
# Some sources suggest 901Y001 or 901Y002
print(f"\n--- Items for 901Y002 (Stocks/Bonds Market) ---")
items = get_items("901Y002")
for i in items:
    print(f"{i['ITEM_CODE']}: {i['ITEM_NAME']}")

# Check BSI (041Y001)
print(f"\n--- Items for 041Y001 (BSI) ---")
items = get_items("041Y001")
for i in items:
    print(f"{i['ITEM_CODE']}: {i['ITEM_NAME']}")

# Check CSI (043Y001)
print(f"\n--- Items for 043Y001 (CSI) ---")
items = get_items("043Y001")
for i in items:
    print(f"{i['ITEM_CODE']}: {i['ITEM_NAME']}")
