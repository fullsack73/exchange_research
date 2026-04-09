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
    return []

# 1. KOSPI
print("--- Items for 802Y001 (KOSPI) ---")
items = get_items("802Y001")
for i in items:
    print(f"{i['ITEM_CODE']}: {i['ITEM_NAME']}")

# 2. BSI
# BSI might have different codes: 041Y001, 041Y013, etc.
print("\n--- Items for 041Y001 (BSI) ---")
items = get_items("041Y001")
for i in items:
    print(f"{i['ITEM_CODE']}: {i['ITEM_NAME']}")

# 3. CSI
print("\n--- Items for 043Y001 (CSI) ---")
items = get_items("043Y001")
for i in items:
    print(f"{i['ITEM_CODE']}: {i['ITEM_NAME']}")
