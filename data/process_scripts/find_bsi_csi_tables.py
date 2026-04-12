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

def get_table_list():
    url = f"http://ecos.bok.or.kr/api/StatisticTableList/{API_KEY}/json/kr/1/500/?"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "StatisticTableList" in data:
            return data["StatisticTableList"]["row"]
    return []

tables = get_table_list()
print("--- BSI Tables ---")
for t in tables:
    if "기업경기실사" in t['STAT_NAME'] or "BSI" in t['STAT_NAME']:
        print(f"{t['STAT_CODE']}: {t['STAT_NAME']}")

print("\n--- CSI Tables ---")
for t in tables:
    if "소비자동향" in t['STAT_NAME'] or "CSI" in t['STAT_NAME']:
        print(f"{t['STAT_CODE']}: {t['STAT_NAME']}")
