import os
import requests
from dotenv import load_dotenv
import json
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

# 1. Search for BOP (Balance of Payments) and find Foreign Investment
print("--- Searching for BOP (국제수지) ---")
tables = get_table_list()
for t in tables:
    if "국제수지" in t['STAT_NAME'] and "월" in t['STAT_NAME']:
        print(f"Table: {t['STAT_CODE']} - {t['STAT_NAME']}")
        items = get_items(t['STAT_CODE'])
        for i in items:
            if "증권투자" in i['ITEM_NAME'] or "지분증권" in i['ITEM_NAME'] or "부채성증권" in i['ITEM_NAME']:
                print(f"  Item: {i['ITEM_CODE']} - {i['ITEM_NAME']}")

# 2. Search for KOSPI
print("\n--- Searching for KOSPI (코스피) ---")
for t in tables:
    if "코스피" in t['STAT_NAME'] or "주가지수" in t['STAT_NAME']:
        print(f"Table: {t['STAT_CODE']} - {t['STAT_NAME']}")
        items = get_items(t['STAT_CODE'])
        for i in items:
            if "코스피" in i['ITEM_NAME'] or "KOSPI" in i['ITEM_NAME']:
                print(f"  Item: {i['ITEM_CODE']} - {i['ITEM_NAME']}")

# 3. Search for BSI
print("\n--- Searching for BSI (기업경기실사) ---")
for t in tables:
    if "기업경기실사" in t['STAT_NAME'] or "BSI" in t['STAT_NAME']:
        print(f"Table: {t['STAT_CODE']} - {t['STAT_NAME']}")
        # Limit item printing for BSI as it might be a lot
        # items = get_items(t['STAT_CODE'])
        # for i in items[:10]:
        #     print(f"  Item: {i['ITEM_CODE']} - {i['ITEM_NAME']}")

# 4. Search for CSI
print("\n--- Searching for CSI (소비자동향) ---")
for t in tables:
    if "소비자동향" in t['STAT_NAME'] or "CSI" in t['STAT_NAME']:
        print(f"Table: {t['STAT_CODE']} - {t['STAT_NAME']}")
        # items = get_items(t['STAT_CODE'])
        # for i in items:
        #     print(f"  Item: {i['ITEM_CODE']} - {i['ITEM_NAME']}")

