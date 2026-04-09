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
    # Fetch top 500 tables
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
    return []

tables = get_table_list()
for t in tables:
    name = t['STAT_NAME']
    code = t['STAT_CODE']
    if any(kw in name for kw in ["KOSPI", "코스피", "기업경기", "소비자동향", "심리지수"]):
        print(f"[{code}] {name}")
        items = get_items(code)
        for i in items:
            if any(ikw in i['ITEM_NAME'] for ikw in ["지수", "Index", "CCSI", "업황"]):
                print(f"  - {i['ITEM_CODE']}: {i['ITEM_NAME']}")
