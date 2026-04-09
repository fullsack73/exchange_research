import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ECOS_API_KEY")

def get_items(stat_code):
    url = f"http://ecos.bok.or.kr/api/StatisticItemList/{API_KEY}/json/kr/1/500/{stat_code}/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "StatisticItemList" in data:
            return data["StatisticItemList"]["row"]
    return None

# Test 512Y001
items = get_items("512Y001")
if items:
    for i in items:
        print(f"{i['ITEM_CODE']}: {i['ITEM_NAME']} ({i['CYCLE']})")
else:
    print("512Y001 failed")
