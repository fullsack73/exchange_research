import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()
API_KEY = os.getenv("ECOS_API_KEY")

def get_items(stat_code):
    url = f"http://ecos.bok.or.kr/api/StatisticItemList/{API_KEY}/json/kr/1/500/{stat_code}/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

codes = ["802Y001", "901Y002", "041Y001", "043Y001", "301Y013"]
results = {}

for c in codes:
    results[c] = get_items(c)

with open("ecos_item_discovery.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved item discovery to ecos_item_discovery.json")
