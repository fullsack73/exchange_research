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

codes = ["521Y001", "511Y002", "041Y001", "043Y001", "041Y011", "043Y070"]
results = {}

for c in codes:
    results[c] = get_items(c)

with open("ecos_item_discovery_v2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved item discovery v2 to ecos_item_discovery_v2.json")
