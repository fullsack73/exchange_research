import requests
import json
import codecs

API_KEY = "N1R89SK6XDA2XE9XLGVY"

def get_items(stat_code):
    url = f"http://ecos.bok.or.kr/api/StatisticItemList/{API_KEY}/json/kr/1/100/{stat_code}/"
    resp = requests.get(url, timeout=10)
    items = resp.json().get('StatisticItemList', {}).get('row', [])
    print(f"\n--- Items for {stat_code} ---")
    for item in items[:20]: # print first 20
        print(f"[{item['ITEM_CODE']}] {item['ITEM_NAME']} (Cycle: {item.get('CYCLE')})")

if __name__ == "__main__":
    codes = ["901Y118", "301Y013", "301Y017", "901Y033", "101Y004", "104Y014"]
    for c in codes:
        get_items(c)
