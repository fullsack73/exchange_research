import requests
import json
import codecs

API_KEY = "N1R89SK6XDA2XE9XLGVY"

def get_items(stat_code):
    url = f"http://ecos.bok.or.kr/api/StatisticItemList/{API_KEY}/json/kr/1/100/{stat_code}/"
    resp = requests.get(url, timeout=10)
    return resp.json().get('StatisticItemList', {}).get('row', [])

if __name__ == "__main__":
    codes = ["901Y118", "301Y013", "301Y017", "901Y033", "101Y004", "104Y014"]
    results = {}
    for c in codes:
        items = get_items(c)
        results[c] = [{"ITEM_CODE": i['ITEM_CODE'], "ITEM_NAME": i['ITEM_NAME'], "CYCLE": i.get('CYCLE')} for i in items]
        
    with codecs.open("ecos_items_out.json", "w", "utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
