import requests
import json
import codecs

API_KEY = "N1R89SK6XDA2XE9XLGVY"

def search():
    url = f"http://ecos.bok.or.kr/api/StatisticTableList/{API_KEY}/json/kr/1/5000/"
    resp = requests.get(url, timeout=10)
    tables = resp.json().get('StatisticTableList', {}).get('row', [])
    
    keywords = ["100대", "수출입", "무역", "경상", "소비자물가", "수입물가", "산업생산", "실업", "M2", "요구불", "예금", "기준금리", "MMF", "수출", "수입"]
    results = {}
    
    for kw in keywords:
        matched = [t for t in tables if kw in t.get('STAT_NAME', '')]
        results[kw] = [{"STAT_CODE": m['STAT_CODE'], "STAT_NAME": m['STAT_NAME']} for m in matched]
        
    with codecs.open("ecos_codes.json", "w", "utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    search()
