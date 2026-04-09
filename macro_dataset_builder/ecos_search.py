import requests

API_KEY = "N1R89SK6XDA2XE9XLGVY"

def get_stat_table_list():
    url = f"http://ecos.bok.or.kr/api/StatisticTableList/{API_KEY}/json/kr/1/5000/"
    try:
        resp = requests.get(url, timeout=10)
        return resp.json().get('StatisticTableList', {}).get('row', [])
    except Exception as e:
        print("Error fetching tables:", e)
        return []

def search(keywords):
    tables = get_stat_table_list()
    if not tables:
        return
        
    for kw in keywords:
        matched = [t for t in tables if kw in t.get('STAT_NAME', '')]
        print(f"\n=== Keyword: {kw} (Found: {len(matched)}) ===")
        # Print first 10 matches to avoid overwhelming output
        for m in matched[:10]:
            print(f"[{m['STAT_CODE']}] {m['STAT_NAME']}")
            
if __name__ == "__main__":
    search(["100대", "수출입", "무역", "경상", "소비자물가지수", "수입물가지수", "산업생산", "실업", "M2", "MMF", "통화", "예금", "기준금리"])
