import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ECOS_API_KEY")

def test_fetch(stat_code, item_code, start_date, end_date):
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1/{stat_code}/M/{start_date}/{end_date}/{item_code}/"
    response = requests.get(url)
    print(f"Testing {stat_code}/{item_code}: {response.status_code}")
    if response.status_code == 200:
        print(response.json())

# KOSPI
test_fetch("802Y001", "0001000", "202401", "202401")
# BSI
test_fetch("041Y001", "0001", "202401", "202401")
# CSI
test_fetch("043Y001", "0000001", "202401", "202401")
