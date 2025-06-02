"""
pull_all.py
"""


import requests, json, gzip, pathlib, datetime as dt, tqdm

BASE = "https://clinicaltrials.gov/api/v2/studies"
RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def pull(page_size=1_000):
    token = ""
    today = dt.date.today().isoformat()
    batch = 0
    bar = tqdm.tqdm(desc="CT.gov batches")
    while True:
        url = f"{BASE}?pageSize={page_size}&format=json" + (f"&pageToken={token}" if token else "")
        js  = requests.get(url, timeout=120).json()
        batch += 1
        with gzip.open(RAW_DIR / f"{today}_batch{batch:03}.json.gz", "wt") as fp:
            json.dump(js, fp)
        token = js.get("nextPageToken")
        bar.update(1)
        if not token:
            break
    bar.close()
    print("All batches saved to", RAW_DIR)

if __name__ == "__main__":
    pull()
