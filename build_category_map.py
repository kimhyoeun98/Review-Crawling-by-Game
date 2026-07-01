"""
Steam GetAppList API로 appid→공식명 매핑을 단번에 구성합니다.
출력: data/game_category_map.json  (game_name → category)
"""
import json
import glob
import os
import requests

APPID_PATTERN = "./data/steam_appid/target_appids_*.jsonl"
COMPLETED_FILE = "./data/completed_appids.txt"
OUT_FILE = "./data/game_category_map.json"
STEAM_APPLIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"


def load_completed_appids():
    if not os.path.exists(COMPLETED_FILE):
        return set()
    with open(COMPLETED_FILE, encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def load_appid_category_map():
    """appid → category 매핑 (로컬 jsonl 파일 기반)"""
    appid_cat = {}
    for path in glob.glob(APPID_PATTERN):
        category = os.path.basename(path).replace("target_appids_", "").replace(".jsonl", "")
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    appid = str(item['appid'])
                    if appid not in appid_cat:
                        appid_cat[appid] = category
                except Exception:
                    pass
    return appid_cat


APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails?appids={}&l=koreana"

def get_official_name(appid, session, retries=3):
    """appid → 공식 게임명. 일시적 실패(429/타임아웃)는 재시도."""
    import time
    for attempt in range(retries):
        try:
            res = session.get(APP_DETAILS_URL.format(appid), timeout=15)
            if res.status_code == 429:  # rate limit → 백오프 후 재시도
                time.sleep(2 + attempt * 2)
                continue
            if res.status_code != 200:
                return None
            data = res.json()
            if data and data.get(str(appid), {}).get('success'):
                return data[str(appid)]['data']['name']
            return None  # success=False (지역제한/삭제 게임 등)
        except Exception:
            time.sleep(1 + attempt)
    return None


def build():
    import time
    os.makedirs("./data", exist_ok=True)

    # 기존 매핑 로드 (이어하기)
    game_category_map = {}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding='utf-8') as f:
            game_category_map = json.load(f)
        print(f"기존 매핑 {len(game_category_map)}개 로드 (이어하기)")

    completed = load_completed_appids()
    print(f"완료된 appid: {len(completed)}개")

    appid_cat = load_appid_category_map()
    print(f"카테고리 매핑 로드: {len(appid_cat)}개 appid")

    already_done = set(game_category_map.keys())
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    to_fetch = [(appid, appid_cat[appid]) for appid in completed if appid in appid_cat]
    total = len(to_fetch)
    print(f"Steam API 조회 시작: {total}개 appid\n")

    for i, (appid, cat) in enumerate(to_fetch, 1):
        name = get_official_name(appid, session)
        if name and name not in already_done:
            game_category_map[name] = cat
            already_done.add(name)

        if i % 20 == 0 or i == total:
            with open(OUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(game_category_map, f, ensure_ascii=False, indent=2)
            print(f"  [{i}/{total}] 저장 ({len(game_category_map)}개 매핑)")

        time.sleep(0.3)

    print(f"\n매핑 완료: {len(game_category_map)}개 게임")
    missing = total - len(game_category_map)
    if missing > 0:
        print(f"조회 실패: {missing}개 (DLC/패키지 등)")

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(game_category_map, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {OUT_FILE}")

    # 카테고리별 분포 출력
    from collections import Counter
    dist = Counter(game_category_map.values())
    print("\n카테고리별 게임 수:")
    for cat, cnt in dist.most_common():
        print(f"  {cat}: {cnt}개")


if __name__ == "__main__":
    build()
