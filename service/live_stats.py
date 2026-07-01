import json
import glob
import os
import time
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CATEGORY_PATTERN = os.path.join(BASE_DIR, "data", "steam_appid", "target_appids_*.jsonl")

CATEGORY_KR = {
    'action_fps': 'FPS',
    'action_tps': 'TPS',
    'hack_and_slash': '핵앤슬래시',
    'arcade_rhythm': '리듬/아케이드',
    'action_run_jump': '런앤점프',
    'shmup': '슈팅게임',
    'fighting_martial_arts': '격투게임',
}


def load_all_appids():
    """카테고리별 {appid, title} 목록 로드"""
    result = {}
    for path in glob.glob(CATEGORY_PATTERN):
        category = os.path.basename(path).replace("target_appids_", "").replace(".jsonl", "")
        result[category] = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    result[category].append({
                        'appid': str(item['appid']),
                        'title': item.get('title', f"Game_{item['appid']}")
                    })
                except:
                    pass
    return result


def get_current_players(appid):
    """Steam API로 현재 동시접속자 수 조회 (API 키 불필요)"""
    url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={appid}"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        if data.get('response', {}).get('result') == 1:
            return data['response']['player_count']
    except Exception:
        pass
    return None


def get_official_name(appid):
    """appdetails API로 실제 게임명 조회 (실패 시 None)"""
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=koreana&filters=basic"
    try:
        res = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        data = res.json()
        if data and data.get(str(appid), {}).get('success'):
            return data[str(appid)]['data']['name']
    except Exception:
        pass
    return None


def fetch_top_games(selected_categories, top_n=20, progress_callback=None):
    """선택한 카테고리에서 실시간 접속자 상위 top_n 게임 반환"""
    all_appids = load_all_appids()

    seen = set()
    flat_list = []
    for cat in selected_categories:
        for g in all_appids.get(cat, []):
            if g['appid'] not in seen:
                seen.add(g['appid'])
                flat_list.append({**g, 'category': cat})

    results = []
    total = len(flat_list)

    for i, game in enumerate(flat_list):
        if progress_callback:
            progress_callback(i, total, game['title'])

        count = get_current_players(game['appid'])
        if count is not None:
            results.append({
                'appid': game['appid'],
                'title': game['title'],
                'category': game['category'],
                'category_kr': CATEGORY_KR.get(game['category'], game['category']),
                'player_count': count,
            })
        time.sleep(0.2)

    results.sort(key=lambda x: -x['player_count'])
    top = results[:top_n]

    # 상위 게임만 실제 게임명 조회 (플레이스홀더 Game_xxx → 진짜 이름)
    for g in top:
        if g['title'].startswith('Game_'):
            real = get_official_name(g['appid'])
            if real:
                g['title'] = real
            time.sleep(0.1)

    return top
