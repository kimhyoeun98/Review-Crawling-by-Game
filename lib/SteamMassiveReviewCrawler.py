import os
import requests
import json
from urllib.parse import quote
import time

class SteamMassiveReviewCrawler:
    """AppID를 기반으로 조회하고 리뷰를 수집하여 JSONL로 저장하는 모듈"""
    def __init__(self):
        # 리뷰 API
        self.api_url = "https://store.steampowered.com/appreviews/{}?json=1&language=koreana&filter=recent&num_per_page=100&cursor={}"
        # 게임 정보 API 
        self.app_details_url = "https://store.steampowered.com/api/appdetails?appids={}&l=koreana"
        
        self.dataset_file = "./data/steam_raw_reviews.jsonl"
        self.checkpoint_file = "./data/completed_appids.txt"

    # 이미 수집된 AppID를 추적하여 중복 저장 방지
    def get_official_name(self, appid):
        """AppID를 이용해 스팀 공식 서버에서 실제 게임 이름을 가져옵니다."""
        try:
            # API 호출하여 게임 이름 가져오기 (한글로 요청)
            url = self.app_details_url.format(appid)
            # API 응답에서 게임 이름 추출 (성공 여부 확인 후)
            res = requests.get(url, timeout=10)
            # API 호출이 성공적이고 데이터가 유효한 경우에만 이름 반환, 그렇지 않으면 기본 이름 사용
            data = res.json()
            # API 응답이 성공적이고 데이터가 유효한 경우에만 이름 반환, 그렇지 않으면 기본 이름 사용
            if data and data[str(appid)]['success']:
                return data[str(appid)]['data']['name']
        except Exception as e:
            print(f"      [경고] 이름을 가져오지 못함 (AppID: {appid}): {e}")
        return f"Game_{appid}"
    
    # 체크포인트 파일에서 이미 수집된 AppID를 불러오고, 새로운 AppID를 추가하여 저장하는 메서드
    def load_completed_appids(self):
        if not os.path.exists(self.checkpoint_file):
            return set()
        with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
        
    # 새로운 AppID를 체크포인트 파일에 추가하여 저장하는 메서드
    def save_completed_appid(self, appid):
        with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(f"{appid}\n")
            
    # AppID를 기반으로 리뷰를 수집하여 JSONL 파일에 저장하는 메서드
    def fetch_and_save_reviews(self, appid, title, category=None):
        official_name = self.get_official_name(appid)
        print(f" -> '{official_name}' (AppID: {appid}) 수집 시작...")
        
        cursor = '*'
        total_collected = 0
        # API를 반복 호출하여 모든 리뷰를 수집 (cursor 기반 페이지네이션)
        while True:
            safe_cursor = quote(cursor)
            url = self.api_url.format(appid, safe_cursor)
            # API 호출 및 리뷰 수집 시도, 에러 발생 시 루프 탈출
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    break
                # API 응답이 성공적이고 데이터가 유효한 경우에만 리뷰 처리, 그렇지 않으면 루프 탈출
                data = response.json()
                if data.get('query_summary', {}).get('num_reviews', 0) == 0:
                    break 
                # 리뷰 데이터가 존재하는 경우에만 저장 시도
                reviews_batch = []
                for review in data.get('reviews', []):
                    if review['review'].strip():
                        review_data = {
                            "appid": str(appid),
                            "게임 이름": official_name,
                            "카테고리": category,
                            "작성자": review['author']['steamid'],
                            "리뷰 내용": review['review'].strip(),
                            "평점": "추천" if review['voted_up'] else "비추천"
                        }
                        reviews_batch.append(review_data)
                # 리뷰 데이터가 존재하는 경우에만 JSONL 파일에 저장 시도
                if reviews_batch:
                    with open(self.dataset_file, 'a', encoding='utf-8') as f:
                        for item in reviews_batch:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
                    total_collected += len(reviews_batch)
                    print(f"    ...현재까지 {total_collected}개 저장 완료...")

                new_cursor = data.get('cursor')
                if new_cursor == cursor:
                    break
                cursor = new_cursor
                time.sleep(1.2)

            except Exception as e:
                print(f"수집 중 에러 발생: {e}")
                break

        print(f" === '{official_name}' 수집 종료 (총 {total_collected}개 확보) ===")