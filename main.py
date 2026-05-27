import time
from lib.SteamAppIDExtractor import SteamAppIDExtractor
from lib.SteamMassiveReviewCrawler import SteamMassiveReviewCrawler
from lib.ReviewPreprocessor import ReviewPreprocessor 

if __name__ == "__main__":
    crawler = SteamMassiveReviewCrawler()

    # 장르 리스트 
    target_categories = [
        "action_fps", 
        "action_tps", 
        "hack_and_slash", 
        "arcade_rhythm", 
        "action_run_jump", 
        "shmup", 
        "fighting_martial_arts"
    ]

    completed_ids = crawler.load_completed_appids()

    # ==========================================
    # 1. 데이터 수집 단계 (Crawling)
    # ==========================================
    for category in target_categories:
        print(f"\n{'='*50}\n[{category}] 장르 수집 시작\n{'='*50}")
        
        # [수정 1] 장르마다 브라우저를 새로 엽니다.
        extractor = SteamAppIDExtractor() 
        target_games = []
        
        try:
            # 셀레니움을 이용해 해당 카테고리의 게임 목록만 빠르게 추출합니다.
            target_games = extractor.get_app_ids(category_keyword=category, max_scrolls=1000)
        except Exception as e:
            print(f"[에러 발생] {category} AppID 추출 중 문제 발생: {e}")
        finally:
            # [수정 2] 핵심 포인트: 목록 추출이 끝났으므로, 리뷰를 수집하기 전에 브라우저를 즉시 닫습니다!
            # 이렇게 하면 긴 시간 리뷰를 수집하는 동안 브라우저가 방치되어 에러가 날 일도 없고, 메모리도 절약됩니다.
            extractor.close()

        # [수정 3] 브라우저가 꺼진 상태에서 가볍게 API(requests)로만 리뷰를 수집합니다.
        for game in target_games:
            appid = game['appid']
            title = game['title']
            
            if appid in completed_ids:
                print(f"[스킵] '{title}'(은)는 이미 수집이 완료된 게임입니다.")
                continue
                
            crawler.fetch_and_save_reviews(appid, title)
            crawler.save_completed_appid(appid)
            time.sleep(2) 

    print(f"\n[크롤링 완료] 데이터가 '{crawler.dataset_file}'에 안전하게 저장되었습니다.")

    # ==========================================
    # 2. 데이터 전처리 단계 (Preprocessing)
    # ==========================================
    preprocessor = ReviewPreprocessor(input_file=crawler.dataset_file)
    processed_df = preprocessor.run_pipeline()

    if processed_df is not None:
        print(f"\n [ALL CLEAR] 수집부터 전처리까지 전체 파이프라인이 성공적으로 종료되었습니다!")