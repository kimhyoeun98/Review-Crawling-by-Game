import time
import os
import json  
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException

class SteamAppIDExtractor:
    def __init__(self):
        self.base_url = "https://store.steampowered.com/category/"
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless') # 필요에 따라 주석 해제하여 헤드리스 모드로 실행 가능
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        
        # 공유 메모리 사용 비활성화
        options.add_argument('--disable-dev-shm-usage') 
        
        # 이미지 로딩 비활성화 (속도 향상)
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)

        self.driver = webdriver.Chrome(options=options)
    
    # 특정 장르 페이지에서 최고 인기 순으로 AppID와 게임 제목을 추출하는 메서드
    def get_app_ids(self, category_keyword, max_scrolls=1000, target_count=1000):
        # 카테고리 페이지 URL 구성 (최고 인기 제품으로 필터링)
        url = f"{self.base_url}{category_keyword}?flavor=contenthub_topsellers"
        print(f">>> '{category_keyword}' 카테고리 (최고 인기 제품) AppID 추출 시작... (목표: {target_count}개)")
        self.driver.get(url)
        time.sleep(4) 

        scroll_count = 0
        app_list = []
        seen_ids = set()
        backup_filename = f"target_appids_{category_keyword}.jsonl"
        
        # 백업 파일 초기화 (기존 덮어쓰기)
        with open(backup_filename, 'w', encoding='utf-8') as f:
            pass 

        wait = WebDriverWait(self.driver, 10)

        # 목표 개수(target_count)를 채울 때까지 반복
        while len(seen_ids) < target_count:
            # 1. 현재 로딩된 게임 요소 추출
            games = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/app/']")
            new_batch = []
            
            for game in games:
                try:
                    href = game.get_attribute('href')
                    if not href: continue
                    
                    # href에서 AppID 추출
                    appid = href.split('/app/')[1].split('/')[0]
                    
                    if appid.isdigit() and appid not in seen_ids:
                        seen_ids.add(appid)

                        title = ""
                        try:
                            # 1. 먼저 게임 제목이 들어있는 요소에서 시도 (대소문자 구분 없이 Title 또는 title 클래스 탐색)
                            title_elems = game.find_elements(By.CSS_SELECTOR, "div[class*='Title'], div[class*='title'], span.title")
                            if title_elems:
                                title = title_elems[0].text.strip()
                        except:
                            pass

                        if not title:
                            try:
                                # 2. 그래도 없으면 이미지 요소의 alt 속성에서 시도
                                img = game.find_element(By.TAG_NAME, 'img')
                                title = img.get_attribute('alt')
                            except:
                                pass
                        
                        if not title:
                            # 최후의 수단으로 AppID를 제목으로 사용
                            title = f"Game_{appid}"
                        
                        item = {
                            'appid': appid, 
                            'title': title,
                            'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        new_batch.append(item)
                        app_list.append(item)

                        # 목표 개수 도달 시 즉시 for문 중단
                        if len(seen_ids) >= target_count:
                            break 
                except:
                    continue

            # 새로 발견된 게임 JSONL 백업
            if new_batch:
                with open(backup_filename, 'a', encoding='utf-8') as f:
                    for item in new_batch:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"    ... 데이터 저장 완료 | 현재까지 {len(seen_ids)} / {target_count}개 발견")

            # 목표 개수에 도달했으면 while문 탈출
            if len(seen_ids) >= target_count:
                break

            # 2. 'Show more' / '더 보기' 버튼 찾아서 클릭하기
            try:
                # 대소문자 상관없이 Show more 또는 더 보기 버튼 찾기
                show_more_button = wait.until(EC.presence_of_element_located((
                    By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'show more') or contains(., '더 보기')]"
                )))
                
                # 버튼이 있는 곳으로 화면을 스크롤하여 이동
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_more_button)
                time.sleep(1) # 스크롤 애니메이션 대기
                
                # 버튼 강제 클릭 (javascript 이벤트로 인터셉트 우회)
                self.driver.execute_script("arguments[0].click();", show_more_button)
                print(f"    ... 'Show more' 버튼 클릭 (누적 {scroll_count+1}회)")
                
                time.sleep(3) # 클릭 후 새 게임 목록이 서버에서 로딩될 때까지 대기
                
            except TimeoutException:
                print("    ... 'Show more' 버튼이 더 이상 없습니다. 페이지 끝 도달.")
                break
            except Exception as e:
                print(f"    ... 버튼 클릭 중 알 수 없는 에러 발생: {e}")
                break

            scroll_count += 1
            if max_scrolls is not None and scroll_count >= max_scrolls:
                 break 

        print(f"--- 총 {len(app_list)}개의 AppID 추출 완료! (JSONL 백업: '{backup_filename}') ---\n")
        return app_list
    
    def close(self):
        self.driver.quit()